"""Persist immutable pre-kickoff snapshots and reconcile verified final scores.

The earliest pre-kickoff snapshot per fixture is the evaluation baseline.
Later snapshots are retained separately for team-news availability analysis.
"""
import json
import argparse
import math
import os
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
import requests
from api_football import ROOT, _convert

ROOT_DIR = Path("data/prematch")
SNAP_DIR = ROOT_DIR / "snapshots"
RESULTS = ROOT_DIR / "verified_results.json"
REPORT = ROOT_DIR / "accuracy_report.json"
LEGACY_RESULTS = ROOT_DIR / "legacy_results.json"
RESULT_EVIDENCE_VERSION = 2

def parse_time(value):
    d = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    return d if d.tzinfo else d.replace(tzinfo=timezone.utc)

def load_json(path, default):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))

def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")

def persist_snapshots():
    SNAP_DIR.mkdir(parents=True, exist_ok=True)
    saved = 0
    for source in sorted(Path("prematch_snapshots").glob("prematch_*.json")):
        payload = load_json(source, {})
        for row in payload.get("records", []):
            try:
                fixture_id = int(row["fixture_id"])
                capture = parse_time(row["captured_at_utc"])
                kickoff = parse_time(row["kickoff_utc"])
                if capture >= kickoff:
                    continue
                filename = capture.strftime("%Y%m%dT%H%M%SZ") + ".json"
                destination = SNAP_DIR / str(fixture_id) / filename
                if not destination.exists():
                    write_json(destination, row)
                    saved += 1
            except (ValueError, KeyError, TypeError):
                continue
    return saved

def earliest_snapshots():
    earliest = {}
    for path in SNAP_DIR.glob("*/*.json"):
        row = load_json(path, {})
        try:
            key = str(int(row["fixture_id"]))
            if parse_time(row["captured_at_utc"]) >= parse_time(row["kickoff_utc"]):
                continue
            if key not in earliest or parse_time(row["captured_at_utc"]) < parse_time(earliest[key]["captured_at_utc"]):
                earliest[key] = row
        except (KeyError, ValueError, TypeError):
            continue
    return earliest

def _team_name(value):
    return " ".join(unicodedata.normalize("NFKC", str(value or "")).casefold().split())

def matches_snapshot(match, row, fixture_id):
    """Match source, provider ID, teams and kickoff before accepting a score."""
    if row.get("provider", "API-Football") != "API-Football" or match.get("provider") != "API-Football":
        return False
    try:
        if int(match.get("id")) != int(fixture_id) or int(row.get("fixture_id")) != int(fixture_id):
            return False
        if parse_time(match.get("utcDate")) != parse_time(row.get("kickoff_utc")):
            return False
    except (TypeError, ValueError):
        return False
    for side in ("home", "away"):
        team = match.get(side + "Team") or {}
        saved_id = row.get(side + "_team_id")
        if saved_id is not None:
            if str(saved_id) != str(team.get("id")):
                return False
        elif not _team_name(row.get(side + "_team")) or _team_name(row.get(side + "_team")) != _team_name(team.get("name")):
            return False
    return True

def verified_result(row, result, fixture_id):
    if not isinstance(result, dict) or result.get("evidence_version") != RESULT_EVIDENCE_VERSION or result.get("score_basis") != "regulation":
        return False
    if any(type(result.get(side + "_goals")) is not int or result[side + "_goals"] < 0 for side in ("home", "away")):
        return False
    match = {"id":result.get("fixture_id"), "provider":result.get("provider"),
             "utcDate":result.get("kickoff_utc"),
             "homeTeam":{"id":result.get("home_team_id"), "name":result.get("home_team")},
             "awayTeam":{"id":result.get("away_team_id"), "name":result.get("away_team")}}
    return matches_snapshot(match, row, fixture_id)

def reconcile(key, snapshots):
    stored = load_json(RESULTS, {})
    # Keep old evidence for inspection, but never count its unverified identity
    # or ambiguous extra-time score in a report. Snapshot files stay immutable.
    legacy = load_json(LEGACY_RESULTS, {})
    results = {}
    for fixture_id, result in stored.items():
        row = snapshots.get(fixture_id)
        if row and verified_result(row, result, fixture_id):
            results[fixture_id] = result
        elif fixture_id not in legacy:
            legacy[fixture_id] = result
    if legacy:
        write_json(LEGACY_RESULTS, legacy)
    now = datetime.now(timezone.utc)
    for fixture_id, row in snapshots.items():
        if fixture_id in results or parse_time(row["kickoff_utc"]) > now:
            continue
        try:
            response = requests.get(ROOT + "/fixtures", params={"id": fixture_id},
                                    headers={"x-apisports-key": key}, timeout=25)
            response.raise_for_status()
            payload = response.json()
            if payload.get("errors"):
                print("Provider error for", fixture_id, str(payload["errors"])[:120])
                continue
            items = payload.get("response") or []
            if not items:
                continue
            match = next((_convert(item, row.get("competition", "WORLD"))
                          for item in items if str((item.get("fixture") or {}).get("id")) == str(fixture_id)), None)
            if not match or not matches_snapshot(match, row, fixture_id):
                continue
            goals = match["score"]["fullTime"]
            if match["status"] != "FINISHED" or any(type(goals.get(side)) is not int or goals[side] < 0 for side in ("home", "away")):
                continue
            results[fixture_id] = {
                "fixture_id": int(fixture_id),
                "verified_at_utc": now.isoformat(),
                "provider": "API-Football",
                "evidence_version": RESULT_EVIDENCE_VERSION,
                "score_basis": "regulation",
                "kickoff_utc": match["utcDate"],
                "home_team_id": match["homeTeam"]["id"],
                "away_team_id": match["awayTeam"]["id"],
                "home_team": match["homeTeam"]["name"],
                "away_team": match["awayTeam"]["name"],
                "status": match["provider_status"],
                "home_goals": goals["home"],
                "away_goals": goals["away"]
            }
        except (requests.RequestException, ValueError, TypeError) as exc:
            print("Could not verify", fixture_id, str(exc)[:120])
    write_json(RESULTS, results)
    return results

def accuracy(snapshots, results):
    total = exact = outcome = 0
    outcome_fallbacks = 0
    expected_hits = 0.0
    probability_count = top3_hits = top3_count = 0
    with_lineups = with_injuries = 0
    by_league = {}
    for fixture_id, row in snapshots.items():
        result = results.get(fixture_id)
        if not verified_result(row, result, fixture_id):
            continue
        predicted = row.get("predicted_score") or []
        if len(predicted) != 2:
            continue
        actual = [result["home_goals"], result["away_goals"]]
        hit = int(predicted == actual)
        sign = lambda a, b: (a > b) - (a < b)
        probabilities = row.get("predicted_outcome_probabilities") or {}
        values = [probabilities.get(name) for name in ("home", "draw", "away")]
        if all(type(v) in (int, float) and math.isfinite(v) and 0 <= v <= 1 for v in values) and sum(values) > 0:
            predicted_outcome = (1, 0, -1)[max(range(3), key=lambda i:values[i])]
        else:
            predicted_outcome = sign(*predicted)
            outcome_fallbacks += 1
        correct_outcome = int(predicted_outcome == sign(*actual))
        probability = row.get("predicted_score_probability")
        if type(probability) in (int, float) and math.isfinite(probability) and 0 <= probability <= 1:
            expected_hits += probability
            probability_count += 1
        candidates = row.get("double_top10_scores") or row.get("top10_scores") or []
        if candidates:
            top3_count += 1
            top3_hits += int(any(list(candidate[:2]) == actual for candidate in candidates[:3]))
        total += 1
        exact += hit
        outcome += correct_outcome
        with_lineups += int(any(l.get("confirmed") for l in row.get("lineups", [])))
        with_injuries += int(bool(row.get("injuries")))
        league = row.get("competition", "Unknown")
        segment = by_league.setdefault(league, {"matches": 0, "exact_hits": 0})
        segment["matches"] += 1
        segment["exact_hits"] += hit
    for segment in by_league.values():
        segment["exact_score_accuracy"] = segment["exact_hits"] / segment["matches"]
    return {
        "unique_pre_kickoff_fixtures": len(snapshots),
        "verified_finished_matches": total,
        "exact_score_hits": exact,
        "exact_score_accuracy": exact / total if total else None,
        "expected_exact_hits": expected_hits if probability_count else None,
        "predictions_with_score_probability": probability_count,
        "top3_score_hits": top3_hits,
        "top3_evaluated_matches": top3_count,
        "top3_score_accuracy": top3_hits / top3_count if top3_count else None,
        "winner_draw_accuracy": outcome / total if total else None,
        "winner_draw_modal_score_fallbacks": outcome_fallbacks,
        "confirmed_lineups_in_earliest_snapshots": with_lineups,
        "injury_records_in_earliest_snapshots": with_injuries,
        "by_league": by_league,
        "method": "Earliest strictly pre-kickoff forecast per fixture; provider, teams and kickoff verified; regulation-time scores. Winner/draw uses highest aggregated outcome probability with disclosed legacy modal-score fallback; no post-match features used."
    }

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--snapshots-only',action='store_true')
    args=parser.parse_args()
    if args.snapshots_only:
        print('New snapshots:',persist_snapshots())
        return
    key = os.environ.get("API_FOOTBALL_KEY", "")
    if not key:
        raise RuntimeError("API_FOOTBALL_KEY is required")
    added = persist_snapshots()
    snapshots = earliest_snapshots()
    results = reconcile(key, snapshots)
    report = accuracy(snapshots, results)
    write_json(REPORT, report)
    print("New snapshots:", added)
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()

