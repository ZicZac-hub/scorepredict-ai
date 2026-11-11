"""Persist immutable pre-kickoff snapshots and reconcile verified final scores.

The earliest pre-kickoff snapshot per fixture is the evaluation baseline.
Later snapshots are retained separately for team-news availability analysis.
"""
import json
import os
from datetime import datetime, timezone
from pathlib import Path
import requests
from api_football import ROOT

ROOT_DIR = Path("data/prematch")
SNAP_DIR = ROOT_DIR / "snapshots"
RESULTS = ROOT_DIR / "verified_results.json"
REPORT = ROOT_DIR / "accuracy_report.json"

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

def reconcile(key, snapshots):
    results = load_json(RESULTS, {})
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
            item = items[0]
            status = ((item.get("fixture") or {}).get("status") or {}).get("short")
            goals = item.get("goals") or {}
            if status not in ("FT", "AET", "PEN") or not isinstance(goals.get("home"), int) or not isinstance(goals.get("away"), int):
                continue
            results[fixture_id] = {
                "fixture_id": int(fixture_id),
                "verified_at_utc": now.isoformat(),
                "status": status,
                "home_goals": goals["home"],
                "away_goals": goals["away"]
            }
        except (requests.RequestException, ValueError, TypeError) as exc:
            print("Could not verify", fixture_id, str(exc)[:120])
    write_json(RESULTS, results)
    return results

def accuracy(snapshots, results):
    total = exact = outcome = 0
    with_lineups = with_injuries = 0
    by_league = {}
    for fixture_id, row in snapshots.items():
        result = results.get(fixture_id)
        if not result:
            continue
        predicted = row.get("predicted_score") or []
        if len(predicted) != 2:
            continue
        actual = [result["home_goals"], result["away_goals"]]
        hit = int(predicted == actual)
        sign = lambda a, b: (a > b) - (a < b)
        correct_outcome = int(sign(*predicted) == sign(*actual))
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
        "winner_draw_accuracy": outcome / total if total else None,
        "confirmed_lineups_in_earliest_snapshots": with_lineups,
        "injury_records_in_earliest_snapshots": with_injuries,
        "by_league": by_league,
        "method": "Earliest strictly pre-kickoff forecast per fixture; verified API-Football final scores; no post-match features used."
    }

def main():
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
