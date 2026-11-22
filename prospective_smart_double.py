"""Freeze prospective Smart Double selections and grade verified regulation results."""
from collections import defaultdict
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

from five_double_accuracy import ROOT as V10_ROOT, _final, _score, utc
from persist_prematch import RESULTS, load_json, write_json
from smart_double_ai import MIN_LEAD_SECONDS, SYDNEY, rank_pairs
from verified_result_index import index_verified_results

OUT = Path("data/prematch/smart_double_pairs")
REPORT = Path("data/prematch/smart_double_comparison.json")
SNAPSHOTS = Path("prematch_snapshots")
VERSION = "smart-double-live-decision-v2"


def eligible_record(record):
    try:
        kickoff = utc(record["kickoff_utc"])
        captured = utc(record["captured_at_utc"])
        evidence = utc(record["evidence_as_of_utc"])
        fixture = {
            "id": record["fixture_id"], "provider": record["provider"],
            "homeTeam": {"name": record["home_team"], "id": record["home_team_id"]},
            "awayTeam": {"name": record["away_team"], "id": record["away_team_id"]},
        }
        scores = record["double_top10_scores"]
        if (not evidence <= captured < kickoff
                or fixture["provider"] != "API-Football"
                or record["score_basis"] != "regulation"
                or not record["input_digest"] or len(scores) < 5
                or any(not str(value or "").isdigit() for value in (
                    fixture["id"], fixture["homeTeam"]["id"], fixture["awayTeam"]["id"]))
                or str(fixture["homeTeam"]["id"]) == str(fixture["awayTeam"]["id"])):
            return None
        return {"fixture": fixture, "kickoff": kickoff.isoformat(),
                "captured": captured, "result": {"scores": scores}}
    except (KeyError, TypeError, ValueError):
        return None


def _publish_once(destination, packet):
    destination.parent.mkdir(parents=True, exist_ok=True)
    contents = json.dumps(packet, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with NamedTemporaryFile(
        dir=destination.parent, mode="w", encoding="utf-8", delete=False
    ) as staged:
        temporary = Path(staged.name)
        try:
            staged.write(contents)
            staged.flush()
            os.fsync(staged.fileno())
            destination.hardlink_to(temporary)
            return True
        except FileExistsError:
            return False
        finally:
            temporary.unlink()


def freeze(*, now=None, out=OUT, snapshots=SNAPSHOTS):
    """Select only from already-captured forecasts with both kickoffs still ahead."""
    created = 0
    for path in sorted(Path(snapshots).glob("prematch_*.json")):
        batch = load_json(path, {})
        by_day = defaultdict(list)
        for record in batch.get("records", []):
            row = eligible_record(record)
            if row is not None:
                day = utc(row["kickoff"]).astimezone(SYDNEY).date().isoformat()
                by_day[day].append(row)
        for day, day_rows in sorted(by_day.items()):
            # New decisions have a separate filename from legacy UTC-day records.
            destination = Path(out) / f"sydney-{day}.json"
            if destination.exists():
                continue
            selected = utc(now if now is not None else datetime.now(timezone.utc))
            candidates = rank_pairs([row for row in day_rows if row["captured"] <= selected],
                                    limit=1, now=selected)
            if not candidates:
                continue
            best = candidates[0]
            first, second = best["first"], best["second"]
            completed = utc(now if now is not None else datetime.now(timezone.utc))
            if any((utc(row["kickoff"]) - completed).total_seconds() <= MIN_LEAD_SECONDS
                   for row in (first, second)):
                continue
            packet = {
                "model_version": VERSION, "sydney_date": day, "source_snapshot": path.name,
                "source_max_captured_at_utc": max(first["captured"], second["captured"]).isoformat(),
                "captured_at_utc": completed.isoformat(), "selected_at_utc": completed.isoformat(),
                "estimated_coverage": best["coverage"],
                "five_doubles": [{"score1": pick["Score 1"], "score2": pick["Score 2"],
                                  "probability": pick["Joint"]} for pick in best["doubles"]],
            }
            for side, row in (("first", first), ("second", second)):
                fixture = row["fixture"]
                packet.update({
                    f"{side}_provider": fixture["provider"],
                    f"{side}_fixture_id": str(fixture["id"]),
                    f"{side}_match": f"{fixture['homeTeam']['name']} vs {fixture['awayTeam']['name']}",
                    f"{side}_kickoff_utc": row["kickoff"],
                    f"{side}_home_team_id": str(fixture["homeTeam"]["id"]),
                    f"{side}_away_team_id": str(fixture["awayTeam"]["id"]),
                })
            created += _publish_once(destination, packet)
    return created


def _frozen_legs(pair, path, clock):
    day = pair["sydney_date"]
    captured, selected = utc(pair["captured_at_utc"]), utc(pair["selected_at_utc"])
    source = utc(pair["source_max_captured_at_utc"])
    if path.stem != f"sydney-{day}" or not source <= captured == selected <= clock:
        raise ValueError("invalid_decision_date_or_time")
    legs = [{key: pair[f"{side}_{key}"] for key in (
        "provider", "fixture_id", "home_team_id", "away_team_id", "kickoff_utc")}
        for side in ("first", "second")]
    if (any(leg["provider"] != "API-Football" or any(
            not str(leg[key]).isdigit() for key in ("fixture_id", "home_team_id", "away_team_id"))
            for leg in legs)
            or legs[0]["fixture_id"] == legs[1]["fixture_id"]
            or len({str(leg[key]) for leg in legs for key in ("home_team_id", "away_team_id")}) != 4):
        raise ValueError("invalid_pair_identity")
    for leg in legs:
        kickoff = utc(leg["kickoff_utc"])
        if (kickoff.astimezone(SYDNEY).date().isoformat() != day
                or (kickoff - selected).total_seconds() <= MIN_LEAD_SECONDS):
            raise ValueError("outside_pre_kickoff_window")
    return legs


def evaluate(*, now=None, out=OUT, root=V10_ROOT, prematch=RESULTS.parent):
    clock = utc(now if now is not None else datetime.now(timezone.utc))
    results, conflicts, _ = index_verified_results(root, prematch, clock)
    rows, legacy = [], []
    pending, invalid = 0, 0
    paths = sorted(Path(out).glob("*.json"))
    for path in paths:
        try:
            pair = load_json(path, {})
            if not isinstance(pair, dict):
                raise ValueError("invalid_packet")
            if pair.get("model_version") != VERSION:
                legacy.append(path.stem)
                continue
            legs = _frozen_legs(pair, path, clock)
            picks = [(_score(pick["score1"]), _score(pick["score2"]))
                     for pick in pair["five_doubles"]]
            coverage = float(pair["estimated_coverage"])
            if len(picks) != 5 or len(set(picks)) != 5 or not math.isfinite(coverage) or not 0 < coverage <= 1:
                raise ValueError("invalid_five_doubles")
        except (ValueError, TypeError, KeyError, OSError):
            invalid += 1
            continue
        scores = tuple(_final(leg, results.get((leg["provider"], str(leg["fixture_id"]))), clock)
                       for leg in legs)
        if any(score is None for score in scores):
            pending += 1
            continue
        rows.append({
            "sydney_date": pair["sydney_date"],
            "first_match": pair.get("first_match"), "second_match": pair.get("second_match"),
            "actual1": f"{scores[0][0]}-{scores[0][1]}",
            "actual2": f"{scores[1][0]}-{scores[1][1]}",
            "hit": scores in picks, "estimated_coverage": coverage,
        })
    wins = sum(row["hit"] for row in rows)
    return {
        "frozen_days": len(rows) + pending, "verified_days": len(rows),
        "pending_days": pending, "invalid_archive_files": invalid,
        "wins": wins, "win_rate": wins / len(rows) if rows else None, "rows": rows,
        "result_conflicts_excluded": len(conflicts),
        "legacy_unverified_publication": {"archived_days": len(legacy), "archive_dates": legacy},
        "note": "Rates use timestamped pre-kickoff selections with two matched regulation scores. "
                "Historical files without a selection timestamp have a separate count.",
    }


if __name__ == "__main__":
    created = freeze()
    report = evaluate()
    write_json(REPORT, report)
    print(f"New Smart Double frozen days: {created}")
    print(f"Verified days: {report['verified_days']}, pending days: {report['pending_days']}, "
          f"legacy days: {report['legacy_unverified_publication']['archived_days']}")
