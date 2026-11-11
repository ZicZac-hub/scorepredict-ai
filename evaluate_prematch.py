"""Evaluate genuinely pre-kickoff ZicZac snapshots against later final scores.

Usage:
  python evaluate_prematch.py snapshots/*.json --results data/backtest_finished_matches.json
Only records captured before kickoff qualify. This script audits availability
and baseline accuracy; it does NOT claim injury information improves scores.
"""
import argparse
import glob
import json
from datetime import datetime, timezone
from pathlib import Path

def dt(value):
    d = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return d.replace(tzinfo=timezone.utc) if d.tzinfo is None else d

def evaluate(snapshot_files, finished):
    results = {str(m["id"]): m for m in finished
               if m.get("provider") == "API-Football"
               and m.get("status") == "FINISHED"
               and (m.get("score") or {}).get("fullTime", {}).get("home") is not None}
    earliest = {}
    for file in snapshot_files:
        with open(file, encoding="utf-8") as source:
            payload = json.load(source)
        for row in payload.get("records", []):
            try:
                if dt(row["captured_at_utc"]) >= dt(row["kickoff_utc"]):
                    continue
                key = str(row["fixture_id"])
                if key not in earliest or dt(row["captured_at_utc"]) < dt(earliest[key]["captured_at_utc"]):
                    earliest[key] = row
            except (ValueError, TypeError, KeyError):
                continue
    evaluated = exact = lineup = injuries = 0
    for key, row in earliest.items():
        match = results.get(key)
        if not match:
            continue
        actual = match["score"]["fullTime"]
        evaluated += 1
        exact += int(row["predicted_score"] == [actual["home"], actual["away"]])
        lineup += int(any(item.get("confirmed") for item in row.get("lineups", [])))
        injuries += int(bool(row.get("injuries")))
    return {"pre_kickoff_unique_fixtures": len(earliest),
            "finished_matches_with_snapshots": evaluated,
            "exact_score_hits": exact,
            "exact_score_hit_rate": exact/evaluated if evaluated else None,
            "matches_with_confirmed_lineups": lineup,
            "matches_with_injury_records": injuries,
            "note": "Descriptive baseline audit only. No causal claims or automatic injury adjustments; do not fit on this evaluation set."}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("snapshots", nargs="+")
    parser.add_argument("--results", required=True)
    parser.add_argument("--output", default="prematch_audit.json")
    args = parser.parse_args()
    paths = [p for pattern in args.snapshots for p in glob.glob(pattern)]
    with open(args.results, encoding="utf-8") as source:
        finished = json.load(source)
    report = evaluate(paths, finished)
    Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
