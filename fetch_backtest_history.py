"""Fetch verified API-Football results for leakage-safe ZicZac backtests.

Uses API_FOOTBALL_KEY from environment (GitHub Actions secret).
Only finished fixtures with valid scores and team IDs are accepted.
No API credentials or downloaded match data are committed to Git.
"""
import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from api_football import LEAGUES, _convert, _request


def collect(key, codes, seasons, max_matches=2500):
    now = datetime.now(timezone.utc)
    fixtures = {}
    issues = []
    for code in codes:
        if code not in LEAGUES:
            issues.append(f"Unsupported league code: {code}")
            continue
        for season in seasons:
            try:
                items = _request(key, "/fixtures", {"league": LEAGUES[code], "season": season})
            except Exception as exc:
                issues.append(f"{code}/{season}: {str(exc)[:160]}")
                continue
            if not isinstance(items, list):
                issues.append(f"{code}/{season}: invalid response")
                continue
            for raw in items:
                m = _convert(raw, code)
                if m["status"] != "FINISHED":
                    continue
                if not m.get("utcDate") or not m.get("id"):
                    continue
                if m["homeTeam"].get("id") is None or m["awayTeam"].get("id") is None:
                    continue
                if not all(type(m["score"]["fullTime"].get(side)) is int for side in ("home", "away")):
                    continue
                if datetime.fromisoformat(m["utcDate"].replace("Z", "+00:00")) >= now:
                    continue
                fixtures[(code, m["id"])] = m
            print(f"{code}/{season}: {len(items)} API fixtures returned", flush=True)
    ordered = sorted(fixtures.values(), key=lambda x: x["utcDate"])
    # Keep oldest-to-newest when capped, preserving historical training context.
    return ordered[-max_matches:], issues


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--leagues", default="PL,PD,BL1,SA,FL1")
    parser.add_argument("--seasons", default="")
    parser.add_argument("--output", default="data/backtest_finished_matches.json")
    parser.add_argument("--max-matches", type=int, default=3500)
    args = parser.parse_args()
    key = os.getenv("API_FOOTBALL_KEY") or os.getenv("API_FOOTBALL_API_KEY")
    if not key:
        print("ERROR: Missing GitHub Actions secret API_FOOTBALL_KEY", file=sys.stderr)
        return 2
    now = datetime.now(timezone.utc)
    seasons = [int(s.strip()) for s in args.seasons.split(",") if s.strip()] if args.seasons else [now.year-1, now.year]
    matches, issues = collect(key, [c.strip() for c in args.leagues.split(",") if c.strip()], seasons, args.max_matches)
    for issue in issues:
        print("Provider warning:", issue, file=sys.stderr)
    if len(matches) < 40:
        print(f"ERROR: Only {len(matches)} verified matches returned; cannot run meaningful backtest", file=sys.stderr)
        return 3
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(matches, indent=2), encoding="utf-8")
    print(f"Saved {len(matches)} verified matches to {target} (not committed)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
