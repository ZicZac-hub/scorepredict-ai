"""Capture timestamped pre-kickoff predictions and verified team news.

Run via GitHub Actions. Each artifact is immutable evidence of what was
available at collection time. Never retroactively fetch team news for
completed fixtures and call it pre-match evidence.
"""
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
import requests
from api_football import LEAGUES, _convert, ROOT
from engine import predict
from team_news import pre_match_report

CODES = ("PL", "PD", "BL1", "SA", "FL1")
MAX_FIXTURES = 12
OUTPUT = Path("prematch_snapshots")

def request_fixtures(key, params):
    response = requests.get(ROOT + "/fixtures", params=params,
                            headers={"x-apisports-key": key}, timeout=25)
    response.raise_for_status()
    payload = response.json()
    if payload.get("errors"):
        raise RuntimeError(str(payload["errors"])[:200])
    return payload.get("response") or []

def main():
    key = os.environ.get("API_FOOTBALL_KEY", "")
    if not key:
        raise RuntimeError("API_FOOTBALL_KEY secret required")
    now = datetime.now(timezone.utc)
    candidates = []
    for code in CODES:
        season = now.year if now.month >= 7 else now.year - 1
        fixtures = request_fixtures(key, {"league": LEAGUES[code], "season": season,
                    "from": now.date().isoformat(),
                    "to": (now + timedelta(days=2)).date().isoformat()})
        upcoming = []
        for raw in fixtures:
            fixture = _convert(raw, code)
            try:
                kickoff = datetime.fromisoformat(fixture["utcDate"].replace("Z","+00:00"))
            except (ValueError, TypeError):
                continue
            if now < kickoff <= now + timedelta(hours=48) and fixture["status"] in ("SCHEDULED", "TIMED"):
                upcoming.append((kickoff, fixture))
        if not upcoming:
            continue
        # Historical matches must be strictly earlier than the snapshot.
        history_raw = request_fixtures(key, {"league": LEAGUES[code], "season": season,
                    "from": (now-timedelta(days=180)).date().isoformat(),
                    "to": now.date().isoformat(), "status": "FT"})
        history = [m for raw in history_raw if (m:=_convert(raw,code))["status"]=="FINISHED"
                   and datetime.fromisoformat(m["utcDate"].replace("Z","+00:00")) < now]
        for kickoff, fixture in upcoming:
            forecast = predict(fixture, history, variant="baseline")
            if forecast is None:
                continue
            candidates.append((kickoff, fixture, forecast, code))
    candidates.sort(key=lambda item: item[0])
    records = []
    for kickoff, fixture, forecast, code in candidates[:MAX_FIXTURES]:
        report = pre_match_report(key, fixture, now=now)
        records.append({
            "captured_at_utc": now.isoformat(),
            "fixture_id": fixture["id"], "competition": code,
            "kickoff_utc": fixture["utcDate"],
            "home_team": fixture["homeTeam"]["name"],
            "away_team": fixture["awayTeam"]["name"],
            "forecast_model": forecast["model_version"],
            "predicted_score": list(forecast["scores"][0][:2]),
            "predicted_score_probability": forecast["scores"][0][2],
            "predicted_outcome_probabilities": {
                "home": forecast["home_win"], "draw": forecast["draw"],
                "away": forecast["away_win"]},
            "expected_goals": {"home": forecast["home_xg"], "away": forecast["away_xg"]},
            "lineups": report.get("lineups", []),
            "injuries": report.get("injuries", []),
            "provider_warnings": report.get("warnings", []),
        })
    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT / ("prematch_" + now.strftime("%Y%m%dT%H%M%SZ") + ".json")
    path.write_text(json.dumps({"captured_at_utc": now.isoformat(),
                                "records": records,
                                "note": "Immutable pre-kickoff snapshot. Injury reports do not prove absence; no post-kickoff data is included."},
                               indent=2), encoding="utf-8")
    print(f"Saved {len(records)} pre-match snapshots to {path}")

if __name__ == "__main__":
    main()
