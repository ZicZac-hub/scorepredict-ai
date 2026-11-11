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

CODES = ("PL", "PD", "BL1", "SA", "FL1", "CL", "EL", "DED", "PPL", "BSA", "ELC")
MAX_FIXTURES = 30
LOOKAHEAD_DAYS = 14
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
    diagnostics = {'leagues_checked': 0, 'upcoming_fixtures': 0, 'history_matches': 0, 'prediction_unavailable': 0, 'prediction_errors': [], 'league_errors': [], 'provider_fixture_counts': {}, 'next_kickoff_by_league': {}, 'lookahead_days': LOOKAHEAD_DAYS}
    for code in CODES:
        diagnostics['leagues_checked'] += 1
        season = now.year if code == "BSA" or now.month >= 7 else now.year - 1
        try:
            fixtures = request_fixtures(key, {"league": LEAGUES[code], "season": season,
                        "from": now.date().isoformat(),
                        "to": (now + timedelta(days=LOOKAHEAD_DAYS)).date().isoformat()})
        except (requests.RequestException, RuntimeError) as exc:
            diagnostics["league_errors"].append({"league": code, "stage": "fixtures", "error": str(exc)[:180]})
            continue
        diagnostics["provider_fixture_counts"][code] = len(fixtures)
        upcoming = []
        for raw in fixtures:
            fixture = _convert(raw, code)
            try:
                kickoff = datetime.fromisoformat(fixture["utcDate"].replace("Z","+00:00"))
            except (ValueError, TypeError):
                continue
            if now < kickoff <= now + timedelta(days=LOOKAHEAD_DAYS) and fixture["status"] in ("SCHEDULED", "TIMED"):
                upcoming.append((kickoff, fixture))
        diagnostics['upcoming_fixtures'] += len(upcoming)
        if upcoming:
            diagnostics['next_kickoff_by_league'][code] = min(item[0] for item in upcoming).isoformat()
        if not upcoming:
            continue
        # Historical matches must be strictly earlier than the snapshot.
        try:
            history_raw = request_fixtures(key, {"league": LEAGUES[code], "season": season,
                        "from": (now-timedelta(days=180)).date().isoformat(),
                        "to": now.date().isoformat(), "status": "FT"})
        except (requests.RequestException, RuntimeError) as exc:
            diagnostics["league_errors"].append({"league": code, "stage": "history", "error": str(exc)[:180]})
            continue
        history = [m for raw in history_raw if (m:=_convert(raw,code))["status"]=="FINISHED"
                   and datetime.fromisoformat(m["utcDate"].replace("Z","+00:00")) < now]
        diagnostics['history_matches'] += len(history)
        for kickoff, fixture in upcoming:
            try:
                forecast = predict(fixture, history, variant="baseline")
                if not forecast:
                    diagnostics['prediction_unavailable'] += 1
                    continue
                candidates.append((kickoff, fixture, forecast, code))
            except Exception as exc:
                diagnostics['prediction_errors'].append({'fixture_id': fixture['id'], 'error': str(exc)[:200]})
    candidates.sort(key=lambda item: item[0])
    diagnostics["candidate_predictions"] = len(candidates)
    records = []
    for kickoff, fixture, forecast, code in candidates[:MAX_FIXTURES]:
        try:
            report = pre_match_report(key, fixture, now=now)
        except Exception as exc:
            report = {"lineups": [], "injuries": [], "warnings": ["Team-news request failed: " + str(exc)[:180]]}
        records.append({
            "captured_at_utc": now.isoformat(),
            "fixture_id": fixture["id"], "competition": code,
            "kickoff_utc": fixture["utcDate"],
            "home_team": fixture["homeTeam"]["name"],
            "away_team": fixture["awayTeam"]["name"],
            "forecast_model": forecast.get("model_version", "baseline"),
            "predicted_score": list(forecast["scores"][0][:2]),
            "predicted_score_probability": forecast["scores"][0][2],
            "predicted_outcome_probabilities": {
                "home": forecast["home_win"], "draw": forecast["draw"],
                "away": forecast["away_win"]},
            "expected_goals": {"home": forecast.get("home_xg"), "away": forecast.get("away_xg")},
            "lineups": report.get("lineups", []),
            "injuries": report.get("injuries", []),
            "provider_warnings": report.get("warnings", []),
        })
    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT / ("prematch_" + now.strftime("%Y%m%dT%H%M%SZ") + ".json")
    path.write_text(json.dumps({"captured_at_utc": now.isoformat(),
                                "records": records,
                                "diagnostics": diagnostics,
                                "note": "Immutable pre-kickoff snapshot. Injury reports do not prove absence; no post-kickoff data is included."},
                               indent=2), encoding="utf-8")
    print(f"Saved {len(records)} pre-match snapshots to {path}")
    print("Diagnostics:", json.dumps(diagnostics))

if __name__ == "__main__":
    main()
