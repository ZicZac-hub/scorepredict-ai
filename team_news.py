"""Verified pre-match team news from API-Football.

Only return data actually provided by the API. Never infer injuries,
confirmed lineups, or player availability from missing records.
"""
import requests
from datetime import datetime, timezone, timedelta

ROOT = "https://v3.football.api-sports.io"

def _fetch(key, path, fixture_id):
    response = requests.get(ROOT + path,
                            params={"fixture": fixture_id},
                            headers={"x-apisports-key": key}, timeout=20)
    response.raise_for_status()
    payload = response.json()
    if payload.get("errors"):
        raise RuntimeError(str(payload["errors"])[:180])
    return payload.get("response") or []

def pre_match_report(key, fixture, now=None):
    if not key or fixture.get("provider") != "API-Football" or not fixture.get("id"):
        return {"available": False, "reason": "Verified team news requires an API-Football fixture ID."}
    now = now or datetime.now(timezone.utc)
    kickoff = datetime.fromisoformat(str(fixture.get("utcDate", "")).replace("Z", "+00:00"))
    if kickoff.tzinfo is None:
        kickoff = kickoff.replace(tzinfo=timezone.utc)
    if kickoff < now:
        return {"available": False, "reason": "Pre-match report is unavailable after kickoff."}
    # Keep API usage bounded: only fetch for the featured fixture within
    # 48 hours of kickoff, not for every worldwide match.
    if kickoff > now + timedelta(hours=48):
        return {"available": False, "reason": "Team news will be checked within 48 hours of kickoff."}
    report = {"available": True, "fixture_id": fixture["id"],
              "checked_at": now.isoformat(), "lineups": [], "injuries": [],
              "warnings": []}
    try:
        lineups = _fetch(key, "/fixtures/lineups", fixture["id"])
        for lineup in lineups:
            team = (lineup.get("team") or {}).get("name", "Unknown")
            starters = [((entry.get("player") or {}).get("name"))
                        for entry in (lineup.get("startXI") or [])]
            starters = [name for name in starters if name]
            if starters:
                report["lineups"].append({"team": team, "formation": lineup.get("formation"),
                                           "starters": starters, "confirmed": len(starters) == 11})
    except (requests.RequestException, RuntimeError, ValueError) as exc:
        report["warnings"].append("Lineups could not be verified: " + str(exc)[:150])
    try:
        injuries = _fetch(key, "/injuries", fixture["id"])
        for injury in injuries[:30]:
            player = (injury.get("player") or {}).get("name")
            if player:
                report["injuries"].append({"team": (injury.get("team") or {}).get("name", "Unknown"),
                                           "player": player,
                                           "reason": (injury.get("player") or {}).get("reason") or "Not specified"})
    except (requests.RequestException, RuntimeError, ValueError) as exc:
        report["warnings"].append("Injury records could not be verified: " + str(exc)[:150])
    return report
