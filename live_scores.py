"""Live API-Football scores for fixtures already predicted by ZicZac.

Only join on API-Football fixture IDs. Never match different providers' IDs.
"""
import requests

ROOT = "https://v3.football.api-sports.io"
LIVE_STATUSES = {"1H", "HT", "2H", "ET", "BT", "P", "INT", "LIVE"}
FINAL_STATUSES = {"FT", "AET", "PEN"}

def live_predicted_scores(api_key, predicted_rows):
    ids = {str(row["fixture"]["id"]) for row in predicted_rows
           if row.get("result") and row.get("fixture", {}).get("provider") == "API-Football"
           and row["fixture"].get("id") is not None}
    if not ids:
        return [], 0
    response = requests.get(
        ROOT + "/fixtures", params={"live": "all"},
        headers={"x-apisports-key": api_key}, timeout=20)
    response.raise_for_status()
    payload = response.json()
    if payload.get("errors"):
        raise RuntimeError(str(payload["errors"])[:250])
    scores = []
    by_id = {str(row["fixture"]["id"]): row for row in predicted_rows
             if row.get("result") and row.get("fixture", {}).get("provider") == "API-Football"}
    for match in payload.get("response", []):
        fixture = match.get("fixture") or {}
        match_id = str(fixture.get("id"))
        if match_id not in ids:
            continue
        status = fixture.get("status") or {}
        short = status.get("short", "")
        if short not in LIVE_STATUSES and short not in FINAL_STATUSES:
            continue
        original = by_id[match_id]
        top = (original.get("result") or {}).get("scores") or []
        goals = match.get("goals") or {}
        scores.append({
            "home": (match.get("teams") or {}).get("home", {}).get("name", "Home"),
            "away": (match.get("teams") or {}).get("away", {}).get("name", "Away"),
            "predicted": f"{top[0][0]}–{top[0][1]}" if top else "—",
            "live": f"{goals.get('home', '—')}–{goals.get('away', '—')}",
            "status": f"{status.get('elapsed')}'" if status.get("elapsed") is not None and short in LIVE_STATUSES else short,
            "fixture_id": match_id,
        })
    return scores, len(ids)
