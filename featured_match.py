"""Stable featured-match selection, with verified completion checks."""
from datetime import datetime, timezone
from time import time
import requests

FINAL = {"FT", "AET", "PEN"}
CANCELLED = {"CANC", "PST", "ABD", "AWD", "WO"}

def fixture_state(api_key, fixture):
    """Return (state, live_score); unknown is not treated as completed."""
    if fixture.get("provider") != "API-Football" or not api_key or not fixture.get("id"):
        status = str(fixture.get("status", "")).upper()
        return ("finished" if status == "FINISHED" else "unknown"), None
    response = requests.get("https://v3.football.api-sports.io/fixtures",
                            params={"id": fixture["id"]},
                            headers={"x-apisports-key": api_key}, timeout=15)
    response.raise_for_status()
    payload = response.json()
    if payload.get("errors"):
        raise RuntimeError(str(payload["errors"])[:180])
    matches = payload.get("response") or []
    if not matches:
        return "unknown", None
    item = matches[0]
    short = ((item.get("fixture") or {}).get("status") or {}).get("short", "")
    goals = item.get("goals") or {}
    score = (goals.get("home"), goals.get("away"))
    return ("finished" if short in FINAL else "cancelled" if short in CANCELLED
            else "active"), score

def rank_feature(row):
    p = row["result"]
    scores = p.get("scores") or []
    venue = min(len(p.get("home_home10", [])), len(p.get("away_away10", [])))
    recent = min(len(p.get("home_last10", [])), len(p.get("away_last10", [])))
    quality = str(p.get("data_quality", "")).lower()
    outcome = max(p.get("home_win", 0), p.get("draw", 0), p.get("away_win", 0))
    return (venue >= 5 and recent >= 5, quality in ("high", "good"),
            min(venue, 10), min(recent, 10), outcome,
            scores[0][2] if scores else 0)

def choose_feature(rows, session, api_key, now=None):
    """Keep one featured prediction until the provider confirms completion."""
    now = now or datetime.now(timezone.utc)
    available = [r for r in rows if r.get("result") and r["result"].get("scores")]
    saved = session.get("featured_match")
    if saved:
        # Refresh from current predictions if present, otherwise retain original.
        key = saved["key"]
        row = next((r for r in available
                    if f"{r['fixture'].get('provider')}:{r['fixture'].get('id')}" == key),
                   saved["row"])
        check = session.get("featured_status")
        if not check or check.get("key") != key or time()-check["time"] >= 60:
            try:
                state, score = fixture_state(api_key, row["fixture"])
                check = {"key": key, "time": time(), "state": state, "score": score}
                session["featured_status"] = check
            except (requests.RequestException, RuntimeError, ValueError):
                check = {"key": key, "time": time(), "state": "unknown", "score": None}
                session["featured_status"] = check
        if check["state"] not in ("finished", "cancelled"):
            return row, check
        session.pop("featured_match", None)
        session.pop("featured_status", None)
    # Rank all future predictions, not just tomorrow, so the hero never
    # disappears solely because a fixture crosses midnight in Sydney.
    candidates = []
    for row in available:
        try:
            kickoff = datetime.fromisoformat(row["kickoff"].replace("Z", "+00:00"))
            if kickoff.tzinfo is None:
                kickoff = kickoff.replace(tzinfo=timezone.utc)
            if kickoff >= now:
                candidates.append(row)
        except (KeyError, ValueError, TypeError):
            continue
    for row in sorted(candidates, key=rank_feature, reverse=True):
        # Check candidates before selecting: never pick an already-finished fixture.
        try:
            state, score = fixture_state(api_key, row["fixture"])
        except (requests.RequestException, RuntimeError, ValueError):
            state, score = "unknown", None
        if state in ("finished", "cancelled"):
            continue
        key = f"{row['fixture'].get('provider')}:{row['fixture'].get('id')}"
        session["featured_match"] = {"key": key, "row": row}
        check = {"key": key, "time": time(), "state": state, "score": score}
        session["featured_status"] = check
        return row, check
    return None, None
