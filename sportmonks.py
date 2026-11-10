"""Sportmonks v3 fixture discovery for subscribed competitions only."""
from datetime import datetime, timedelta, timezone
import requests
import streamlit as st
from worldwide import SYDNEY

BASE = "https://api.sportmonks.com/v3/football"

@st.cache_data(ttl=21600, show_spinner=False)
def sportmonks_fixtures(token, start_day, days):
    """Return upcoming Sydney-date fixtures and non-secret diagnostics."""
    if not token:
        return [], ["Sportmonks key is missing."]
    # Sportmonks date range is UTC; add a day either side for Sydney timezone.
    first = (start_day - timedelta(days=1)).isoformat()
    last = (start_day + timedelta(days=days)).isoformat()
    url = f"{BASE}/fixtures/between/{first}/{last}"
    matches, errors = {}, []
    try:
        for page in range(1, 5):  # Protect free quota; 50 fixtures per page.
            response = requests.get(
                url,
                headers={"Authorization": token, "Accept": "application/json"},
                params={"include": "participants;league", "per_page": 50, "page": page},
                timeout=20,
            )
            if response.status_code != 200:
                errors.append(f"Sportmonks HTTP {response.status_code}: check subscription and token.")
                break
            payload = response.json()
            for item in payload.get("data", []):
                stamp = item.get("starting_at_timestamp")
                if not stamp:
                    continue
                kickoff = datetime.fromtimestamp(int(stamp), tz=timezone.utc)
                local = kickoff.astimezone(SYDNEY)
                if not (start_day <= local.date() < start_day + timedelta(days=days)):
                    continue
                if kickoff <= datetime.now(timezone.utc):
                    continue
                teams = {}
                for p in item.get("participants") or []:
                    side = (p.get("meta") or {}).get("location")
                    if side in ("home", "away"):
                        teams[side] = p
                if "home" not in teams or "away" not in teams:
                    continue
                league = item.get("league") or {}
                home, away = teams["home"]["name"], teams["away"]["name"]
                identity = (kickoff.isoformat(), home.casefold(), away.casefold())
                matches[identity] = {
                    "id": f"sportmonks-{item['id']}",
                    "kickoff": kickoff, "local": local,
                    "home": home, "away": away,
                    "league": league.get("name") or "Unknown competition",
                    "country": "Sportmonks",
                    "source": "Sportmonks",
                    "league_id": item.get("league_id"),
                    "season": item.get("season_id"),
                }
            if not (payload.get("pagination") or {}).get("has_more"):
                break
        else:
            errors.append("Sportmonks results capped at 200 fixtures to conserve quota.")
    except (requests.RequestException, ValueError, TypeError, KeyError) as exc:
        errors.append(f"Sportmonks request failed: {type(exc).__name__}.")
    return sorted(matches.values(), key=lambda m: m["kickoff"]), errors
