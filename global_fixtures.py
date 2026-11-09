"""Quota-aware global API-Football date search. All dates use Sydney local time."""
from datetime import datetime, timedelta, timezone
from api_football import _request, _convert
from worldwide import SYDNEY

def global_fixtures(key, start_day, days, max_calls=16):
    if not key:
        return [], ["API_FOOTBALL_KEY is not configured."]
    # One date request per Sydney calendar day; responses are cached for six hours.
    # The API's date parameter is interpreted in the requested timezone.
    found = {}
    errors = []
    for offset in range(min(days, max_calls)):
        day = (start_day + timedelta(days=offset)).isoformat()
        try:
            payload = _request(key, "/fixtures", {"date": day, "timezone": "Australia/Sydney"})
            for item in payload:
                raw = item.get("fixture") or {}
                status = (raw.get("status") or {}).get("short")
                if status not in ("NS", "TBD"):
                    continue
                league = item.get("league") or {}
                fixture = _convert(item, str(league.get("id") or "unknown"))
                if not fixture.get("utcDate"):
                    continue
                utc = datetime.fromisoformat(fixture["utcDate"].replace("Z","+00:00"))
                if utc <= datetime.now(timezone.utc):
                    continue
                local = utc.astimezone(SYDNEY)
                if local.date().isoformat() != day:
                    continue
                home = fixture["homeTeam"]["name"]
                away = fixture["awayTeam"]["name"]
                identity = (utc.isoformat(),home.casefold(),away.casefold())
                found[identity] = {
                    "id": fixture["id"], "kickoff": utc, "local": local,
                    "home": home, "away": away,
                    "league": league.get("name") or "Unknown league",
                    "country": league.get("country") or "Unknown",
                    "source": "API-Football", "fixture": fixture,
                    "league_id": league.get("id"),
                    "season": league.get("season")
                }
        except Exception as exc:
            errors.append(f"{day}: {str(exc)[:160]}")
            # Do not spend the remaining quota on repeated failed calls.
            if "quota" in str(exc).lower() or "429" in str(exc):
                break
    return sorted(found.values(), key=lambda m:m["kickoff"]), errors
