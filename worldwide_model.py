"""On-demand, quota-aware historical modelling for API-Football fixtures."""
from datetime import datetime, timedelta, timezone
from api_football import _request, _convert
from engine import predict

def model_worldwide(key, event):
    fixture=event.get("fixture")
    if not key or not fixture or fixture.get("provider")!="API-Football":
        return None, "This match needs API-Football fixture IDs and accessible team history."
    home=(fixture.get("homeTeam") or {}).get("id")
    away=(fixture.get("awayTeam") or {}).get("id")
    if not home or not away:
        return None, "Team IDs are missing."
    try:
        kickoff=datetime.fromisoformat(fixture["utcDate"].replace("Z","+00:00"))
        # Free plans can reject 'last'. Use a date range for the fixture's league
        # and season, then keep only historical matches involving either team.
        league_id=event.get("league_id")
        season=event.get("season")
        if not league_id or not season:
            return None, "League or season ID missing from this fixture."
        end=(min(kickoff,datetime.now(timezone.utc))-timedelta(days=1)).date()
        start=end-timedelta(days=125)
        params={"league":league_id,"season":season,"from":start.isoformat(),"to":end.isoformat()}
        past=[]
        for raw in _request(key,"/fixtures",params):
            match=_convert(raw,"WORLD")
            ids=((match.get("homeTeam") or {}).get("id"),(match.get("awayTeam") or {}).get("id"))
            if match["status"]=="FINISHED" and (home in ids or away in ids):
                past.append(match)
        result=predict(fixture,past)
        if result is None:
            return None, f"Only {len(past)} relevant completed matches returned for this league/season; both teams need at least three."
        return {"league":event["league"],"fixture":fixture,"result":result,
                "kickoff":fixture["utcDate"],"past":past},None
    except Exception as exc:
        return None, f"History unavailable: {str(exc)[:160]}"
