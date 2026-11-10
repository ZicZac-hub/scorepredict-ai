"""On-demand, quota-aware historical modelling for API-Football fixtures."""
from datetime import datetime, timedelta, timezone
from api_football import _request, _convert
from engine import predict

def model_worldwide(key, event):
    fixture=event.get("fixture")
    if not fixture:
        return None, "Fixture details are unavailable."
    if not key or fixture.get("provider")!="API-Football":
        try:
            past,error=([], 'Public-web historical data is not licensed for automated AI ingestion; use an authorised source')
            result=predict(fixture,past) if past else None
            if result:
                return {"league":event["league"],"fixture":fixture,"result":result,
                        "kickoff":fixture["utcDate"],"past":past,
                        "research_sources":["football-data.co.uk (public historical CSV)"]},None
            return None,error or "Not enough verified public historical results for both teams"
        except Exception as exc:
            return None,f"Public web history unavailable: {str(exc)[:130]}"
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
            public_past,public_error=public_web_history(event)
            if public_past:
                result=predict(fixture,public_past)
                if result is not None:
                    return {"league":event["league"],"fixture":fixture,"result":result,
                            "kickoff":fixture["utcDate"],"past":public_past,
                            "research_sources":["football-data.co.uk (public historical CSV)"]},None
        if result is None:
            return None, f"Insufficient verified historical results from API and public web sources ({len(past)} API matches)."
        return {"league":event["league"],"fixture":fixture,"result":result,
                "kickoff":fixture["utcDate"],"past":past,
                "research_sources":["API-Football"]},None
    except Exception as exc:
        try:
            public_past,public_error=public_web_history(event)
            if public_past:
                result=predict(fixture,public_past)
                if result is not None:
                    return {"league":event["league"],"fixture":fixture,"result":result,
                            "kickoff":fixture["utcDate"],"past":public_past,
                            "research_sources":["football-data.co.uk (public historical CSV)"]},None
            return None,f"API history unavailable; public web fallback: {public_error or 'insufficient recent team results'}"
        except Exception as fallback_exc:
            return None,f"API and public web history unavailable: {str(fallback_exc)[:130]}"

