"""On-demand, quota-aware historical modelling for API-Football fixtures."""
from datetime import datetime, timedelta, timezone
from api_football import _request, _convert
from engine import predict

def model_worldwide(key, event):
    """Retrieve permitted historical results; explain coverage gaps precisely."""
    fixture=event.get("fixture")
    if not fixture:
        return None,"Fixture has no provider identifiers. Search independent results links below."
    if not key or fixture.get("provider")!="API-Football":
        return None,"No compatible historical-results provider for this fixture."
    home=(fixture.get("homeTeam") or {}).get("id")
    away=(fixture.get("awayTeam") or {}).get("id")
    if not home or not away:
        return None,"Provider team IDs are missing."
    kickoff=datetime.fromisoformat(fixture["utcDate"].replace("Z","+00:00"))
    league_id=event.get("league_id")
    season=event.get("season")
    if not league_id or not season:
        return None,"League or season ID missing from the fixture feed."
    end=(min(kickoff,datetime.now(timezone.utc))-timedelta(days=1)).date()
    start=end-timedelta(days=125)
    past=[]
    errors=[]
    # The league-wide endpoint may be unavailable on the free tier.
    # Team-specific date-range requests can return matches from multiple
    # competitions, improving form and venue coverage.
    attempts=[
        ("league",{"league":league_id,"season":season,"from":start.isoformat(),"to":end.isoformat()}),
        ("home team",{"team":home,"season":season,"from":start.isoformat(),"to":end.isoformat()}),
        ("away team",{"team":away,"season":season,"from":start.isoformat(),"to":end.isoformat()})
    ]
    seen=set()
    for label,params in attempts:
        try:
            for raw in _request(key,"/fixtures",params):
                match=_convert(raw,"WORLD")
                ids=((match.get("homeTeam") or {}).get("id"),(match.get("awayTeam") or {}).get("id"))
                if match.get("status")!="FINISHED" or not (home in ids or away in ids):
                    continue
                ident=(match.get("utcDate"),match.get("homeTeam",{}).get("id"),match.get("awayTeam",{}).get("id"))
                if ident not in seen:
                    seen.add(ident)
                    past.append(match)
        except Exception as exc:
            errors.append(label+": "+str(exc)[:110])
        result=predict(fixture,past)
        if result is not None:
            return {"league":event.get("league","Unknown"),"fixture":fixture,
                    "result":result,"kickoff":fixture["utcDate"],"past":past,
                    "research_sources":["API-Football · verified completed matches"]},None
    home_count=sum(home in (m["homeTeam"]["id"],m["awayTeam"]["id"]) for m in past)
    away_count=sum(away in (m["homeTeam"]["id"],m["awayTeam"]["id"]) for m in past)
    reason=f"Verified results found: {home_count} home-team matches, {away_count} away-team matches; at least 5 each required for this model."
    if errors:reason+=" Provider restrictions: "+" | ".join(errors[:2])
    return None,reason
