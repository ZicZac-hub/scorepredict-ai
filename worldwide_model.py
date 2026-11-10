"""On-demand API-Football history for a selected worldwide fixture."""
from datetime import datetime, timezone
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
        # Fetch each team's latest completed fixtures; never combine provider IDs.
        all_games={}
        for team in (home,away):
            for raw in _request(key,"/fixtures",{"team":team,"last":20}):
                match=_convert(raw,"WORLD")
                if match["status"]=="FINISHED" and match["utcDate"] and datetime.fromisoformat(match["utcDate"].replace("Z","+00:00"))<datetime.now(timezone.utc):
                    all_games[match["id"]]=match
        past=list(all_games.values())
        result=predict(fixture,past)
        if result is None:
            return None, "Not enough accessible completed games (minimum three per team)."
        return {"league":event["league"],"fixture":fixture,"result":result,
                "kickoff":fixture["utcDate"],"past":past},None
    except Exception as exc:
        return None, f"History unavailable: {str(exc)[:160]}"
