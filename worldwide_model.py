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

# Independent public-web historical results fallback (no API key).
# Source: football-data.co.uk season CSV files; no arbitrary website scraping.
import csv
import io
import re
import requests
from functools import lru_cache
from difflib import SequenceMatcher

PUBLIC_DIVISIONS={
    39:"E0",40:"E1",41:"E2",42:"E3",140:"SP1",141:"SP2",
    78:"D1",79:"D2",135:"I1",136:"I2",61:"F1",62:"F2",
    88:"N1",94:"P1",203:"T1",179:"SC0"
}
def _normal(name):
    name=str(name or "").casefold().replace("&"," and ")
    name=re.sub(r"\\b(fc|cf|afc|sc|ac|club|football|de|the)\\b"," ",name)
    return re.sub(r"[^a-z0-9]","",name)
def _match_name(name,candidates):
    needle=_normal(name)
    if not needle:return None
    exact=[x for x in candidates if _normal(x)==needle]
    if len(exact)==1:return exact[0]
    ranked=sorted(((SequenceMatcher(None,needle,_normal(x)).ratio(),x) for x in candidates),reverse=True)
    # Conservative matching prevents attaching another club's statistics.
    if ranked and ranked[0][0]>=.88 and (len(ranked)==1 or ranked[0][0]-ranked[1][0]>=.07):
        return ranked[0][1]
    return None

@lru_cache(maxsize=64)
def _public_csv(division,season):
    url=f"https://www.football-data.co.uk/mmz4281/{season}/{division}.csv"
    response=requests.get(url,timeout=9,headers={"User-Agent":"ZicZacPredictAI/1.0 (public historical football research)"})
    response.raise_for_status()
    if len(response.content)>3_000_000:raise ValueError("Historical dataset too large")
    return list(csv.DictReader(io.StringIO(response.content.decode("utf-8-sig",errors="replace"))))

def public_web_history(event):
    fixture=event.get("fixture") or {}
    league_id=event.get("league_id")
    division=PUBLIC_DIVISIONS.get(int(league_id)) if league_id is not None else None
    if not division:
        return [],"No supported public historical CSV for this competition"
    kickoff=datetime.fromisoformat(fixture["utcDate"].replace("Z","+00:00"))
    season_start=kickoff.year if kickoff.month>=7 else kickoff.year-1
    seasons=[season_start,season_start-1]
    records=[]
    for y in seasons:
        season=f"{y%100:02d}{(y+1)%100:02d}"
        try:records.extend(_public_csv(division,season))
        except (requests.RequestException,ValueError):continue
    candidates={x.get("HomeTeam","") for x in records}|{x.get("AwayTeam","") for x in records}
    home=fixture["homeTeam"];away=fixture["awayTeam"]
    home_name=_match_name(home.get("name"),candidates)
    away_name=_match_name(away.get("name"),candidates)
    if not home_name or not away_name or home_name==away_name:
        return [],"Public results found, but team names could not be matched safely"
    names={home_name:home["id"],away_name:away["id"]}
    # Other clubs need stable synthetic IDs, disjoint from API numeric IDs.
    for n in sorted(candidates):
        if n and n not in names:names[n]="public:"+division+":"+_normal(n)
    history=[]
    for row in records:
        hn,an=row.get("HomeTeam"),row.get("AwayTeam")
        if hn not in names or an not in names:continue
        if hn not in (home_name,away_name) and an not in (home_name,away_name):continue
        try:
            date=datetime.strptime(row["Date"],"%d/%m/%Y").replace(tzinfo=timezone.utc)
            hg,ag=int(row["FTHG"]),int(row["FTAG"])
        except (KeyError,ValueError,TypeError):continue
        if date>=kickoff:continue
        history.append({"id":"public:"+division+":"+date.isoformat()+":"+_normal(hn)+":"+_normal(an),
                        "provider":"football-data.co.uk","utcDate":date.isoformat(),
                        "status":"FINISHED",
                        "homeTeam":{"id":names[hn],"name":hn},
                        "awayTeam":{"id":names[an],"name":an},
                        "score":{"fullTime":{"home":hg,"away":ag}}})
    return history,None
