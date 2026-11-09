"""Optional API-Football provider. Uses the API-Sports direct endpoint.

Free-tier league/season availability varies. No fabricated matches or results.
"""
from datetime import datetime, timedelta, timezone
import hashlib
import json
import time
from pathlib import Path
import requests

ROOT = "https://v3.football.api-sports.io"
CACHE = Path("/tmp/scorepredict_api_sports")
CACHE.mkdir(parents=True, exist_ok=True)
TTL = 21600  # six hours: preserve the free daily quota
LEAGUES = {"PL":39, "PD":140, "BL1":78, "SA":135, "FL1":61,
           "CL":2, "EL":3, "DED":88, "PPL":94, "BSA":71, "ELC":40}
NAMES = {"PL":"Premier League","PD":"La Liga","BL1":"Bundesliga",
         "SA":"Serie A","FL1":"Ligue 1","CL":"Champions League",
         "EL":"Europa League","DED":"Eredivisie","PPL":"Primeira Liga",
         "BSA":"Brazil Serie A","ELC":"English Championship"}

def _request(key, endpoint, params):
    signature=json.dumps([endpoint,params,hashlib.sha256(key.encode()).hexdigest()],sort_keys=True)
    path=CACHE/(hashlib.sha256(signature.encode()).hexdigest()+".json")
    saved=None
    try:
        saved=json.loads(path.read_text())
        if time.time()-saved["time"] < TTL:
            return saved["data"]
    except (OSError, ValueError, KeyError, TypeError):
        pass
    try:
        response=requests.get(ROOT+endpoint,params=params,
                              headers={"x-apisports-key":key},timeout=25)
        if response.status_code==429:
            raise RuntimeError("API-Football daily or per-minute quota reached")
        response.raise_for_status()
        payload=response.json()
        if payload.get("errors"):
            raise RuntimeError("API-Football: "+str(payload["errors"])[:250])
        data=payload.get("response",[])
        path.write_text(json.dumps({"time":time.time(),"data":data}))
        return data
    except (requests.RequestException,RuntimeError) as exc:
        if saved and time.time()-saved.get("time",0)<86400:
            return saved["data"]  # last-good snapshot; caller must not claim live freshness
        raise RuntimeError(str(exc)) from exc

def _convert(item, code):
    f=item.get("fixture") or {}
    teams=item.get("teams") or {}
    goals=item.get("goals") or {}
    h=teams.get("home") or {}
    a=teams.get("away") or {}
    status=(f.get("status") or {}).get("short","")
    date=f.get("date","")
    # Convert offset-aware kickoff to UTC for consistent filtering and sorting.
    try:
        date=datetime.fromisoformat(date.replace("Z","+00:00")).astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    except (ValueError,TypeError):
        date=""
    done=status in ("FT","AET","PEN")
    return {"id":f.get("id"),"utcDate":date,
            "status":"FINISHED" if done else ("SCHEDULED" if status in ("NS","TBD") else "TIMED"),
            "homeTeam":{"id":h.get("id"),"name":h.get("name","Home"),"shortName":h.get("name","Home")},
            "awayTeam":{"id":a.get("id"),"name":a.get("name","Away"),"shortName":a.get("name","Away")},
            "score":{"fullTime":{"home":goals.get("home"),"away":goals.get("away")}},
            "competition":{"code":code},"provider":"API-Football"}

def _season(code, now):
    # Most European seasons start in summer; Brazil uses a calendar year.
    if code=="BSA":return now.year
    return now.year if now.month>=7 else now.year-1

def get_fixtures(key,code,now,days):
    if code not in LEAGUES:return []
    season=_season(code,now)
    params={"league":LEAGUES[code],"season":season,
            "from":now.date().isoformat(),
            "to":(now.date()+timedelta(days=days+1)).isoformat()}
    return [m for item in _request(key,"/fixtures",params)
            if (m:=_convert(item,code))["status"] in ("SCHEDULED","TIMED")]

def get_history(key,code,now):
    if code not in LEAGUES:return []
    params={"league":LEAGUES[code],"season":_season(code,now),
            "from":(now.date()-timedelta(days=100)).isoformat(),
            "to":now.date().isoformat()}
    return [m for item in _request(key,"/fixtures",params)
            if (m:=_convert(item,code))["status"]=="FINISHED"]

def supported_codes():
    return LEAGUES.copy()
