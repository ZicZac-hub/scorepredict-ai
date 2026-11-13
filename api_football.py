"""Optional API-Football provider. Uses the API-Sports direct endpoint.

Free-tier league/season availability varies. No fabricated matches or results.
"""
from datetime import date as calendar_date, datetime, timedelta, timezone
import hashlib
import json
import time
import os
from uuid import uuid4
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

def cache_ttl(endpoint, params):
    """Cache by data lifecycle instead of retaining lineups/status for six hours."""
    if endpoint == '/fixtures/lineups': return 900
    if endpoint == '/fixtures' and params.get('live'): return 30
    if endpoint == '/fixtures' and params.get('id') is not None: return 60
    if endpoint == '/fixtures':
        if params.get('season') is not None:
            try:
                return 86400 if int(params['season']) < datetime.now(timezone.utc).year-1 else 900
            except (TypeError, ValueError):
                return 900
        return 300
    if endpoint == '/injuries': return 14400
    if endpoint == '/odds': return 10800
    if endpoint == '/teams/statistics': return 43200
    if endpoint == '/standings': return 3600
    return TTL


def _expose(data, fetched_at, ttl, cache_status, with_metadata):
    if not with_metadata:
        return data
    now = time.time()
    return {'data': data, 'metadata': {
        'provider': 'API-Football',
        'fetched_at_utc': datetime.fromtimestamp(fetched_at, timezone.utc).isoformat(),
        'retrieved_at_utc': datetime.fromtimestamp(now, timezone.utc).isoformat(),
        'age_seconds': max(0.0, now-fetched_at), 'cache_status': cache_status,
        'stale': cache_status == 'stale_fallback', 'ttl_seconds': ttl,
        'freshness_known': True}}


def _request(key, endpoint, params, with_metadata=False):
    signature=json.dumps([endpoint,params,hashlib.sha256(key.encode()).hexdigest()],sort_keys=True)
    path=CACHE/(hashlib.sha256(signature.encode()).hexdigest()+".json")
    ttl = cache_ttl(endpoint, params)
    saved=None
    try:
        saved=json.loads(path.read_text())
        if time.time()-saved["time"] < ttl:
            return _expose(saved['data'], saved['time'], ttl, 'fresh_cache', with_metadata)
    except (OSError, ValueError, KeyError, TypeError):
        saved=None
    try:
        response=requests.get(ROOT+endpoint,params=params,
                              headers={"x-apisports-key":key},timeout=25)
        if response.status_code==429:
            raise RuntimeError("API-Football daily or per-minute quota reached")
        response.raise_for_status()
        payload=response.json()
        if payload.get("errors"):
            raise RuntimeError("API-Football request was rejected by the provider")
        data=payload.get("response",[])
        fetched_at=time.time()
        temporary = path.with_name(path.name+'.'+uuid4().hex+'.tmp')
        try:
            temporary.write_text(json.dumps({"time":fetched_at,"data":data}))
            os.replace(temporary,path)
        except OSError:
            try: temporary.unlink(missing_ok=True)
            except OSError: pass
        return _expose(data, fetched_at, ttl, 'network', with_metadata)
    except (requests.RequestException,RuntimeError) as exc:
        # Stale data remain useful only with their original fetch time. Fast
        # endpoints never receive a day-old fixture status or lineup fallback.
        stale_limit = min(86400, max(300, ttl*4))
        if saved and time.time()-saved.get("time",0)<stale_limit:
            return _expose(saved['data'], saved['time'], ttl, 'stale_fallback', with_metadata)
        raise RuntimeError('API-Football request failed ('+type(exc).__name__+')') from exc

def _as_datetime(value):
    """Accept the cached UI's ISO day strings as well as date/datetime callers."""
    if isinstance(value, datetime):
        result = value
    elif isinstance(value, calendar_date):
        result = datetime.combine(value, datetime.min.time())
    elif isinstance(value, str):
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    else:
        raise TypeError("Expected a datetime, date, or ISO date string")
    if result.tzinfo is None:
        result = result.replace(tzinfo=timezone.utc)
    return result.astimezone(timezone.utc)

def _convert(item, code):
    f=item.get("fixture") or {}
    teams=item.get("teams") or {}
    goals=item.get("goals") or {}
    h=teams.get("home") or {}
    a=teams.get("away") or {}
    league=item.get("league") or {}
    status=(f.get("status") or {}).get("short","")
    date=f.get("date","")
    # Convert offset-aware kickoff to UTC for consistent filtering and sorting.
    try:
        date=_as_datetime(date).strftime("%Y-%m-%dT%H:%M:%SZ")
    except (ValueError,TypeError):
        date=""
    done=status in ("FT","AET","PEN")
    # Exact-score forecasts target regulation time. API-Football's `goals`
    # includes extra time for AET/PEN, while score.fulltime is the 90-minute
    # result. Missing regulation scores must remain unknown, never inferred.
    fulltime=(item.get("score") or {}).get("fulltime") or {}
    if status == "FT" and not all(type(fulltime.get(s)) is int for s in ("home", "away")):
        fulltime=goals
    score={side:fulltime.get(side) if type(fulltime.get(side)) is int else None
           for side in ("home", "away")}
    league_id=league.get("id")
    actual_code=next((name for name,identifier in LEAGUES.items() if identifier==league_id),
                     f"API-{league_id}" if league_id is not None else code)
    result={"id":f.get("id"),"utcDate":date,
            "status":"FINISHED" if done else ("SCHEDULED" if status in ("NS","TBD") else "TIMED"),
            "homeTeam":{"id":h.get("id"),"name":h.get("name","Home"),"shortName":h.get("name","Home")},
            "awayTeam":{"id":a.get("id"),"name":a.get("name","Away"),"shortName":a.get("name","Away")},
            "score":{"fullTime":score,"duration":"REGULAR"},
            "competition":{"id":league_id,"code":actual_code,"season":league.get("season"),
                           "name":league.get("name")},
            "season":league.get("season"),"provider":"API-Football",
            "provider_status":status,"score_basis":"regulation"}
    explicit_finish=item.get('finishedAt') or f.get('finishedAt')
    if explicit_finish:
        try: result['finishedAt']=_as_datetime(explicit_finish).isoformat()
        except (TypeError,ValueError): pass
    return result

def _season(code, now):
    now=_as_datetime(now)
    # Most European seasons start in summer; Brazil uses a calendar year.
    if code=="BSA":return now.year
    return now.year if now.month>=7 else now.year-1

def get_fixtures(key,code,now,days):
    if code not in LEAGUES:return []
    now=_as_datetime(now)
    season=_season(code,now)
    params={"league":LEAGUES[code],"season":season,
            "from":now.date().isoformat(),
            "to":(now.date()+timedelta(days=days+1)).isoformat()}
    return [m for item in _request(key,"/fixtures",params)
            if (m:=_convert(item,code))["status"] in ("SCHEDULED","TIMED")]

def get_history(key,code,now,with_evidence=False):
    if code not in LEAGUES:return []
    from history_pipeline import load_evidence
    now=_as_datetime(now)
    fixture={'utcDate':now.isoformat(),'provider':'API-Football',
             'competition':{'id':LEAGUES[code],'code':code,'season':_season(code,now)}}
    evidence=load_evidence(fixture,as_of=now,api_key=key,api_request=_request)
    return evidence if with_evidence else evidence['history']

def supported_codes():
    return LEAGUES.copy()

