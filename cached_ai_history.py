"""Reuse first-party fixture cache for AI context with zero network requests.

The original statistical forecast stays unchanged. These are independently
qualified research facts observed no later than the common AI decision cutoff.
Missing or expired cache files remain missing, without triggering a paid fetch.
"""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path

from history_pipeline import load_evidence
from v10_prospective import utc


def cached_history(row, *, as_of, api_key="", cache_root=None):
    from api_football import CACHE,LEAGUES,cache_ttl
    cutoff=utc(as_of)
    code=str(row.get("competition") or "")
    fixture={"provider":row.get("provider"),"id":row.get("fixture_id"),
             "utcDate":row["kickoff_utc"],
             "homeTeam":{"id":row.get("home_team_id"),"name":row.get("home_team")},
             "awayTeam":{"id":row.get("away_team_id"),"name":row.get("away_team")},
             "competition":{"code":code,"id":LEAGUES.get(code)}}
    root=Path(CACHE if cache_root is None else cache_root)
    reads=[]

    def cache_only(key,endpoint,params,with_metadata=False):
        if not api_key or key!=api_key or endpoint!="/fixtures":
            raise ValueError("matching_local_history_cache_unavailable")
        signature=json.dumps([endpoint,params,hashlib.sha256(key.encode()).hexdigest()],sort_keys=True)
        path=root/(hashlib.sha256(signature.encode()).hexdigest()+".json")
        saved=json.loads(path.read_text(encoding="utf-8"))
        stamp=saved["time"]
        if type(stamp) not in (int,float):
            raise ValueError("invalid_cached_source_time")
        observed=datetime.fromtimestamp(stamp,timezone.utc)
        age=(cutoff-observed).total_seconds()
        ttl=cache_ttl(endpoint,params)
        if not 0<=age<ttl or not isinstance(saved.get("data"),list):
            raise ValueError("expired_or_future_history_cache")
        metadata={"provider":"API-Football","endpoint":endpoint,
                  "fetched_at_utc":observed.isoformat(),"retrieved_at_utc":cutoff.isoformat(),
                  "age_seconds":age,"cache_status":"fresh_cache","stale":False,
                  "ttl_seconds":ttl,"freshness_known":True}
        reads.append({**metadata,"season":params.get("season")})
        return {"data":saved["data"],"metadata":metadata}

    evidence=load_evidence(fixture,as_of=cutoff,api_key=api_key,api_request=cache_only)
    ids={str(row.get("home_team_id")),str(row.get("away_team_id"))}
    history=[game for game in evidence["history"] if ids.intersection(
        str((game.get(side) or {}).get("id")) for side in ("homeTeam","awayTeam"))]
    return {"history":history,"input_digest":evidence["input_digest"],
            "source_metadata":reads,"warnings":evidence.get("warnings") or [],
            "external_requests":0,"is_original_statistical_input":False}
