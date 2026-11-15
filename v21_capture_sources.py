"""Opt-in scheduled V21 research provider capture for ORIGINAL frozen pairs.

Requires owner-set GitHub Actions variable V21_EVIDENCE_ENABLED=true. Max one
pair per six-hour run (and max 2 historical xG queries/team). New endpoint
traffic may consume existing API-Football quotas. No AI-provider charges.
One immutable, source-matched packet per original forecast.
"""
import json
import os
from datetime import datetime,timezone,timedelta
from pathlib import Path

from v10_prospective import ROOT as ORIGINAL_ROOT, utc
from v16_pair_research import original_model
from history_pipeline import load_evidence
from v21_provider_evidence import capture,valid_snapshot
from v21_prospective import rank_pairs,_models
from persist_prematch import load_json,write_json

ROOT=Path("data/v21_sources")


def source_packet_path(root,leg):
    def safe(v):
        value=str(v)
        if not value or len(value)>120 or any(c not in
             "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_-." for c in value) or value in (".",".."):
            raise ValueError("Unsafe source identifier")
        return value
    return Path(root)/(safe(leg["provider"])+"_"+safe(leg["fixture_id"])+"_"+
                      safe(leg["forecast_id"])+".json")


def qualified_packet(record,leg,now=None):
    if not isinstance(record,dict):
        return False
    source=record.get("source_forecast") or {}
    keys=("provider","fixture_id","forecast_id","input_digest","home_team_id",
          "away_team_id","kickoff_utc")
    if any(str(source.get(k))!=str(leg.get(k)) for k in keys):
        return False
    fixture={"provider":source["provider"],"id":source["fixture_id"],
             "utcDate":source["kickoff_utc"],
             "homeTeam":{"id":source["home_team_id"]},
             "awayTeam":{"id":source["away_team_id"]}}
    return valid_snapshot(record.get("provider_snapshot"),fixture,now=now)


def current_pair(root=ORIGINAL_ROOT,now=None):
    clock=utc(now or datetime.now(timezone.utc))
    rows=[]
    for path in sorted((Path(root)/"doubles").glob("*.json")):
        pair=load_json(path,{})
        try:
            if pair.get("status")!="frozen_pre_kickoff" or utc(pair["captured_at_utc"])>clock:
                continue
            legs=(pair["first"],pair["second"])
            if any(not clock+timedelta(minutes=45)<utc(x["kickoff_utc"])<=
                   clock+timedelta(hours=36) for x in legs):
                continue
            models=[original_model(root,leg) for leg in legs]
            if not all(models):
                continue
            coverage=float(pair.get("estimated_coverage") or 0)
            rows.append((-coverage,str(pair.get("sydney_date")),models))
        except (KeyError,TypeError,ValueError):
            continue
    return min(rows,key=lambda x:x[:2])[2] if rows else []


def capture_pair(*,root=ROOT,source_root=ORIGINAL_ROOT,now=None,
                 api_key=None,enabled=None,api_request=None):
    clock=utc(now or datetime.now(timezone.utc))
    active=(os.environ.get("V21_EVIDENCE_ENABLED")=="true" if enabled is None else bool(enabled))
    result={"at_utc":clock.isoformat(),"enabled":active,
            "new_packets":[],"calls":0,"skipped":[]}
    key=api_key if api_key is not None else os.environ.get("API_FOOTBALL_KEY","")
    if not active or not key:
        result["reason"]="not_opted_in_or_api_key_missing"
        return result
    from api_football import _request,LEAGUES
    cb=api_request or _request
    models=current_pair(source_root,clock)
    if len(models)!=2:
        result["reason"]="no_original_pair_both_pre_match"
        return result
    cache={}
    # Do not partially present a pair as complete, but valid single-leg
    # snapshots can be retained for prospective audit.
    for leg in models:
        try:
            path=source_packet_path(root,leg)
            if path.exists():
                result["skipped"].append(str(leg["fixture_id"])+":already_frozen")
                continue
            comp=str(leg.get("competition") or "")
            league=LEAGUES.get(comp)
            if league is None:
                result["skipped"].append(str(leg["fixture_id"])+":league_not_supported")
                continue
            fixture={"provider":leg["provider"],"id":int(leg["fixture_id"]),
                     "utcDate":leg["kickoff_utc"],
                     "homeTeam":{"id":int(leg["home_team_id"]),"name":leg.get("home_team")},
                     "awayTeam":{"id":int(leg["away_team_id"]),"name":leg.get("away_team")},
                     "competition":{"code":comp,"id":league}}
            evidence=load_evidence(fixture,as_of=clock,api_key=key,
                                   api_request=cb,cache=cache)
            # Never claim matching historical input_digest after refreshing
            # league history. The packet is supplemental source evidence.
            provider=capture(fixture,evidence["history"],key,
                             now=clock,request=cb,max_history_per_team=2)
            if not valid_snapshot(provider,fixture,now=clock):
                result["skipped"].append(str(leg["fixture_id"])+":no_qualified_packet")
                continue
            record={"schema":1,"source_forecast":{k:leg[k] for k in (
                "provider","fixture_id","forecast_id","input_digest",
                "home_team_id","away_team_id","kickoff_utc")},
                "provider_snapshot":provider,
                "supplemental_history_digest":evidence.get("input_digest"),
                "not_baseline_forecast_digest":True,
                "source_scope":"API-Football endpoints, not arbitrary live websites"}
            path.parent.mkdir(parents=True,exist_ok=True)
            try:
                with path.open("x",encoding="utf-8") as fh:
                    fh.write(json.dumps(record,indent=2,sort_keys=True,allow_nan=False)+"\n")
                result["new_packets"].append(str(leg["fixture_id"]))
            except FileExistsError:
                result["skipped"].append(str(leg["fixture_id"])+":race_preserved")
            # existing provider history requests can consume large quota;
            # V21 never retries failed historical calls within this run.
            result["calls"]+=provider["diagnostics"]["queries"]
        except (ValueError,TypeError,KeyError,OSError):
            result["skipped"].append(str(leg.get("fixture_id"))+":unavailable")
    result["reason"]="partial_or_complete_source_packets" if result["new_packets"] else "no_new_source_evidence"
    write_json(Path(root)/"status.json",result)
    return result


if __name__=="__main__":
    print(json.dumps(capture_pair(),indent=2))
