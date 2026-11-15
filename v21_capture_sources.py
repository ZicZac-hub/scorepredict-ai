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
from source_capture_clock import capture_clock

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



def best_packet(root, leg, *, now=None, priority="research"):
    """Best currently valid write-once packet for a NEW forecast.

    First frozen forecasts retain their own source hash; selecting stronger
    pre-match evidence for a new forecast never rewrites those outcomes.
    Priority uses only source availability and timestamps, NEVER match results.
    """
    root=Path(root)
    base=source_packet_path(root,leg)
    paths=[base]
    paths.extend(sorted((root/"versions").glob(base.stem+"_*.json")))
    valid=[]
    for path in paths:
        record=load_json(path,{})
        if not qualified_packet(record,leg,now=now):
            continue
        snap=record["provider_snapshot"]
        xg=snap.get("verified_historical_fixture_xg") or []
        ids=(str(leg["home_team_id"]),str(leg["away_team_id"]))
        counts=[len({str(r.get("fixture_id")) for r in xg
                     if team_id in (str(r.get("home_team_id")),
                                    str(r.get("away_team_id")))})
                for team_id in ids]
        markets=snap.get("multi_market") or {}
        nmarkets=len(markets)
        valid_market=False
        latest_market=""
        try:
            if ("1x2" in markets and
                ("over_2_5" in markets or "over_3_5" in markets)):
                times=[utc(markets[k]["last_source_update_utc"])
                       for k in markets if k in ("1x2","over_2_5","over_3_5")]
                clock=utc(now or datetime.now(timezone.utc))
                valid_market=all(t<=clock and clock-t<=timedelta(hours=12)
                                 for t in times) and bool(times)
                latest_market=max(t.isoformat() for t in times)
        except (KeyError,TypeError,ValueError):
            valid_market=False
        if priority=="market":
            quality=(int(valid_market),latest_market,nmarkets,
                     min(counts),str(snap.get("captured_at_utc") or ""))
        else:
            quality=(int(min(counts)>=6 and valid_market),
                     min(counts),int(valid_market),nmarkets,
                     str(snap.get("captured_at_utc") or ""))
        valid.append((quality,record))
    return max(valid,key=lambda x:x[0])[1] if valid else None


def current_pair(root=ORIGINAL_ROOT,now=None):
    """Select frozen-original pair LEGS 8-35 min before each kickoff.

    A pair's two matches need not kick off simultaneously. A later match
    remains eligible after the earlier one has started, but it must itself
    still be pre-match and match the ORIGINAL frozen source.
    """
    clock=utc(now or datetime.now(timezone.utc))
    options=[]
    seen=set()
    for path in sorted((Path(root)/"doubles").glob("*.json")):
        pair=load_json(path,{})
        try:
            if pair.get("status")!="frozen_pre_kickoff" or utc(pair["captured_at_utc"])>clock:
                continue
            for leg in (pair["first"],pair["second"]):
                kickoff=utc(leg["kickoff_utc"])
                if not clock+timedelta(minutes=8)<kickoff<=clock+timedelta(minutes=35):
                    continue
                key=(str(leg["provider"]),str(leg["fixture_id"]))
                if key in seen:
                    continue
                model=original_model(root,leg)
                if model is not None and model.get("input_digest")==leg.get("input_digest"):
                    seen.add(key)
                    options.append((kickoff,model))
        except (KeyError,TypeError,ValueError):
            continue
    # Include all individual V8 fixtures, not just already-formed doubles.
    # This only runs after the separate explicit V21_EVIDENCE_ENABLED opt-in.
    from v10_prospective import read_latest
    rows=read_latest(root)
    for leg in rows.values() if isinstance(rows,dict) else rows:
        try:
            if (leg.get("model_key") not in ("v8","v8_near") or
                    leg.get("provider")!="API-Football"):
                continue
            kickoff=utc(leg["kickoff_utc"])
            if not clock+timedelta(minutes=8)<kickoff<=clock+timedelta(minutes=35):
                continue
            key=(str(leg["provider"]),str(leg["fixture_id"]))
            if key in seen:
                continue
            model=original_model(root,leg)
            if model is not None and model.get("input_digest")==leg.get("input_digest"):
                seen.add(key)
                options.append((kickoff,model))
        except (KeyError,TypeError,ValueError):
            continue
    options.sort(key=lambda x:(x[0],str(x[1]["fixture_id"])))
    return [x[1] for x in options[:2]]


def capture_pair(*,root=ROOT,source_root=ORIGINAL_ROOT,now=None,
                 api_key=None,enabled=None,api_request=None,clock=None):
    clock,observation_time=capture_clock(now,clock)
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
    if not models:
        result["reason"]="no_original_frozen_leg_in_8_to_35_minute_window"
        return result
    cache={}
    # Do not partially present a pair as complete, but valid single-leg
    # snapshots can be retained for prospective audit.
    for leg in models:
        try:
            path=source_packet_path(root,leg)
            deep=(os.environ.get("V22_DEEP_XG_ENABLED")=="true")
            previous=best_packet(root,leg,now=clock)
            existing=(previous or {}).get("provider_snapshot") or {}
            prior_xg=existing.get("verified_historical_fixture_xg") or []
            prior_market=existing.get("multi_market") or {}
            upgrade_needed=bool(deep and path.exists() and
                (len(prior_xg)<12 or len(prior_market)<2))
            deep_marker=Path(root)/"deep_attempts"/path.name
            if path.exists() and (not upgrade_needed or deep_marker.exists()):
                result["skipped"].append(str(leg["fixture_id"])+":original_preserved")
                continue
            if upgrade_needed:
                # Reserve one funded refresh attempt BEFORE any API requests,
                # including failed/quota-limited calls. Do not retry every
                # 15 minutes and silently spend a larger quota.
                deep_marker.parent.mkdir(parents=True,exist_ok=True)
                try:
                    with deep_marker.open("x",encoding="utf-8") as marker:
                        marker.write(json.dumps({"attempted_at_utc":clock.isoformat(),
                            "fixture_id":str(leg["fixture_id"]),
                            "reason":"explicit_V22_deep_xg_opt_in"},sort_keys=True)+"\n")
                except FileExistsError:
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
                             now=observation_time(),request=cb,clock=observation_time,
                             max_history_per_team=(8 if
                                 os.environ.get("V22_DEEP_XG_ENABLED")=="true"
                                 else 1))
            clock=observation_time()
            result["calls"]+=(provider.get("diagnostics") or {}).get("queries",0)
            if not valid_snapshot(provider,fixture,now=clock):
                result["skipped"].append(str(leg["fixture_id"])+":no_qualified_packet")
                continue
            if not any((provider.get("available_sources") or {}).values()):
                result["skipped"].append(str(leg["fixture_id"])+":all_sources_unavailable")
                continue
            record={"schema":1,"source_forecast":{k:leg[k] for k in (
                "provider","fixture_id","forecast_id","input_digest",
                "home_team_id","away_team_id","kickoff_utc")},
                "provider_snapshot":provider,
                "supplemental_history_digest":evidence.get("input_digest"),
                "not_baseline_forecast_digest":True,
                "source_scope":"API-Football endpoints, not arbitrary live websites"}
            if upgrade_needed:
                # Freeze a SECOND source packet by its content hash. Never
                # replace the earlier original path or use a postkickoff file.
                path=Path(root)/"versions"/(path.stem+"_"+
                     provider["snapshot_sha256"][:16]+".json")
            path.parent.mkdir(parents=True,exist_ok=True)
            try:
                with path.open("x",encoding="utf-8") as fh:
                    fh.write(json.dumps(record,indent=2,sort_keys=True,allow_nan=False)+"\n")
                result["new_packets"].append(str(leg["fixture_id"]))
            except FileExistsError:
                result["skipped"].append(str(leg["fixture_id"])+":race_preserved")
            # existing provider history requests can consume large quota;
            # V21 never retries failed historical calls within this run.
        except (ValueError,TypeError,KeyError,OSError):
            result["skipped"].append(str(leg.get("fixture_id"))+":unavailable")
    result["reason"]="partial_or_complete_source_packets" if result["new_packets"] else "no_new_source_evidence"
    write_json(Path(root)/"status.json",result)
    return result


if __name__=="__main__":
    print(json.dumps(capture_pair(),indent=2))
