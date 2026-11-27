"""Recover blocked historical receipts using only this run's existing cache.

Never invoke a provider adapter or load_evidence. Both season responses must
already be present and their adapter freshness metadata must qualify. The
current-season fetch must postdate the earlier near forecast. This is not
proof that the provider updated any results. One recovery attempt per run.
"""
from collections import Counter
from pathlib import Path

from forecast_ledger import read_file_records
from history_pipeline import build_evidence,season_year
from v10_prospective import utc
from v44_refresh_priority import targets,fixture_key,model_key
from v44_refresh_expiry_audit import inspect_leg
from v44_recovery_source_freshness import inspect as inspect_freshness
import v44_recovery_capture as recovery
import v44_early_history as early
import v42_research_trial as sources
from datetime import timedelta

MAX_ATTEMPTS=1

def packet_from_cache(fixture,cache,observed,prior_capture):
    observed=utc(observed)
    current=season_year(fixture)
    provider=str(fixture["provider"])
    competition=fixture.get("competition") or {}
    league=competition.get("id") or competition.get("code")
    if league is None:
        return None,"missing_competition"
    rows=[];metadata=[]
    for season in (current,current-1):
        item=cache.get((provider,str(league),season))
        if item is None:
            return None,"season_not_in_run_cache"
        matches,meta,error=item
        if error or not isinstance(matches,list) or not isinstance(meta,dict):
            return None,"unusable_cached_season"
        rows.extend(matches)
        metadata.append(meta)
    evidence=build_evidence(fixture,rows,observed,metadata)
    freshness=inspect_freshness(fixture,evidence,observed)
    if freshness["status"]!="adapter_fresh":
        return None,"cache_not_fully_adapter_fresh"
    current_source=next(s for s in freshness["seasons"] if s["season"]==current)
    if not utc(prior_capture)<utc(current_source["fetched_at_utc"]):
        return None,"current_season_fetch_not_newer_than_prior"
    if not evidence["history"]:
        return None,"no_qualified_history"
    return evidence,"ready"

def run(fixtures,history_cache,*,clock,time_available,refresher,
        root=recovery.ROOT,source_root=sources.SOURCE_ROOT,
        pair_root=sources.PAIR_ROOT,history_root=early.ROOT):
    counts=Counter();attempted=0
    def result():
        return {"attempted":attempted,"max_attempts":MAX_ATTEMPTS,
                "reasons":dict(counts),"external_requests":0}
    if not history_cache:
        counts["no_in_run_history_cache"]+=1
        return result()
    if not time_available():
        counts["time_budget"]+=1
        return result()
    now=utc(clock())
    required=targets(now=now,pair_root=pair_root,source_root=source_root)
    candidates=[]
    for row in fixtures:
        try:
            key=fixture_key(row[1])
            if key in required:
                candidates.append((required[key],key[2],key[0],key[1],row[1]))
        except (ValueError,KeyError,TypeError,AttributeError):
            counts["invalid_fixture_identity"]+=1
    counts["authenticated_targets_in_discovery"]=len(candidates)
    seen=set()
    for _,_,_,_,fixture in sorted(candidates,key=lambda x:x[:4]):
        if attempted>=MAX_ATTEMPTS or not time_available():
            break
        try:
            key=fixture_key(fixture)
            if key in seen:
                continue
            seen.add(key)
            observed=utc(clock())
            earliest=required[key]
            if not timedelta(minutes=45)<earliest-observed<=timedelta(hours=24):
                counts["outside_refresh_window"]+=1
                continue
            existing=read_file_records(root=Path(root)/"archive",kind="forecasts",
                                       provider=key[0],fixture_id=key[1])
            if any(m.get("model_key")==recovery.MODEL_KEY for m in existing):
                counts["already_recovered"]+=1
                continue
            original=read_file_records(root=source_root,kind="forecasts",
                                       provider=key[0],fixture_id=key[1])
            near=[m for m in original if m.get("model_key")=="v8_near"
                  and model_key(m)==key and utc(m["captured_at_utc"])<observed]
            if not near:
                counts["no_prior_near_forecast"]+=1
                continue
            prior=min(near,key=lambda m:utc(m["captured_at_utc"]))
            opening=max(observed,earliest-timedelta(hours=3))
            status=inspect_leg(prior,opening,observed,source_root,history_root)
            if status not in ("one_shot_near_without_exact_history","one_shot_near_expired"):
                counts["existing_history_usable_or_not_blocked"]+=1
                continue
            evidence,reason=packet_from_cache(fixture,history_cache,observed,prior["captured_at_utc"])
            if evidence is None:
                counts[reason]+=1
                continue
            if not time_available():
                counts["time_budget"]+=1
                break
            attempted+=1
            outcome=refresher(fixture,evidence,utc(clock()))
            counts[outcome.get("status","unknown")]+=1
        except (ValueError,TypeError,KeyError,OSError,AttributeError,ArithmeticError) as exc:
            counts["error_"+type(exc).__name__]+=1
    return result()
