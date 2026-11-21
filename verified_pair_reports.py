"""Authoritative read-only grading of V22/V23 immutable research pairs.

Original forecasts, source packets, decision times and the five choices are
reconstructed before any outcome is counted. Legacy freeze/model modules stay
unchanged so registered research implementations remain reproducible.
"""
from collections import Counter
from datetime import datetime,timedelta,timezone
import json
import math
from pathlib import Path
from zoneinfo import ZoneInfo

from five_double_accuracy import _final,_result_index,_sources,wilson
from persist_prematch import load_json,write_json
from v10_prospective import ROOT as ORIGINAL_ROOT,read_latest,utc
from v21_capture_sources import qualified_packet,source_packet_path
from v21_goal_model_comparison import top_scores
from v22_calibration import fitted_temperature,scale
from v22_prospective import _fixture,_options
from v22_shadow import shadow as xg_market_shadow
from v23_market_form import shadow as market_shadow

PROVIDER_ROOT=Path("data/v21_sources")
SYDNEY=ZoneInfo("Australia/Sydney")
METHODS={"v22_research":"original_pair_same_verified_xg_and_market_gate",
         "v23_market_form":"time_checked_market_and_original_goals_when_xg_unavailable"}


def _same(saved,expected):
    if type(saved) in (int,float) and type(expected) in (int,float):
        return math.isfinite(saved) and math.isfinite(expected) and math.isclose(
            saved,expected,rel_tol=0,abs_tol=1e-12)
    if type(saved) is not type(expected):
        return False
    if isinstance(saved,dict):
        return saved.keys()==expected.keys() and all(_same(saved[k],expected[k]) for k in saved)
    if isinstance(saved,list):
        return len(saved)==len(expected) and all(_same(a,b) for a,b in zip(saved,expected))
    return saved==expected


def _packet(model,digest,root,decision):
    if not isinstance(digest,str) or len(digest)!=64:
        raise ValueError("source_packet_hash_missing")
    base=source_packet_path(root,model)
    for path in [base,*sorted((Path(root)/"versions").glob(base.stem+"_*.json"))]:
        record=load_json(path,{})
        packet=record.get("provider_snapshot") if isinstance(record,dict) else None
        if isinstance(packet,dict) and packet.get("snapshot_sha256")==digest:
            if not qualified_packet(record,model,now=decision):
                raise ValueError("source_packet_not_available_or_identity_matched")
            return packet
    raise ValueError("original_source_packet_missing")


def _reconstruct(pair,path,*,source_root,provider_root,method,originals,finals,now):
    if (not isinstance(pair,dict) or pair.get("status")!="frozen_pre_kickoff"
            or pair.get("schema")!=1 or pair.get("policy")!=METHODS[method]):
        raise ValueError("unsupported_pair_policy")
    decision=utc(pair["captured_at_utc"])
    if decision>now:
        raise ValueError("pair_not_available_at_evaluation")
    day=pair["sydney_date"]
    if path.stem!=day:
        raise ValueError("pair_filename_sydney_date_mismatch")
    models=_sources(source_root,pair,decision)
    if any(str(pair[side].get("input_digest"))!=str(model["input_digest"])
           for side,model in zip(("first","second"),models)):
        raise ValueError("original_input_digest_mismatch")
    if any(utc(model["captured_at_utc"])>=utc(model["kickoff_utc"])
           or utc(model["kickoff_utc"])<=decision for model in models):
        raise ValueError("pair_not_captured_before_both_kickoffs")
    if any(not decision+timedelta(minutes=5)<utc(model["kickoff_utc"])<=decision+timedelta(hours=36)
           for model in models):
        raise ValueError("pair_outside_predeclared_capture_window")
    if {utc(model["kickoff_utc"]).astimezone(SYDNEY).date().isoformat() for model in models}!={day}:
        raise ValueError("same_sydney_date_required")
    ids=[(model["provider"],str(model[key])) for model in models
         for key in ("home_team_id","away_team_id")]
    names=[" ".join(str(model[key]).casefold().split()) for model in models
           for key in ("home_team","away_team")]
    if len(set(ids))!=4 or not all(names) or len(set(names))!=4:
        raise ValueError("four_distinct_known_teams_required")
    hashes=pair["source_packet_hashes"]
    if not isinstance(hashes,list) or len(hashes)!=2:
        raise ValueError("two_original_source_packet_hashes_required")
    baseline=[];challenger=[]
    for index,model in enumerate(models):
        packet=_packet(model,hashes[index],provider_root,decision)
        shaped={"fixture":_fixture(model),"result":{"matrix":model["score_matrix"]},
                "league_history_matches":model.get("league_history_matches")}
        forecast=(xg_market_shadow if method=="v22_research" else market_shadow)(
            shaped,packet,as_of=decision)
        if not forecast.get("ready"):
            raise ValueError("original_challenger_cannot_be_reproduced")
        grid=forecast["matrix"]
        if method=="v22_research":
            fit=fitted_temperature(originals,finals,as_of=decision,
                                   league=str(model.get("competition") or "unknown"))
            if fit.get("ready"):
                expected={"training_samples":fit["sample_size"],
                          "training_ended_at_utc":fit["training_end_at_utc"],
                          "temperature":fit["temperature"],"status":"past_only_shadow_not_promoted"}
                grid=scale(grid,fit["temperature"])
            else:
                expected={"status":"abstained_insufficient_verified_history",
                          "training_samples":fit.get("sample_size",0),
                          "minimum_required":fit.get("min_required")}
            if not _same(pair["calibration"][index],expected):
                raise ValueError("calibration_not_reproduced_from_pre_decision_results")
        baseline.append({"scores":top_scores(model["score_matrix"],10)})
        challenger.append({"scores":top_scores(grid,10)})
    reconstructed={}
    coverage={}
    for label,rows in (("baseline_v8",baseline),(method,challenger)):
        picks,mass=_options(*rows)
        if len(picks)!=5 or len({(p["score1"],p["score2"]) for p in picks})!=5:
            raise ValueError("five_distinct_supported_combinations_required")
        reconstructed[label]=picks;coverage[label]=mass
    if not _same(pair["doubles"],reconstructed) or not _same(pair["model_estimated_coverage"],coverage):
        raise ValueError("saved_choices_or_coverage_not_original_model_optimum")
    return models,reconstructed


def evaluate(*,root=Path("data/v22_prospective"),source_root=ORIGINAL_ROOT,
             provider_root=PROVIDER_ROOT,prematch_root=Path("data/prematch"),
             research_method="v22_research",verified=None,now=None):
    """Return compatible counters only after original source reconstruction."""
    if research_method not in METHODS:
        raise ValueError("unsupported_research_method")
    clock=utc(now or datetime.now(timezone.utc))
    if verified is None:
        finals,conflicts,result_rejections=_result_index(source_root,prematch_root)
        calibration_finals=load_json(Path(source_root)/"verified_results.json",{})
    else:
        finals={(str(row.get("provider")),str(row.get("fixture_id"))):row
                for row in verified.values() if isinstance(row,dict)}
        conflicts=set();result_rejections={};calibration_finals=verified
    originals=list(read_latest(source_root).values())
    counts={name:{"verified":0,"hits":0,"pending":0} for name in ("baseline_v8",research_method)}
    rejected=Counter();records=[];seen_ids=set();seen_events=set();entries=[]
    for path in sorted((Path(root)/"pairs").glob("*.json")):
        try:
            pair=load_json(path,{})
            entries.append((utc(pair["captured_at_utc"]),path,pair))
        except (ValueError,TypeError,KeyError,OSError,AttributeError):
            rejected["unreadable_pair_or_capture_time"]+=1
    for _,path,pair in sorted(entries,key=lambda item:(item[0],item[1].name)):
        try:
            models,picks=_reconstruct(pair,path,source_root=source_root,provider_root=provider_root,
                                     method=research_method,originals=originals,
                                     finals=calibration_finals,now=clock)
            identities={(model["provider"],str(model["fixture_id"])) for model in models}
            events={(utc(model["kickoff_utc"]).isoformat(),
                     " ".join(model["home_team"].casefold().split()),
                     " ".join(model["away_team"].casefold().split())) for model in models}
            if identities&seen_ids or events&seen_events:
                raise ValueError("repeated_fixture_or_cross_provider_event")
            seen_ids.update(identities);seen_events.update(events)
            actual=[_final(model,finals.get((model["provider"],str(model["fixture_id"]))),clock)
                    for model in models]
            finished=all(score is not None for score in actual)
            scores=[f"{h}-{a}" for h,a in actual] if finished else None
            flags={}
            for method in counts:
                flags[method]=(any(p["score1"]==scores[0] and p["score2"]==scores[1]
                                   for p in picks[method]) if finished else None)
                counts[method]["verified" if finished else "pending"]+=1
                counts[method]["hits"]+=int(flags[method] is True)
            records.append({"date":pair["sydney_date"],"scores":scores,
                            "hits":flags,"verified":finished})
        except (ValueError,TypeError,KeyError,OSError,AttributeError,IndexError,OverflowError) as exc:
            rejected[str(exc) if isinstance(exc,ValueError) else type(exc).__name__]+=1
    for item in counts.values():
        item["hit_rate"]=item["hits"]/item["verified"] if item["verified"] else None
        item["wilson_95"]=wilson(item["hits"],item["verified"])
    return {"schema":2,"evaluated_at_utc":clock.isoformat(),
            "kind":"source_reconstructed_immutable_same_pair_verified_final",
            "pairs":records,"methods":counts,"research_method":research_method,
            "evidence_rejections":dict(rejected),"result_rejections":dict(result_rejections),
            "conflicting_result_identities":len(conflicts),
            "calibration_and_promotion":"unproven_shadow_only","ai_panel_promoted":False,
            "model_promoted":False,"external_requests":0,"target_70_percent_proven":False,
            "intervals_are_descriptive_independence_not_established":True}


def main():
    summaries={}
    for version,method in (("v22_prospective","v22_research"),("v23_prospective","v23_market_form")):
        root=Path("data")/version
        report=evaluate(root=root,research_method=method)
        write_json(root/"performance.json",report)
        summaries[version]={"methods":report["methods"],"evidence_rejections":report["evidence_rejections"]}
    return summaries


if __name__=="__main__":
    print(json.dumps(main(),indent=2))
