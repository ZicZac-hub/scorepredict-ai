"""V22 immutable pre-kickoff five-double shadow versus untouched V8.

Read-only frozen original V8 inputs and source-matched provider packets.
No football/AI API requests. The new method cannot be auto-promoted.
"""
import json
from collections import defaultdict
from datetime import datetime,timedelta,timezone
from itertools import combinations
from pathlib import Path
from zoneinfo import ZoneInfo
import numpy as np

from persist_prematch import load_json,write_json
from v10_prospective import ROOT as V8_ROOT,read_latest,utc
from v16_pair_research import original_model
from v21_capture_sources import source_packet_path,qualified_packet
from v21_prospective import _write_once,_verified
from v22_shadow import shadow
from v22_calibration import fitted_temperature,scale
from v21_goal_model_comparison import top_scores

ROOT=Path("data/v22_prospective")
SOURCES=Path("data/v21_sources")
SYDNEY=ZoneInfo("Australia/Sydney")
MAX_LOOKAHEAD_HOURS=36
MIN_LEAD_MINUTES=5
MIN_COVERAGE=.055 # Uncalibrated model-coverage gate, not a hit-rate guarantee.


def _fixture(row):
    return {"id":row["fixture_id"],"provider":row["provider"],
            "utcDate":row["kickoff_utc"],
            "homeTeam":{"id":row["home_team_id"],"name":row.get("home_team","")},
            "awayTeam":{"id":row["away_team_id"],"name":row.get("away_team","")}}

def _short(row):
    return {k:row[k] for k in ("provider","fixture_id","forecast_id",
        "input_digest","home_team_id","away_team_id","kickoff_utc")}


def _options(a,b):
    one=a["scores"][:5]
    two=b["scores"][:5]
    choices=sorted(({"score1":f"{h}-{g}","score2":f"{u}-{v}",
                     "joint_probability":float(p*q)}
                    for h,g,p in one for u,v,q in two),
                   key=lambda x:(-x["joint_probability"],x["score1"],x["score2"]))[:5]
    return choices, sum(x["joint_probability"] for x in choices)


def frozen_candidates(*,now,source_root=V8_ROOT,provider_root=SOURCES,
                      originals=None):
    clock=utc(now)
    eligible={}
    rows=(originals if originals is not None else read_latest(source_root))
    source_rows=list(rows.values()) if isinstance(rows,dict) else list(rows)
    finals=load_json(Path(source_root)/"verified_results.json",{})
    per_league_fit={}
    for row in source_rows:
        try:
            if row["model_key"] not in ("v8","v8_near"):continue
            kickoff=utc(row["kickoff_utc"])
            if not clock+timedelta(minutes=MIN_LEAD_MINUTES)<kickoff<=clock+timedelta(hours=MAX_LOOKAHEAD_HOURS):
                continue
            source=original_model(source_root,row)
            if not source or source.get("input_digest")!=row.get("input_digest"):
                continue
            packet_row=load_json(source_packet_path(provider_root,row),{})
            if not qualified_packet(packet_row,row,now=clock):
                continue
            packet=packet_row["provider_snapshot"]
            shaped={"fixture":_fixture(source),
                    "result":{"matrix":np.asarray(source["score_matrix"]).tolist()}}
            forecast=shadow(shaped,packet,as_of=clock)
            if not forecast.get("ready"):
                continue
            league=str(source.get("competition") or "unknown")
            if league not in per_league_fit:
                per_league_fit[league]=fitted_temperature(
                    source_rows,finals,as_of=clock,league=league)
            fit=per_league_fit[league]
            if fit.get("ready"):
                adjusted=scale(forecast["matrix"],fit["temperature"])
                forecast={**forecast,"matrix":adjusted,
                          "scores":top_scores(adjusted),
                          "temporal_calibration":{
                              "training_samples":fit["sample_size"],
                              "training_ended_at_utc":fit["training_end_at_utc"],
                              "temperature":fit["temperature"],
                              "status":"past_only_shadow_not_promoted"}}
            else:
                forecast={**forecast,"temporal_calibration":{
                    "status":"abstained_insufficient_verified_history",
                    "training_samples":fit.get("sample_size",0),
                    "minimum_required":fit.get("min_required")}}
            baseline={"scores":[[int(h),int(a),float(p)] for h,a,p in source["top10_scores"]]}
            key=(str(row["provider"]),str(row["fixture_id"]))
            prior=eligible.get(key)
            # Prefer the predeclared closer-to-kickoff V8_near when properly
            # frozen; NEVER sort using the final result or hit statistics.
            if prior is None or (row["model_key"]=="v8_near" and prior["source"]["model_key"]!="v8_near"):
                eligible[key]={"source":source,"v22":forecast,"v8":baseline,
                               "snapshot_hash":packet["snapshot_sha256"]}
        except (ValueError,KeyError,TypeError,OSError):
            continue
    return list(eligible.values())


def rank_pairs(candidates,*,min_coverage=MIN_COVERAGE):
    days=defaultdict(list)
    for row in candidates:
        src=row["source"]
        days[utc(src["kickoff_utc"]).astimezone(SYDNEY).date().isoformat()].append(row)
    pairs=[]
    for day,choices in days.items():
        for a,b in combinations(choices,2):
            ids=[(str(row["source"]["provider"]),str(row["source"][key]))
                 for row in (a,b) for key in ("home_team_id","away_team_id")]
            if len(set(ids))!=4:
                continue
            a_first,b_first=(a,b) if str(a["source"]["fixture_id"])<=str(b["source"]["fixture_id"]) else (b,a)
            first_base,coverage_base=_options(a_first["v8"],b_first["v8"])
            first_v22,coverage_v22=_options(a_first["v22"],b_first["v22"])
            lower=min(coverage_base,coverage_v22)
            if lower<min_coverage:continue
            # Disclose uncertain source quality, not use it as a fabricated
            # scalar confidence. All candidates passed real xG+markets gates.
            pairs.append({"day":day,"a":a_first,"b":b_first,
                          "baseline":first_base,"challenger":first_v22,
                          "baseline_coverage":coverage_base,
                          "v22_coverage":coverage_v22,
                          "conservative_coverage":lower})
    return sorted(pairs,key=lambda x:(-x["conservative_coverage"],x["day"],
              str(x["a"]["source"]["fixture_id"]),str(x["b"]["source"]["fixture_id"])))


def freeze(*,now=None,root=ROOT,source_root=V8_ROOT,provider_root=SOURCES):
    clock=utc(now or datetime.now(timezone.utc))
    candidates=frozen_candidates(now=clock,source_root=source_root,provider_root=provider_root)
    options=rank_pairs(candidates)
    first_by_day={}
    for option in options:
        first_by_day.setdefault(option["day"],option)
    saved=[]
    for day,choice in first_by_day.items():
        first,second=choice["a"],choice["b"]
        legs=(first["source"],second["source"])
        if clock>=min(utc(x["kickoff_utc"]) for x in legs):
            continue
        record={"schema":1,"policy":"original_pair_same_verified_xg_and_market_gate",
                "status":"frozen_pre_kickoff","captured_at_utc":clock.isoformat(),
                "sydney_date":day,"first":_short(legs[0]),"second":_short(legs[1]),
                "source_packet_hashes":[first["snapshot_hash"],second["snapshot_hash"]],
                "calibration": [
                    first["v22"].get("temporal_calibration"),
                    second["v22"].get("temporal_calibration")],
                "doubles":{"baseline_v8":choice["baseline"],
                           "v22_research":choice["challenger"]},
                "model_estimated_coverage":{
                    "baseline_v8":choice["baseline_coverage"],
                    "v22_research":choice["v22_coverage"]},
                "no_production_promotion":True,"ai_votes_numerical_weight":0.0,
                "lineup_weight":"no_fitted_player_model"}
        if _write_once(Path(root)/"pairs"/(day+".json"),record):
            saved.append(day)
    status={"checked_at_utc":clock.isoformat(),"eligible_source_qualified_fixtures":len(candidates),
            "pairs_above_threshold":len(options),"newly_frozen_days":saved,
            "abstained_no_qualified_pair":len(options)==0,
            "source_packets_required_for_both_legs":True,
            "coverage_is_not_empirical_accuracy":True}
    write_json(Path(root)/"status.json",status)
    return status


def evaluate(*,root=ROOT,source_root=V8_ROOT,verified=None):
    finals=verified if verified is not None else load_json(
        Path(source_root)/"verified_results.json",{})
    counts={k:{"verified":0,"hits":0,"pending":0}
            for k in ("baseline_v8","v22_research")}
    records=[]
    for path in sorted((Path(root)/"pairs").glob("*.json")):
        pair=load_json(path,{})
        if pair.get("status")!="frozen_pre_kickoff":
            continue
        try:
            legs=(pair["first"],pair["second"])
            base=(original_model(source_root,legs[0]),
                  original_model(source_root,legs[1]))
            if not all(base[i] and str(base[i]["input_digest"])==
                       str(legs[i]["input_digest"]) for i in (0,1)):
                continue
            if utc(pair["captured_at_utc"])>=min(utc(x["kickoff_utc"]) for x in legs):
                continue
            results=(finals.get(str(legs[0]["fixture_id"])),
                     finals.get(str(legs[1]["fixture_id"])))
            finished=all(_verified(legs[i],results[i]) for i in (0,1))
            scores=tuple(f"{x['home_goals']}-{x['away_goals']}" for x in results) if finished else None
            flags={}
            for name in counts:
                choices=pair["doubles"][name]
                if len(choices)!=5:continue
                if finished:
                    counts[name]["verified"]+=1
                    hit=any(x["score1"]==scores[0] and x["score2"]==scores[1] for x in choices)
                    counts[name]["hits"]+=int(hit)
                    flags[name]=bool(hit)
                else:
                    counts[name]["pending"]+=1
                    flags[name]=None
            records.append({"date":pair["sydney_date"],"scores":scores,"hits":flags,
                            "verified":bool(finished)})
        except (ValueError,KeyError,TypeError):
            continue
    for m in counts.values():
        m["hit_rate"]=m["hits"]/m["verified"] if m["verified"] else None
    return {"schema":1,"evaluated_at_utc":datetime.now(timezone.utc).isoformat(),
            "kind":"only_immutable_frozen_matching_verified_final",
            "pairs":records,"methods":counts,
            "calibration_and_promotion":"unproven_shadow_only",
            "ai_panel_promoted":False,"model_promoted":False}


def main():
    saved=freeze()
    report=evaluate()
    write_json(ROOT/"performance.json",report)
    return {"frozen":saved,"performance":report["methods"]}


if __name__=="__main__":
    print(json.dumps(main(),indent=2))
