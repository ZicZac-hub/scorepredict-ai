"""V23 frozen same-pair market+form vs untouched V8.

No API calls. Adds a separate auditable path when genuine shot xG is absent,
but two time-checked goal/winner market types exist. No model is promoted.
"""
import json
from datetime import datetime,timedelta,timezone
from pathlib import Path

from persist_prematch import load_json,write_json
from v10_prospective import ROOT as V8_ROOT,read_latest,utc
from v16_pair_research import original_model
from v21_capture_sources import best_packet
from v21_prospective import _write_once
from v22_prospective import (
    _fixture,_short,_options,rank_pairs as rank_common,
    evaluate as strict_evaluate)
from v23_market_form import shadow as score_market_form

ROOT=Path("data/v23_prospective")
SOURCE_ROOT=Path("data/v21_sources")
METHOD="v23_market_form"


def frozen_candidates(*,now,source_root=V8_ROOT,provider_root=SOURCE_ROOT,
                      originals=None):
    clock=utc(now)
    rows=originals if originals is not None else read_latest(source_root)
    rows=list(rows.values()) if isinstance(rows,dict) else list(rows)
    found={}
    for row in rows:
        try:
            if row.get("model_key") not in ("v8","v8_near"):
                continue
            kickoff=utc(row["kickoff_utc"])
            if not clock+timedelta(minutes=5)<kickoff<=clock+timedelta(hours=36):
                continue
            source=original_model(source_root,row)
            if not source or source.get("input_digest")!=row.get("input_digest"):
                continue
            if int(source.get("league_history_matches") or 0)<45:
                continue
            evidence=best_packet(provider_root,source,now=clock)
            if not evidence:
                continue
            packet=evidence["provider_snapshot"]
            fixture=_fixture(source)
            model=score_market_form(
                {"fixture":fixture,
                 "result":{"matrix":source["score_matrix"]},
                 "league_history_matches":source["league_history_matches"]},
                packet,as_of=clock)
            if not model.get("ready"):
                continue
            old={"scores":[[int(h),int(a),float(p)] for h,a,p in source["top10_scores"]]}
            new={"scores":model["scores"],"probabilities":model}
            key=(str(source["provider"]),str(source["fixture_id"]))
            prior=found.get(key)
            if prior is None or (source["model_key"]=="v8_near" and
                                  prior["source"]["model_key"]!="v8_near"):
                found[key]={"source":source,"v8":old,"v22":new,
                            "snapshot_hash":packet["snapshot_sha256"]}
        except (ValueError,KeyError,TypeError,IndexError,OSError):
            continue
    return list(found.values())


def freeze(*,now=None,root=ROOT,source_root=V8_ROOT,provider_root=SOURCE_ROOT):
    clock=utc(now or datetime.now(timezone.utc))
    choices=frozen_candidates(now=clock,source_root=source_root,
                              provider_root=provider_root)
    ranked=rank_common(choices)
    per_day={}
    for match in ranked:
        per_day.setdefault(match["day"],match)
    saved=[]
    for day,pair in per_day.items():
        leg1,leg2=pair["a"]["source"],pair["b"]["source"]
        if clock>=min(utc(x["kickoff_utc"]) for x in (leg1,leg2)):
            continue
        record={"schema":1,"status":"frozen_pre_kickoff",
                "sydney_date":day,"captured_at_utc":clock.isoformat(),
                "policy":"time_checked_market_and_original_goals_when_xg_unavailable",
                "first":_short(leg1),"second":_short(leg2),
                "source_packet_hashes":[pair["a"]["snapshot_hash"],
                                        pair["b"]["snapshot_hash"]],
                "doubles":{"baseline_v8":pair["baseline"],
                           METHOD:pair["challenger"]},
                "model_estimated_coverage":{
                    "baseline_v8":pair["baseline_coverage"],
                    METHOD:pair["v22_coverage"]},
                "no_production_promotion":True,
                "calibrated_hit_rate_claim":False,
                "source_fallback":"genuine_xg_is_optional_never_invented"}
        if _write_once(Path(root)/"pairs"/(day+".json"),record):
            saved.append(day)
    status={"checked_at_utc":clock.isoformat(),
            "eligible_source_qualified_fixtures":len(choices),
            "qualifying_pairs":len(ranked),"newly_frozen_days":saved,
            "abstained":not ranked,"model_estimates_are_not_observed_accuracy":True}
    write_json(Path(root)/"status.json",status)
    return status


def evaluate(*,root=ROOT,source_root=V8_ROOT,verified=None):
    return strict_evaluate(root=root,source_root=source_root,verified=verified,
                           research_method=METHOD)


def main():
    status=freeze()
    verified=evaluate()
    write_json(ROOT/"performance.json",verified)
    return {"status":status,"verified":verified["methods"]}


if __name__=="__main__":
    print(json.dumps(main(),indent=2))
