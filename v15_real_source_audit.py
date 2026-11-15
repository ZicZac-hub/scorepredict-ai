"""V15 REAL saved-source audit. Read-only; NEVER calls AI/football APIs.

Count configured services, genuinely frozen individual model answers,
whether BOTH legs of frozen historical daily five-double pairs have AI
responses, and genuine independently verified performance. Do not
infer missing future votes or backfill historic predictions.
"""
import argparse
import json
from collections import Counter
from pathlib import Path

from persist_prematch import load_json
from v10_prospective import ROOT as MATCH_ROOT, utc
from llm_research_panel import ROOT as AI_ROOT, PROVIDERS

def audit(base_root=MATCH_ROOT, ai_root=AI_ROOT):
    base_root=Path(base_root);ai_root=Path(ai_root)
    status=load_json(ai_root/"status.json",{})
    evaluation=load_json(ai_root/"evaluation.json",{})
    performance=load_json(base_root/"performance.json",{})
    forecasts=[]
    provider_counts=Counter()
    for filepath in sorted((ai_root/"forecasts").glob("*.json")):
        row=load_json(filepath,{})
        if not isinstance(row,dict) or not row.get("research_only"):
            continue
        try:
            captured=utc(row["captured_at_utc"])
            kickoff=utc(row["kickoff_utc"])
            if captured>=kickoff:
                continue
            votes={key:value for key,value in (row.get("votes") or {}).items()
                   if key in PROVIDERS and isinstance(value,dict)
                   and isinstance(value.get("score"),list)
                   and len(value["score"])==2}
            if not votes:
                continue
            identity=(str(row["provider"]),str(row["fixture_id"]))
            forecasts.append({"fixture":identity,"kickoff":kickoff,
                              "provider_votes":set(votes)})
            provider_counts.update(votes.keys())
        except (TypeError,KeyError,ValueError):
            continue
    sets=[];complete=0;partial=0
    for filepath in sorted((base_root/"doubles").glob("*.json")):
        row=load_json(filepath,{})
        if row.get("mode")!="prospective" or row.get("status")!="frozen_pre_kickoff":
            continue
        try:
            legs=[row["first"],row["second"]]
            counts=[]
            for leg in legs:
                match_id=(str(leg["provider"]),str(leg["fixture_id"]))
                kickoff=utc(leg["kickoff_utc"])
                known=[item for item in forecasts if item["fixture"]==match_id
                       and item["kickoff"]==kickoff]
                counts.append(len(known[0]["provider_votes"]) if known else 0)
            valid=all(n>=2 for n in counts)
            complete+=int(valid);partial+=int(any(n>0 for n in counts))
            sets.append({"sydney_day":row.get("sydney_date"),
                         "actual_provider_votes_per_leg":counts,
                         "both_legs_have_at_least_two_providers":valid})
        except (TypeError,KeyError,ValueError):
            continue
    double_metrics=performance.get("five_doubles") or {}
    return {
        "kind":"read_only_v15_real_ai_source_coverage_audit",
        "configuration_is_not_successful_votes":True,
        "configured_provider_names":status.get("configured_providers") or [],
        "configured_provider_count":len(status.get("configured_providers") or []),
        "latest_scheduled_attempted":status.get("attempted"),
        "latest_scheduled_captured":status.get("captured"),
        "saved_valid_individual_ai_fixture_records":len(forecasts),
        "actual_successful_provider_counts_by_name":dict(provider_counts),
        "unique_providers_with_valid_saved_votes":len(provider_counts),
        "frozen_original_double_sets_checked":len(sets),
        "original_sets_with_any_individual_ai_evidence":partial,
        "original_sets_with_at_least_two_ai_votes_on_both_legs":complete,
        "complete_five_ai_panels_with_verified_results":
            evaluation.get("complete_panels_with_verified_results",0),
        "complete_ten_ai_panels_with_verified_results":
            evaluation.get("ten_provider_complete_verified",0),
        "actually_verified_original_five_doubles":double_metrics.get("verified_pairs",0),
        "actually_verified_original_five_double_wins":double_metrics.get("wins",0),
        "original_five_doubles_pending":double_metrics.get("awaiting_both_results",0),
        "v15_real_ai_double_accuracy":None,
        "v15_approval_for_production":False,
        "explanation":"A genuine two-fixture V15 score cannot be accuracy-tested before both legs have real earlier-frozen votes and independently verified final scores.",
        "source_rows":sets,
    }


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-root",default=str(MATCH_ROOT))
    p.add_argument("--ai-root",default=str(AI_ROOT))
    p.add_argument("--output",default="work/v15_real_source_audit.json")
    args=p.parse_args()
    report=audit(Path(args.base_root),Path(args.ai_root))
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    print("===== ACTUAL SAVED V15 AI SOURCE COVERAGE AUDIT =====",flush=True)
    print(json.dumps({k:v for k,v in report.items() if k!="source_rows"},indent=2),
          flush=True)


if __name__=="__main__":
    main()
