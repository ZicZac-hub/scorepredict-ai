"""Read-only comparison of independently authenticated early/late V42 trials."""
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import v42_research_trial as early
import v42_late_research_trial as late
from persist_prematch import write_json

def _index(report):
    rows = report["rows"]
    result = {r["sydney_date"]:r for r in rows}
    if len(result)!=len(rows):
        raise ValueError("duplicate_study_day")
    return result

def _arm_summary(rows):
    issued = sum(r["methods"]["unified_final"]["status"] in ("pending","verified") for r in rows)
    verified = [r["methods"]["unified_final"]["metrics"] for r in rows if r["methods"]["unified_final"]["status"]=="verified"]
    n = len(verified)
    hits = sum(m["hit"] for m in verified)
    return {"valid_pairs":len(rows),"issued":issued,"pending":issued-n,
            "abstained":len(rows)-issued,"verified":n,"hits":hits,
            "hit_rate":hits/n if n else None,"wilson_95":early.wilson(hits,n),
            "score_log_loss":math.fsum(m["score_log_loss"] for m in verified)/n if n else None,
            "outcome_brier":math.fsum(m["outcome_brier"] for m in verified)/n if n else None,
            "paired_xg_used":sum(r["unified_final_evidence"].get("paired_xg_used") is True for r in rows),
            "paired_market_used":sum(r["unified_final_evidence"].get("paired_market_used") is True for r in rows),
            "both_fixtures_confirmed_xi":sum(all(x["both_confirmed_xis"] for x in r["lineup_context"]) for r in rows)}

def compare_reports(a,b,*,early_root,late_root):
    # Inputs are fresh evaluate() results, not unvalidated cached JSON summaries.
    ai,bi=_index(a),_index(b)
    rejects=Counter()
    paired=[]
    settled=[]
    for day in sorted(ai.keys() & bi.keys()):
        x,y=ai[day],bi[day]
        try:
            er=early._load(Path(early_root)/"trials"/(day+".json"))
            lr=late._load(Path(late_root)/"trials"/(day+".json"))
            if er["baseline_pair"]!=lr["baseline_pair"] or er["source_forecast_refs"]!=lr["source_forecast_refs"]:
                raise ValueError("different_pair_identity")
            e,l=early.utc(er["decision_as_of_utc"]),early.utc(lr["decision_as_of_utc"])
            if l<=e:
                raise ValueError("late_cutoff_not_strictly_later")
            if x["actual_scores"]!=y["actual_scores"] or x["status"]!=y["status"]:
                raise ValueError("settlement_disagreement")
            paired.append((x,y,(l-e).total_seconds()))
            if all(r["methods"]["unified_final"]["status"]=="verified" for r in (x,y)):
                settled.append((x,y))
        except (ValueError,KeyError,TypeError,OSError,AttributeError) as exc:
            rejects[str(exc) if isinstance(exc,ValueError) else type(exc).__name__]+=1
    em=[x["methods"]["unified_final"]["metrics"] for x,y in settled]
    lm=[y["methods"]["unified_final"]["metrics"] for x,y in settled]
    def advantage(k):
        return early._mean_interval([x[k]-y[k] for x,y in zip(em,lm)])
    return {"early_all":_arm_summary(list(ai.values())),"late_all":_arm_summary(list(bi.values())),
            "early_only_days":len(ai.keys()-bi.keys()),"late_only_days":len(bi.keys()-ai.keys()),
            "matched_valid_pairs":len(paired),"matched_verified_issued_pairs":len(settled),
            "matched_pending_pairs":sum(x["status"]=="pending" for x,y,_ in paired),
            "matched_early":_arm_summary([x for x,y in settled]),
            "matched_late":_arm_summary([y for x,y in settled]),
            "matched_early_evidence":_arm_summary([x for x,y,_ in paired]),
            "matched_late_evidence":_arm_summary([y for x,y,_ in paired]),
            "early_only_wins":sum(x["hit"] and not y["hit"] for x,y in zip(em,lm)),
            "late_only_wins":sum(y["hit"] and not x["hit"] for x,y in zip(em,lm)),
            "score_log_loss_advantage":advantage("score_log_loss"),
            "outcome_brier_advantage":advantage("outcome_brier"),
            "mean_cutoff_delay_seconds":math.fsum(d for _,_,d in paired)/len(paired) if paired else None,
            "pair_rejections":dict(rejects),"minimum_review_verified_pairs":100,
            "target_70_percent_proven":False,"model_promoted":False,"external_requests":0,
            "limitations":["Same authenticated original pair and strictly later cutoff required.",
                "Matched metrics exclude abstentions; all-arm counts and unmatched days separately reported.",
                "Timing comparison does not establish a causal effect or betting profitability.",
                "Lineups are contextual only and have zero numerical weight.",
                "Uncertainty is descriptive and serial dependence may invalidate normal intervals."]}

def main(*,early_root=early.ROOT,late_root=late.ROOT,now=None,**kwargs):
    clock=early.utc(now or datetime.now(timezone.utc))
    a=early.evaluate(root=early_root,now=clock,**kwargs)
    b=late.evaluate(root=late_root,now=clock,**kwargs)
    report=compare_reports(a,b,early_root=early_root,late_root=late_root)
    return {"version":"v42-matched-cutoff-audit-v1","evaluated_at_utc":clock.isoformat(),**report}

if __name__=="__main__":
    report=main()
    write_json(late.ROOT/"cutoff_comparison.json",report)
    print(json.dumps(report,indent=2))
