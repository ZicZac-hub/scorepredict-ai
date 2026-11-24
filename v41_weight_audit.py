"""Read-only chronological weight sensitivity audit for V41.

No new predictions or promotion. The old forecasts are genuine; applying
newly defined weights retrospectively is NOT a prospective V41 result.
Selection uses only independently verified results known before each source
capture, excludes the target event, and never tunes a continuous parameter.
"""
from datetime import datetime, timezone
from pathlib import Path
import math
import numpy as np
from five_double_accuracy import utc
from v26_temporal_score import finished_samples
from v41_unified_ensemble import core, normalized
from v39_poisson_marginal import transform_score_grid
from v25_score_accuracy_research import describe_score, summarize_scores

ROOT = Path("data/v41_weight_audit")
METHODS = ("fixed_v41", "baseline_anchored", "poisson_anchored")
MIN_PRIOR = 30
MIN_HOLDOUT = 50
# A fixed loss penalty for leaving the original policy; not fitted.
SWITCH_PENALTY = .01


def distributions(grid):
    base=normalized(grid)
    fixed=core(base)
    poisson=normalized(transform_score_grid(base,"poisson_marginal"),base.shape)
    return {"fixed_v41":fixed,
            "baseline_anchored":normalized(.5*fixed+.5*base),
            "poisson_anchored":normalized(.5*fixed+.5*poisson)}


def identity(row):
    source=row["source"]
    return str(source["provider"]),str(source["fixture_id"])


def physical(row):
    source=row["source"]
    names=tuple(" ".join(str(source.get(k) or "").casefold().split())
                for k in ("home_team","away_team"))
    return (utc(row["kickoff"]).isoformat(),*names) if all(names) else ("provider",*identity(row))


def audit(samples, *, now):
    clock=utc(now)
    prepared=[]
    exclusions=0
    ids=set()
    events=set()
    for row in sorted(samples,key=lambda r:(utc(r["captured"]),identity(r))):
        try:
            captured,kickoff,verified=map(utc,(row["captured"],row["kickoff"],row["verified_at"]))
            event=physical(row)
            if not captured<kickoff<verified<=clock or identity(row) in ids or event in events:
                exclusions+=1
                continue
            grids=distributions(row["grid"])
            scores={name:describe_score(grid,row["score"]) for name,grid in grids.items()}
            ids.add(identity(row)); events.add(event)
            prepared.append((row,scores))
        except (ValueError,TypeError,KeyError,OverflowError):
            exclusions+=1
    # Fixed-policy same-fixture sensitivities are descriptive only.
    fixed={name:summarize_scores([s[name] for _,s in prepared]) for name in METHODS}
    holdout=[]; reference=[]; chosen_scores=[]; max_prior=0
    for row,scores in prepared:
        prior=[s for other,s in prepared if identity(other)!=identity(row)
               and utc(other["verified_at"])<=utc(row["captured"])]
        max_prior=max(max_prior,len(prior))
        if len(prior)<MIN_PRIOR:
            continue
        penalized={name:math.fsum(s[name]["exact_score_log_loss"] for s in prior)/len(prior)
                        +(0 if name=="fixed_v41" else SWITCH_PENALTY) for name in METHODS}
        chosen=min(METHODS,key=lambda name:penalized[name])
        reference.append(scores["fixed_v41"])
        chosen_scores.append(scores[chosen])
        holdout.append({"provider":identity(row)[0],"fixture_id":identity(row)[1],
                        "forecast_id":row["source"]["forecast_id"],
                        "captured_at_utc":utc(row["captured"]).isoformat(),
                        "prior_verified_count":len(prior),"chosen_method":chosen,
                        "fixed_log_loss":scores["fixed_v41"]["exact_score_log_loss"],
                        "selected_log_loss":scores[chosen]["exact_score_log_loss"]})
    return {"evaluated_at_utc":clock.isoformat(),"kind":"retrospective_chronological_single_fixture_audit",
            "verified_unique_fixtures":len(prepared),"excluded":exclusions,
            "maximum_prior_before_historical_capture":max_prior,
            "minimum_prior":MIN_PRIOR,"minimum_holdout":MIN_HOLDOUT,
            "chronological_holdout_count":len(holdout),
            "holdout_minimum_met":len(holdout)>=MIN_HOLDOUT,
            "fixed_sensitivity_metrics_NOT_prospective":fixed,
            "chronological_fixed":summarize_scores(reference),
            "chronological_selected":summarize_scores(chosen_scores),
            "holdout_receipts":holdout,"original_forecasts_changed":False,
            "no_live_weight_change":True,"no_automatic_promotion":True,
            "not_a_five_double_trial":True,"target_70_percent_proven":False}


def run(*, now=None, out=ROOT):
    from persist_prematch import write_json
    clock=utc(now or datetime.now(timezone.utc))
    samples, quality=finished_samples(now=clock)
    report=audit(samples,now=clock)
    report["source_quality"]=quality
    write_json(Path(out)/"status.json",report)
    return report


if __name__=="__main__":
    import json
    report=run()
    print(json.dumps({k:v for k,v in report.items() if k!="holdout_receipts"},indent=2))
