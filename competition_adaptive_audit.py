"""Chronological competition-aware goal-family audit, retrospective only.

Use immutable original V8 grids and independently verified regulation finals.
Learn from results available before each original capture, not future scores.
No new provider requests, live predictions, or five-double issuance.
"""
from collections import Counter
from datetime import datetime, timezone
import json
import math

import numpy as np
from five_double_accuracy import utc
from v26_temporal_score import finished_samples
from v41_weight_audit import identity, physical
from v21_goal_model_comparison import variants, MODEL_NAMES
from v25_score_accuracy_research import describe_score, summarize_scores

MIN_GLOBAL = 30
MIN_LOCAL = 10
SHRINKAGE = 50
BASELINE_ANCHOR = .5


def family_grids(grid):
    families = variants(grid)
    if any(p.shape != families["baseline_v8"].shape for p in families.values()):
        raise ValueError("incompatible_score_support")
    return families


def mixture_weights(prior, competition=None):
    """Local evidence shrinks towards a global comparison of the same families."""
    if len(prior) < MIN_GLOBAL:
        return None
    def losses(rows):
        return np.array([math.fsum(r["scores"][k]["exact_score_log_loss"] for r in rows)/len(rows)
                         for k in MODEL_NAMES])
    global_loss = losses(prior)
    local = [r for r in prior if r["competition"] == competition] if competition else []
    weight = len(local)/(len(local)+SHRINKAGE) if len(local)>=MIN_LOCAL else 0.
    blended = (1-weight)*global_loss + weight*losses(local) if weight else global_loss
    relative = np.exp(-4*(blended-blended.min()))
    weights = (1-BASELINE_ANCHOR)*relative/relative.sum()
    weights[0] += BASELINE_ANCHOR
    return weights, len(local), weight


def mix(grids, weights):
    p = sum(float(w)*grids[name] for w,name in zip(weights,MODEL_NAMES))
    return p/p.sum()


def audit(samples, *, now):
    clock=utc(now)
    prepared=[]
    seen, events=set(),set()
    invalid=0
    for row in sorted(samples,key=lambda r:(utc(r["captured"]),identity(r))):
        try:
            key, event = identity(row),physical(row)
            captured,kickoff,verified=map(utc,(row["captured"],row["kickoff"],row["verified_at"]))
            if not captured<kickoff<verified<=clock or key in seen or event in events:
                raise ValueError("invalid_or_duplicate_event")
            grids=family_grids(row["grid"])
            scores={name:describe_score(p,row["score"]) for name,p in grids.items()}
            league=row["source"].get("competition")
            # Provider-scoped league code only; no unknown-league pooling.
            comp=(key[0],league) if isinstance(league,str) and league.strip() else None
            prepared.append({"row":row,"key":key,"competition":comp,"grids":grids,
                             "scores":scores,"captured":captured,"verified":verified})
            seen.add(key);events.add(event)
        except (ValueError,TypeError,KeyError,OverflowError):
            invalid+=1
    metrics={k:[] for k in ("baseline","global","competition")}
    local_metrics={k:[] for k in metrics}
    receipts=[]; max_prior=0; competitions=Counter()
    for target in prepared:
        prior=[r for r in prepared if r["key"]!=target["key"] and
               r["verified"]<=target["captured"]]
        max_prior=max(max_prior,len(prior))
        glob=mixture_weights(prior)
        loc=mixture_weights(prior,target["competition"])
        if glob is None:
            continue
        g,_,_=glob
        l,n,w=loc
        row=target["row"]; grids=target["grids"]
        scored={"baseline":target["scores"]["baseline_v8"],
                "global":describe_score(mix(grids,g),row["score"]),
                "competition":describe_score(mix(grids,l),row["score"])}
        for k in metrics:
            metrics[k].append(scored[k])
            if w: local_metrics[k].append(scored[k])
        if w: competitions["/".join(target["competition"])]+=1
        receipts.append({"provider":target["key"][0],"fixture_id":target["key"][1],
                         "forecast_id":row["source"]["forecast_id"],
                         "captured_at_utc":target["captured"].isoformat(),
                         "prior_global":len(prior),"prior_competition":n,
                         "competition_weight":w,"global_weights":dict(zip(MODEL_NAMES,g.tolist())),
                         "competition_weights":dict(zip(MODEL_NAMES,l.tolist()))})
    return {"version":"competition-adaptive-goal-family-audit-v1",
            "evaluated_at_utc":clock.isoformat(),"verified_unique_fixtures":len(prepared),
            "invalid_or_duplicate_excluded":invalid,"maximum_prior_at_original_capture":max_prior,
            "minimum_global_prior":MIN_GLOBAL,"minimum_competition_prior":MIN_LOCAL,
            "chronological_holdout_count":len(receipts),
            "competition_specific_holdout_count":len(local_metrics["baseline"]),
            "competition_specific_counts":dict(competitions),
            "same_fixture_metrics":{k:summarize_scores(v) for k,v in metrics.items()},
            "local_eligible_same_fixture_metrics":{k:summarize_scores(v) for k,v in local_metrics.items()},
            "receipts":receipts,"retrospective_not_prospective":True,
            "no_live_change":True,"not_a_five_double_trial":True,
            "automatic_promotion":False,"target_70_percent_proven":False}


def run():
    now=datetime.now(timezone.utc)
    rows,quality=finished_samples(now=now)
    report=audit(rows,now=now)
    report["source_quality"]=quality
    return report


if __name__=="__main__":
    import argparse
    from pathlib import Path
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",default="/tmp/competition_adaptive_audit.json")
    args=parser.parse_args()
    report=run()
    dest=Path(args.output)
    dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    print(json.dumps({k:v for k,v in report.items() if k!="receipts"},indent=2))
