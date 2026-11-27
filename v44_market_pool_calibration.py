"""Chronological calibration experiment on archived same-time model grids.

Research only: convex probability pooling, not new selections or deployment.
Inspired by Egidi et al., Statistical Modelling 18 (2018), 436–459,
and Ranjan & Gneiting, JRSS B 72 (2010), 71–91.
"""
from collections import Counter
from datetime import datetime,timezone
import json,math
from pathlib import Path
import numpy as np

from v10_prospective import utc
from five_double_accuracy import _final,_result_index,wilson
import v44_market_ablation as audit
import v44_fixture_evidence as inventory
from v44_family_pool_research import _grid
from v44_single_calibration import _metrics,_prob
import v42_research_trial as sources

MIN_TRAIN=30
WEIGHTS=(0.,.25,.5,.75,1.)
PRIOR=4.


def pool(first,second,weight):
    if weight not in WEIGHTS:
        raise ValueError("unregistered_weight")
    a,b=_grid(first),_grid(second)
    shape=tuple(max(x,y) for x,y in zip(a.shape,b.shape))
    def pad(g):
        return np.pad(g,[(0,shape[i]-g.shape[i]) for i in range(2)])
    return (1-weight)*pad(a)+weight*pad(b)


def fit(history,decision,identity,event):
    eligible=[r for r in history if r["known"]<decision
              and r["identity"]!=identity and r["event"]!=event]
    if len(eligible)<MIN_TRAIN:
        return None,len(eligible)
    # Regularize towards equal weight and retain a fixed candidate grid.
    def loss(w):
        return math.fsum(-math.log(max(1e-15,(1-w)*r["p0"]+w*r["p1"]))
                         for r in eligible)+PRIOR*(w-.5)**2
    return min(WEIGHTS,key=lambda w:(loss(w),abs(w-.5),w)),len(eligible)


def evaluate(*,source_root=sources.SOURCE_ROOT,prematch_root=sources.PREMATCH_ROOT,
             pair_root=sources.PAIR_ROOT,now=None):
    clock=utc(now or datetime.now(timezone.utc))
    matches,counts,rejections=audit.match_archive(source_root,clock)
    finals,conflicts,result_rejects=_result_index(source_root,prematch_root)
    records=[];metrics=[];weights=Counter()
    for control,market in matches:
        identity=inventory._identity(control)
        final=finals.get(identity)
        actual=_final(control,final,clock)
        if actual is None or _final(market,final,clock)!=actual:
            continue
        known=utc(final["verified_at_utc"])
        decision=utc(control["captured_at_utc"])
        if not decision<utc(control["kickoff_utc"])<known<=clock:
            continue
        a,b=_grid(control["score_matrix"]),_grid(market["score_matrix"])
        records.append({"identity":identity,"event":inventory._event(control),
                        "decision":decision,"known":known,"actual":actual,
                        "a":a,"b":b,"p0":_prob(a,actual),"p1":_prob(b,actual)})
    for r in sorted(records,key=lambda r:(r["decision"],r["identity"])):
        w,n=fit(records,r["decision"],r["identity"],r["event"])
        if w is None:
            continue
        arms={"control":r["a"],"market":r["b"],
              "equal":pool(r["a"],r["b"],.5),"adaptive":pool(r["a"],r["b"],w)}
        weights[str(w)]+=1
        metrics.append({"fixture_id":r["identity"][1],"training":n,"weight":w,
                        "metrics":{k:_metrics(g,r["actual"]) for k,g in arms.items()}})
    summary={}
    for name in ("control","market","equal","adaptive"):
        m=[r["metrics"][name] for r in metrics]
        n=len(m);hits=sum(v["top_score_hit"] for v in m)
        summary[name]={"fixtures":n,"hits":hits,"hit_rate":hits/n if n else None,
                       "wilson_95":wilson(hits,n),
                       **{k:math.fsum(v[k] for v in m)/n if n else None
                          for k in ("exact_log_loss","score_brier","outcome_brier")}}
    # Inventory only: do not create synthetic pairs or issue new predictions.
    lookup={inventory._identity(a):(a,b) for a,b in matches}
    coverage=Counter()
    for path in sorted((Path(pair_root)/"sets").glob("*.json")):
        try:
            pair,models=sources._pair(path,source_root=source_root,decision=clock)
            cutoff=utc(pair["predicted_at_utc"])
            available=0
            for model in models:
                item=lookup.get(inventory._identity(model))
                if item is None:
                    continue
                a,b=item
                if (inventory._event(a)==inventory._event(model)
                        and str(a["home_team_id"])==str(model["home_team_id"])
                        and str(a["away_team_id"])==str(model["away_team_id"])
                        and utc(a["captured_at_utc"])<=cutoff
                        and utc(a["kickoff_utc"])==utc(model["kickoff_utc"])):
                    available+=1
            coverage["pairs_with_"+str(available)+"_matched_legs"]+=1
        except (ValueError,KeyError,TypeError,OSError,AttributeError,ArithmeticError):
            coverage["invalid_pairs"]+=1
    return {"study_type":"chronological_archived_model_pool_calibration",
            "verified_training_fixtures":len(records),"minimum_prior_fixtures":MIN_TRAIN,
            "comparisons":summary,"chosen_weights":dict(weights),
            "original_pair_evidence_inventory":dict(coverage),
            "source_rejections":dict(rejections),"conflicting_results":len(conflicts),
            "result_rejections":dict(result_rejects),"external_requests":0,
            "automatic_promotion":False,"rows":metrics,
            "limitations":["Archived models share underlying inputs and are not independent.",
                           "Calibration is an exploratory retrospective replay.",
                           "Original pair inventory is not a five-double accuracy result."]}

if __name__=="__main__":
    print(json.dumps(evaluate(),indent=2,allow_nan=False))
