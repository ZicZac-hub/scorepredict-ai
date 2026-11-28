"""Offline empirical-Bayes goal-rate calibration; not a full team-strength hierarchy.

Gamma-Poisson rate multipliers with cross-league shrinkage, separately for
home and away. Fixed priors; no tuning on evaluated results. Posterior
predictive grids assume conditional independence and are compared with
both original matched grids. No archive writes, API calls or promotion.
"""
from collections import Counter
from datetime import datetime,timezone
import json,math
import numpy as np
from v10_prospective import utc
from five_double_accuracy import _final,_result_index
from v44_family_pool_research import _grid
from v44_single_calibration import _metrics
import v44_market_ablation as archive
import v44_fixture_evidence as inventory
import v42_research_trial as sources

MIN_TRAIN=30
PRIOR_EXPOSURE=20.
SUPPORT=40
ARMS=("v8_control","v9","global_gamma","league_gamma")

def means(grid):
    p=_grid(grid)
    return (float(np.dot(np.arange(p.shape[0]),p.sum(axis=1))),
            float(np.dot(np.arange(p.shape[1]),p.sum(axis=0))))

def posterior(rows,side,prior_mean=1.):
    return (PRIOR_EXPOSURE*prior_mean+math.fsum(r["actual"][side] for r in rows),
            PRIOR_EXPOSURE+math.fsum(r["means"][side] for r in rows))

def predictive(exposure,shape,rate):
    if not all(math.isfinite(v) and v>0 for v in (exposure,shape,rate)):
        raise ValueError("invalid_gamma_poisson_parameter")
    p=np.empty(SUPPORT)
    p[0]=math.exp(-shape*math.log1p(exposure/rate))
    q=exposure/(rate+exposure)
    for k in range(1,SUPPORT):
        p[k]=p[k-1]*(shape+k-1)/k*q
    mass=float(p.sum())
    if not math.isfinite(mass) or abs(1.-mass)>1e-8:
        raise ValueError("insufficient_predictive_support")
    return p/mass

def fit(history,record):
    eligible=[r for r in history if r["known"]<record["decision"]
              and r["identity"]!=record["identity"] and r["event"]!=record["event"]]
    if len(eligible)<MIN_TRAIN:
        return None,len(eligible),0
    same=[r for r in eligible if r["league"]==record["league"]]
    other=[r for r in eligible if r["league"]!=record["league"]]
    global_parts=[];league_parts=[];scales=[]
    for side,e in enumerate(record["means"]):
        a,b=posterior(eligible,side)
        x,y=posterior(other,side)
        c,d=posterior(same,side,x/y)
        global_parts.append(predictive(e,a,b))
        league_parts.append(predictive(e,c,d))
        scales.append(c/d)
    return (np.outer(*global_parts),np.outer(*league_parts),scales),len(eligible),len(same)

def walk(records):
    rows=[];excluded=Counter();seen=set();events=set()
    ordered=sorted(records,key=lambda r:(r["decision"],r["identity"]))
    for r in ordered:
        if r["identity"] in seen or r["event"] in events or not r["decision"]<r["kickoff"]<r["known"]:
            raise ValueError("invalid_or_duplicate_record")
        seen.add(r["identity"]);events.add(r["event"])
    for r in ordered:
        try:
            fitted,n,local=fit(ordered,r)
            if fitted is None:
                excluded["insufficient_prior_fixtures"]+=1
                continue
            global_grid,league_grid,scale=fitted
            grids=dict(zip(ARMS,(r["a"],r["b"],global_grid,league_grid)))
            rows.append({"identity":list(r["identity"]),"decision_utc":r["decision"].isoformat(),
                         "training":n,"league_training":local,"league":r["league"],"league_scales":scale,
                         "metrics":{name:_metrics(p,r["actual"]) for name,p in grids.items()}})
        except ValueError as exc:
            excluded[str(exc)]+=1
    comparisons={}
    for arm in ARMS:
        items=[r["metrics"][arm] for r in rows];n=len(items)
        comparisons[arm]={"fixtures":n,
            **{k:math.fsum(m[k] for m in items)/n if n else None
               for k in ("exact_log_loss","score_brier","outcome_brier","top_score_hit")}}
    return {"verified_inputs":len(ordered),"matched_comparison_fixtures":len(rows),
            "exclusions":dict(excluded),"comparisons":comparisons,"rows":rows}

def evaluate(*,source_root=sources.SOURCE_ROOT,prematch_root=sources.PREMATCH_ROOT,now=None):
    clock=utc(now or datetime.now(timezone.utc))
    matched,counts,rejects=archive.match_archive(source_root,clock)
    finals,conflicts,result_rejections=_result_index(source_root,prematch_root)
    records=[]
    for control,market in matched:
        identity=inventory._identity(control)
        final=finals.get(identity)
        actual=_final(control,final,clock)
        if actual is None or _final(market,final,clock)!=actual:
            continue
        known=utc(final["verified_at_utc"]);decision=utc(control["captured_at_utc"])
        kickoff=utc(control["kickoff_utc"])
        league=str(control.get("competition") or "").strip()
        if not decision<kickoff<known<=clock or not league:
            continue
        a,b=_grid(control["score_matrix"]),_grid(market["score_matrix"])
        records.append({"identity":identity,"event":inventory._event(control),"league":league,
                        "decision":decision,"kickoff":kickoff,"known":known,
                        "actual":actual,"a":a,"b":b,"means":means(b)})
    return {"study_type":"empirical_bayes_gamma_poisson_calibration_v1",
            **walk(records),"source_rejections":dict(rejects),"result_rejections":dict(result_rejections),
            "result_conflicts":len(conflicts),"external_requests":0,"automatic_promotion":False,
            "limitations":["Empirical-Bayes rate calibration, not a full hierarchical team model.",
                           "Independent home and away predictive marginals may lose existing score dependence.",
                           "Retrospective chronological experiment; no prospective predictions or five-double hit rates.",
                           "Prior exposure fixed at 20; no tuning against holdout results."]}

if __name__=="__main__":
    print(json.dumps(evaluate(),indent=2,allow_nan=False))
