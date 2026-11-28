"""Exploratory dependence-preserving goal-rate correction on archived V9.

Apply a separable exponential tilt to the existing joint score grid. This
preserves structural zeros and every defined cross-product odds ratio,
unlike replacing the grid with independent goal marginals. Fixed full/half
correction arms, strictly chronological training; no live forecasts.
"""
from collections import Counter
from datetime import datetime,timezone
import json,math
import numpy as np
import v44_bayesian_rate_calibration as bayes
from v44_family_pool_research import _grid
from v44_single_calibration import _metrics
from v10_prospective import utc
from five_double_accuracy import _final,_result_index
import v44_market_ablation as archive
import v44_fixture_evidence as inventory
import v42_research_trial as sources

ARMS=("v8_control","v9","independent_league_gamma","joint_full","joint_half")

def tilt(grid,scales,strength):
    if strength not in (0.,.5,1.) or len(scales)!=2 or not all(
        math.isfinite(x) and x>0 for x in scales):
        raise ValueError("invalid_tilt")
    p=_grid(grid)
    if strength==0.:
        return p.copy()
    h,a=np.indices(p.shape)
    log_weight=strength*(h*math.log(scales[0])+a*math.log(scales[1]))
    positive=p>0
    log_p=np.full(p.shape,-np.inf)
    log_p[positive]=np.log(p[positive])+log_weight[positive]
    values=np.exp(log_p-np.max(log_p))
    z=float(values.sum())
    if not math.isfinite(z) or z<=0:
        raise ValueError("invalid_tilt_normalizer")
    q=values/z
    if np.any(q[positive]==0):
        raise ValueError("tilt_underflow")
    return q

def walk(records):
    rows=[];excluded=Counter();seen=set();events=set()
    ordered=sorted(records,key=lambda r:(r["decision"],r["identity"]))
    for r in ordered:
        if r["identity"] in seen or r["event"] in events or not r["decision"]<r["kickoff"]<r["known"]:
            raise ValueError("invalid_or_duplicate_record")
        seen.add(r["identity"]);events.add(r["event"])
    for r in ordered:
        try:
            fitted,n,local=bayes.fit(ordered,r)
            if fitted is None:
                excluded["insufficient_prior_fixtures"]+=1
                continue
            _,league,scales=fitted
            arms=dict(zip(ARMS,(r["a"],r["b"],league,
                                 tilt(r["b"],scales,1.),tilt(r["b"],scales,.5))))
            rows.append({"identity":list(r["identity"]),"decision_utc":r["decision"].isoformat(),
                         "training":n,"league_training":local,"scales":scales,
                         "metrics":{name:_metrics(p,r["actual"]) for name,p in arms.items()}})
        except ValueError as exc:
            excluded[str(exc)]+=1
    summary={}
    for arm in ARMS:
        items=[r["metrics"][arm] for r in rows];n=len(items)
        summary[arm]={"fixtures":n,
            **{k:math.fsum(m[k] for m in items)/n if n else None
               for k in ("exact_log_loss","score_brier","outcome_brier","top_score_hit")}}
    return {"verified_inputs":len(ordered),"matched_comparison_fixtures":len(rows),
            "exclusions":dict(excluded),"comparisons":summary,"rows":rows}

def evaluate(*,source_root=sources.SOURCE_ROOT,prematch_root=sources.PREMATCH_ROOT,now=None):
    clock=utc(now or datetime.now(timezone.utc))
    matched,_,rejects=archive.match_archive(source_root,clock)
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
                        "actual":actual,"a":a,"b":b,"means":bayes.means(b)})
    return {"study_type":"exploratory_joint_grid_rate_tilt_v1",**walk(records),
            "source_rejections":dict(rejects),"result_rejections":dict(result_rejections),
            "result_conflicts":len(conflicts),"external_requests":0,"automatic_promotion":False,
            "limitations":["Retrospective experiment motivated by previous results on the same archive; not independent confirmation.",
                           "Preserves cross-product odds ratios, not Pearson correlation or exact target goal means.",
                           "Half and full strengths fixed before this replay; do not select a winner for production from this sample.",
                           "No prospective tickets or five-double accuracy evidence."]}

if __name__=="__main__":
    print(json.dumps(evaluate(),indent=2,allow_nan=False))
