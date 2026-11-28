"""Chronological conditional-margin calibration, exploratory/read-only.

Given the observed total in each earlier verified match, fit one penalized
log-odds offset for winning margins >=3. At forecast time the actual total
is unknown: adjust each possible total separately, preserving its original
mass. Compare the existing fixed V38 hypothesis and an adaptive offset.
No provider calls, new pair selection, original archive changes or promotion.
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
from v38_margin_calibration import transform_score_grid

MIN_INFORMATIVE=20
PENALTY=8.
MAX_OFFSET=math.log(2.)
ARMS=("v8_control","v8_fixed_margin","v8_learned_margin","v9","v9_fixed_margin","v9_learned_margin")

def conditional_probability(grid,total):
    p=_grid(grid)
    i,j=np.indices(p.shape)
    band=i+j==total
    mass=float(p[band].sum())
    if mass<=0:
        return None
    return float(p[band & (abs(i-j)>=3)].sum()/mass)

def adjust(grid,offset):
    if not math.isfinite(offset) or abs(offset)>MAX_OFFSET+1e-12:
        raise ValueError("invalid_margin_offset")
    p=_grid(grid)
    i,j=np.indices(p.shape)
    weights=p*np.where(abs(i-j)>=3,math.exp(offset),1.)
    q=np.zeros_like(p)
    for total in range(sum(p.shape)-1):
        band=i+j==total
        mass=float(p[band].sum())
        if mass:
            q[band]=weights[band]*(mass/float(weights[band].sum()))
    return q

def fit(history,target,key):
    # Conditioning on earlier *observed* totals estimates the margin
    # distribution within each total; never condition on the target result.
    eligible=[]
    for r in history:
        if r["known"]>=target["decision"] or r["identity"]==target["identity"] or r["event"]==target["event"]:
            continue
        h,a=r["actual"]
        prob=conditional_probability(r[key],h+a)
        if prob is not None and 0<prob<1:
            eligible.append((prob,int(abs(h-a)>=3)))
    n=len(eligible)
    if n<MIN_INFORMATIVE:
        return None,n
    p,y=np.array(eligible).T
    def derivative(offset):
        e=math.exp(offset)
        return float(np.sum(y-p*e/(1-p+p*e))-PENALTY*offset)
    lo,hi=-MAX_OFFSET,MAX_OFFSET
    for _ in range(48):
        mid=(lo+hi)/2
        if derivative(mid)>0:
            lo=mid
        else:
            hi=mid
    return (lo+hi)/2,n

def metric(grid,actual):
    p=_grid(grid);i,j=np.indices(p.shape)
    margin=float(p[abs(i-j)>=3].sum())
    high=float(p[i+j>=4].sum())
    return {**_metrics(p,actual),"margin_brier":(margin-int(abs(actual[0]-actual[1])>=3))**2,
            "high_total_brier":(high-int(sum(actual)>=4))**2}

def walk(records):
    ordered=sorted(records,key=lambda r:(r["decision"],r["identity"]))
    ids=set();events=set();rows=[];exclusions=Counter()
    for r in ordered:
        if r["identity"] in ids or r["event"] in events or not r["decision"]<r["kickoff"]<r["known"]:
            raise ValueError("duplicate_or_invalid_fixture")
        ids.add(r["identity"]);events.add(r["event"])
    for r in ordered:
        offsets=[fit(ordered,r,key) for key in ("a","b")]
        if any(x[0] is None for x in offsets):
            exclusions["insufficient_informative_prior_fixtures"]+=1
            continue
        grids=[]
        for key,(offset,n) in zip(("a","b"),offsets):
            p=r[key]
            grids.extend([p,transform_score_grid(p,"conditional_margin_125"),adjust(p,offset)])
        rows.append({"identity":list(r["identity"]),"offsets":[v[0] for v in offsets],
                     "prior_informative":[v[1] for v in offsets],
                     "metrics":{k:metric(g,r["actual"]) for k,g in zip(ARMS,grids)}})
    summary={}
    for arm in ARMS:
        m=[r["metrics"][arm] for r in rows];n=len(m)
        summary[arm]={"fixtures":n,**{k:math.fsum(v[k] for v in m)/n if n else None
                     for k in ("exact_log_loss","score_brier","outcome_brier","top_score_hit","margin_brier","high_total_brier")}}
    return {"verified_inputs":len(ordered),"matched_comparison_fixtures":len(rows),
            "exclusions":dict(exclusions),"comparisons":summary,"rows":rows}

def evaluate(*,source_root=sources.SOURCE_ROOT,prematch_root=sources.PREMATCH_ROOT,now=None):
    clock=utc(now or datetime.now(timezone.utc))
    matches,_,rejects=archive.match_archive(source_root,clock)
    finals,conflicts,result_rejections=_result_index(source_root,prematch_root)
    records=[]
    for a,b in matches:
        identity=inventory._identity(a);final=finals.get(identity)
        actual=_final(a,final,clock)
        if actual is None or _final(b,final,clock)!=actual:
            continue
        decision,kickoff,known=map(utc,(a["captured_at_utc"],a["kickoff_utc"],final["verified_at_utc"]))
        if not decision<kickoff<known<=clock:
            continue
        records.append({"identity":identity,"event":inventory._event(a),
                        "decision":decision,"kickoff":kickoff,"known":known,"actual":actual,
                        "a":_grid(a["score_matrix"]),"b":_grid(b["score_matrix"])})
    return {"study_type":"chronological_conditional_margin_calibration_v1",**walk(records),
            "source_rejections":dict(rejects),"result_conflicts":len(conflicts),
            "result_rejections":dict(result_rejections),"external_requests":0,"automatic_promotion":False,
            "limitations":["Exploratory replay motivated by earlier archive diagnostics, not an independent holdout.",
                           "Total goal mass is unchanged; outcome and exact-score probability may change.",
                           "Offsets are bounded and fitted from earlier known finals only.",
                           "Single-match metrics are not prospective five-double hit rates."]}

if __name__=="__main__":
    print(json.dumps(evaluate(),indent=2,allow_nan=False))
