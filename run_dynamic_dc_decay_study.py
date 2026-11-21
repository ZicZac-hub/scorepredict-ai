"""Fixed half-life sensitivity with chronological validation, research only.

Select by proper exact-score log loss on 2023-24, then report 2024-25
without changing that choice. Both seasons are existing development data,
not an untouched holdout. No production model or archived forecasts change.
"""
import json
import math
from datetime import timedelta
import numpy as np
from engine import _date, _grid
from dynamic_dixon_coles import (_observations, _fit, _calibrate, _rates,
    _safe_rho, _groups, fit_forecast, five_doubles, HALF_LIFE, MIN_HISTORY)
from backtest_dynamic_dixon_coles import evaluate, SYDNEY
from run_dynamic_dc_pinned_replay import load_matches

CANDIDATES=(90,180,365)
BOUNDARY="2024-08-01"

def fit_decay(fixture, history, half_life):
    if half_life not in CANDIDATES:
        raise ValueError("Unknown predeclared half-life")
    if half_life==HALF_LIFE:
        return fit_forecast(fixture,history)
    rows=[(*r[:-1],r[-1]**(HALF_LIFE/half_life))
          for r in _observations(fixture,history)]
    if len(rows)<MIN_HISTORY or sum(r[-1] for r in rows)<30:
        return None
    h,a=fixture["homeTeam"]["id"],fixture["awayTeam"]["id"]
    if min(sum(t in r[1:3] for r in rows) for t in (h,a))<5:
        return None
    if min(sum(r[-1] for r in rows if t in r[1:3]) for t in (h,a))<3.0:
        return None
    split=rows[int(.75*len(rows))][0]
    early=_fit([r for r in rows if r[0]<split])
    full=_fit(rows)
    if early is None or full is None:
        return None
    rho,factors,n=_calibrate([r for r in rows if r[0]>=split],early)
    rates=_rates(full,h,a)
    if rates is None:
        return None
    matrix=_grid(*rates,10,rho=_safe_rho(*rates,rho))
    matrix*=factors[_groups(matrix)]
    matrix/=matrix.sum()
    return {"matrix":matrix,"model_version":"decay-research-only"}

def score_pair(matrices,actual):
    picks=five_doubles(*matrices)
    hit=int(any((s,t)==actual for _,s,t in picks))
    loss=0.
    for matrix,(h,g) in zip(matrices,actual):
        p=float(matrix[h,g]) if h<matrix.shape[0] and g<matrix.shape[1] else 0.
        loss-=math.log(max(p,1e-12))/2
    return hit,loss,sum(p for p,_,_ in picks)

def compare_phase(matches, reference, start, stop=None):
    index={m["id"]:m for m in matches}
    labels=["baseline"]+[str(x) for x in CANDIDATES]
    sums={k:{"hits":0,"score_log_loss":0.,"mean_coverage":0.} for k in labels}
    eligible=0
    completed=0
    for trial in reference["matched_trials"]:
        day=trial["sydney_date"]
        if day<start or (stop is not None and day>=stop):
            continue
        eligible+=1
        pair=[index[i] for i in trial["fixture_ids"]]
        cutoff=min(_date(m["utcDate"]) for m in pair)-timedelta(days=1)
        past=[m for m in matches if _date(m["utcDate"])<cutoff
              and _date(m["utcDate"]).astimezone(SYDNEY).date().isoformat()<day]
        groups={h:[fit_decay(m,past,h) for m in pair] for h in CANDIDATES}
        if any(f is None for group in groups.values() for f in group):
            continue
        actual=tuple(tuple(s) for s in trial["actual_scores"])
        baseline=trial["models"]["baseline"]
        scores={"baseline":(baseline["hit"],baseline["actual_pair_mean_score_log_loss"],baseline["coverage"])}
        for half,group in groups.items():
            scores[str(half)]=score_pair([f["matrix"] for f in group],actual)
        completed+=1
        for k,(hit,loss,coverage) in scores.items():
            sums[k]["hits"]+=hit
            sums[k]["score_log_loss"]+=loss
            sums[k]["mean_coverage"]+=coverage
    for row in sums.values():
        row["pairs"]=completed
        row["hit_rate"]=row["hits"]/completed if completed else None
        for key in ("score_log_loss","mean_coverage"):
            row[key]=row[key]/completed if completed else None
    return {"eligible_reference_pairs":eligible,"matched_pairs":completed,
            "excluded_pairs":eligible-completed,"models":sums}

def select_half_life(validation):
    if validation["matched_pairs"]<30:
        return None
    rows=validation["models"]
    return min(CANDIDATES,key=lambda h:(rows[str(h)]["score_log_loss"],h))

def main():
    matches,missing=load_matches()
    matches=[m for m in matches if m["utcDate"][:10] not in missing]
    # The loader uses calendar dates only, with a conservative training buffer.
    reference=evaluate(matches,start="2023-08-01")
    validation=compare_phase(matches,reference,"2023-08-01",BOUNDARY)
    selected=select_half_life(validation)
    evaluation=compare_phase(matches,reference,BOUNDARY)
    print(json.dumps({"development_only":True,"production_promotion":False,
        "selection_rule":"lowest validation exact-score log loss; >=30 matched pairs",
        "selected_half_life_days":selected,"validation":validation,
        "evaluation":evaluation,"selected_evaluation":evaluation["models"].get(str(selected)),
        "source_date_surrogate":True},indent=2))

if __name__=="__main__":
    main()
