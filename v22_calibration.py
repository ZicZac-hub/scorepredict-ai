"""V22 chronological calibration of complete score distributions.

Only source-matched verified past forecasts can fit a temperature. No live
result, post-kickoff snapshot or future score may enter. A fitted shadow
temperature is NOT automatically promoted and needs prospective evaluation.
"""
import math
from collections import defaultdict
from datetime import datetime,timezone,timedelta

import numpy as np
from persist_prematch import verified_result
from v10_prospective import utc

CANDIDATES=(.80,.90,1.0,1.10,1.20,1.30)
MIN_PER_LEAGUE=100
MAX_RECENT=450


def scale(matrix, temperature):
    p=np.asarray(matrix,dtype=float)
    if p.ndim!=2 or min(p.shape)<5 or np.any(~np.isfinite(p)) or np.any(p<0) or abs(p.sum()-1)>1e-5:
        raise ValueError("Normalized score probability grid required")
    if not .6<=temperature<=1.6:
        raise ValueError("Unsafe temperature")
    p=np.maximum(p,1e-14)**(1.0/float(temperature))
    return p/p.sum()


def _training_rows(rows, verified, as_of):
    clock=utc(as_of)
    seen=set()
    filtered=[]
    for row in rows.values() if isinstance(rows,dict) else rows:
        try:
            if row.get("model_key") not in ("v8","v8_near"):
                continue
            key=(str(row["provider"]),str(row["fixture_id"]))
            if key in seen:
                continue
            kickoff=utc(row["kickoff_utc"])
            saved=utc(row["captured_at_utc"])
            final=verified.get(str(row["fixture_id"]))
            if not final or not verified_result(row,final,str(row["fixture_id"])):
                continue
            available=utc(final["verified_at_utc"])
            if not saved<kickoff or not kickoff+timedelta(hours=3)<=available<=clock:
                continue
            if row.get("score_basis")!="regulation" or final.get("status") not in ("FT","FINISHED","AET","PEN"):
                continue
            grid=np.asarray(row["score_matrix"],dtype=float)
            if grid.ndim!=2 or min(grid.shape)<5 or abs(grid.sum()-1)>1e-5:
                continue
            goals=(int(final["home_goals"]),int(final["away_goals"]))
            league=str(row.get("competition") or "unknown")
            filtered.append({"kickoff":kickoff,"available":available,"predicted":saved,
                             "model_id":str(row.get("forecast_id") or ""),
                             "fixture":key,"league":league,
                             "matrix":grid,"goals":goals})
            seen.add(key)
        except (KeyError,TypeError,ValueError,OverflowError):
            continue
    filtered.sort(key=lambda x:(x["available"],x["kickoff"],x["fixture"]))
    return filtered


def _score(p,score):
    h,a=score
    return -math.log(max(float(p[h,a]) if h<p.shape[0] and a<p.shape[1] else 0.0,1e-12))


def fitted_temperature(rows, verified, *,as_of,league,min_samples=MIN_PER_LEAGUE):
    """Only the same-league verified forecasts available before as_of."""
    sample=[x for x in _training_rows(rows,verified,as_of)
            if x["league"]==str(league)][-MAX_RECENT:]
    if len(sample)<min_samples:
        return {"ready":False,"reason":"verified_training_samples_insufficient",
                "sample_size":len(sample),"min_required":min_samples}
    loss={}
    for t in CANDIDATES:
        loss[t]=sum(_score(scale(x["matrix"],t),x["goals"]) for x in sample)/len(sample)
    choice=min(CANDIDATES,key=lambda x:(loss[x],abs(x-1)))
    return {"ready":True,"temperature":choice,
            "source":"strictly_previous_verified_original_forecasts",
            "league":league,"sample_size":len(sample),
            "train_mean_log_loss":loss[choice],
            "train_baseline_mean_log_loss":loss[1.0],
            "training_end_at_utc":sample[-1]["available"].isoformat(),
            "not_out_of_sample_evidence":True,"promoted_to_production":False}


def chronological_evaluation(rows, verified, *,min_samples=MIN_PER_LEAGUE, as_of=None):
    """Actual prequential evaluation of a temporal calibration algorithm.

    At each prediction, only ORIGINAL records whose final scores were
    verifiably known before that match's forecast could influence fitting.
    Multiple outputs from a single fixture never inflate sample size.
    """
    records=_training_rows(rows,verified,as_of or datetime.now(timezone.utc))
    by_league=defaultdict(list)
    for item in records:
        by_league[item["league"]].append(item)
    output={}
    for league,group in by_league.items():
        played=sorted(group,key=lambda x:(x["kickoff"],x["fixture"]))
        evaluated=[]
        for current in played:
            known=[x for x in played if x["available"]<current["predicted"]]
            if len(known)<min_samples:
                continue
            candidates={}
            for temp in CANDIDATES:
                candidates[temp]=sum(_score(scale(x["matrix"],temp),x["goals"])
                                      for x in known[-MAX_RECENT:])/min(MAX_RECENT,len(known))
            chosen=min(CANDIDATES,key=lambda x:(candidates[x],abs(x-1)))
            before=_score(current["matrix"],current["goals"])
            after=_score(scale(current["matrix"],chosen),current["goals"])
            evaluated.append((before,after,chosen))
        n=len(evaluated)
        output[league]={"verified_evaluated":n,
                        "original_log_loss":sum(x[0] for x in evaluated)/n if n else None,
                        "candidate_log_loss":sum(x[1] for x in evaluated)/n if n else None,
                        "temperatures":{str(t):sum(x[2]==t for x in evaluated)
                                        for t in CANDIDATES},
                        "validated_out_of_sample":n>0,
                        "production_promotion":False}
    return {"kind":"strictly_prequential_same_fixture_comparison",
            "leagues":output,"automatic_promotion":False}
