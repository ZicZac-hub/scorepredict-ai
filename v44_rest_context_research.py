"""Chronological rest-context calibration for original football score grids.

Research hypothesis inspired by fixture-congestion literature; archived
same-competition kickoff intervals are NOT complete rest or player fatigue.
Fit a signed, shrunk log-linear score adjustment on earlier verified finals.
No invented injuries, external requests, new pairs or live promotion.
"""
from collections import Counter
from datetime import datetime,timedelta,timezone
import json,math
from pathlib import Path
import numpy as np

from forecast_ledger import _forecast,LedgerValidationError
from five_double_accuracy import _final,_result_index,wilson
from professional_double_shadow import _validated
from v10_prospective import utc
import v42_research_trial as sources
import v44_fixture_evidence as inventory
from v44_family_pool_research import _grid
from v44_single_calibration import _metrics,_prob
from v44_pair_transfer_research import _arm

COEFFICIENTS=(-.12,-.06,0.,.06,.12)
MIN_TRAIN=60
MIN_NONZERO=15
PRIOR=12.
# Short intervals measured from kickoffs, not final whistle or training load.
SHORT_INTERVAL_DAYS=4.


def feature(model):
    context=model.get("research_context")
    if not isinstance(context,dict):
        raise ValueError("missing_schedule_context")
    cutoff=utc(model["evidence_as_of_utc"])
    capture=utc(model["captured_at_utc"])
    kickoff=utc(model["kickoff_utc"])
    if utc(context["as_of_utc"])!=cutoff or not cutoff<=capture<kickoff:
        raise ValueError("invalid_schedule_context_time")
    days=[]
    for side in ("home","away"):
        row=context[side]
        observed=utc(row["last_match_utc"])
        if observed+timedelta(hours=3)>cutoff or type(row["finished_matches"]) is not int or row["finished_matches"]<5:
            raise ValueError("insufficient_qualified_schedule_context")
        interval=(kickoff-observed).total_seconds()/86400
        saved=row["rest_days_since_last_kickoff"]
        if type(saved) not in (float,int) or not math.isfinite(saved) or abs(saved-interval)>.011:
            raise ValueError("rest_interval_mismatch")
        days.append(interval)
    # Difference in short-interval burden: positive when away has shorter interval.
    # Missing other-competition schedules are NOT assumed to be absence of matches.
    burden=[max(0.,min(1.,(SHORT_INTERVAL_DAYS-d)/SHORT_INTERVAL_DAYS)) for d in days]
    return burden[1]-burden[0]


def adjust(raw,x,beta):
    p=_grid(raw)
    if not math.isfinite(x) or abs(x)>1 or beta not in COEFFICIENTS:
        raise ValueError("invalid_rest_adjustment")
    h,a=np.indices(p.shape)
    # Bounded margin prevents high-score tails dominating the adjustment.
    multiplier=np.exp(beta*x*np.clip(h-a,-4,4))
    q=p*multiplier
    return q/q.sum()


def fit(prior):
    # Shrink towards no effect. Do not tune on the target or future outcomes.
    if len(prior)<MIN_TRAIN or sum(abs(r["feature"])>1e-12 for r in prior)<MIN_NONZERO:
        return None
    return min(COEFFICIENTS,key=lambda beta:(math.fsum(r["loss"][beta] for r in prior)
                       +PRIOR*(beta/.12)**2,abs(beta),beta))


def training_row(model,actual,known):
    x=feature(model)
    p=_grid(model["score_matrix"])
    return {"id":inventory._identity(model),"event":inventory._event(model),
            "known":utc(known),"decision":utc(model["captured_at_utc"]),
            "feature":x,"actual":actual,"grid":p,
            "loss":{b:-math.log(max(1e-15,_prob(adjust(p,x,b),actual))) for b in COEFFICIENTS}}


def earlier(history,cutoff,models):
    ids={inventory._identity(m) for m in models}
    events={inventory._event(m) for m in models}
    return [r for r in history if r["known"]<cutoff and r["id"] not in ids and r["event"] not in events]


def _summary(rows,pair=False):
    out={}
    for name in ("baseline","rest_calibrated"):
        m=[r[name] for r in rows]
        n=len(m);hits=sum(v["hit" if pair else "top_score_hit"] for v in m)
        out[name]={"observations":n,"hits":hits,"hit_rate":hits/n if n else None,
                   "wilson_95":wilson(hits,n),
                   "mean_log_loss":math.fsum(v["score_log_loss" if pair else "exact_log_loss"] for v in m)/n if n else None}
    return out


def evaluate(*,source_root=sources.SOURCE_ROOT,prematch_root=sources.PREMATCH_ROOT,
             provider_root=sources.PROVIDER_ROOT,pair_root=sources.PAIR_ROOT,now=None):
    clock=utc(now or datetime.now(timezone.utc))
    inv=inventory.evaluate(source_root=source_root,prematch_root=prematch_root,provider_root=provider_root,now=clock)
    history=[];rejects=Counter();counts=Counter()
    for row in inv["rows"]:
        try:
            path=Path(source_root)/"forecasts"/row["provider"]/row["fixture_id"]/(row["forecast_id"]+".json")
            m=_forecast(json.loads(path.read_text()))
            if (inventory._identity(m)!=(row["provider"],row["fixture_id"])
                    or m["forecast_id"]!=row["forecast_id"]
                    or list(inventory._event(m))!=row["event"]
                    or utc(m["captured_at_utc"])!=utc(row["decision_utc"])
                    or not utc(m["kickoff_utc"])<utc(row["final_known_utc"])<=clock):
                raise ValueError("inventory_source_mismatch")
            history.append(training_row(m,row["actual"],row["final_known_utc"]))
        except (KeyError,ValueError,TypeError,OSError,AttributeError,ArithmeticError,LedgerValidationError) as exc:
            rejects[str(exc) if isinstance(exc,ValueError) else type(exc).__name__]+=1
    singles=[]
    for r in sorted(history,key=lambda r:(r["decision"],r["id"])):
        prior=[p for p in history if p["known"]<r["decision"] and p["id"]!=r["id"] and p["event"]!=r["event"]]
        beta=fit(prior)
        if beta is None:
            continue
        singles.append({"id":list(r["id"]),"coefficient":beta,"training":len(prior),
                        "baseline":_metrics(r["grid"],r["actual"]),
                        "rest_calibrated":_metrics(adjust(r["grid"],r["feature"],beta),r["actual"])})
    finals,conflicts,result_rejects=_result_index(source_root,prematch_root)
    pairs=[];seen=set();events=set()
    for path in sorted((Path(pair_root)/"sets").glob("*.json")):
        try:
            record=json.loads(path.read_text())
            models,choices,coverage=_validated(record,source_root=source_root,now=clock)
            if path.stem!=record["sydney_date"]:
                raise ValueError("filename_mismatch")
            ids={inventory._identity(m) for m in models}
            physical={inventory._event(m) for m in models}
            if len(ids)!=2 or len(physical)!=2 or seen&ids or events&physical:
                raise ValueError("repeated_fixture")
            seen.update(ids);events.update(physical)
            decision=utc(record["predicted_at_utc"])
            actual=[_final(m,finals.get(inventory._identity(m)),clock) for m in models]
            if any(s is None for s in actual):
                counts["pending_pairs"]+=1
                continue
            x=[feature(m) for m in models]
            prior=earlier(history,decision,models)
            beta=fit(prior)
            if beta is None:
                counts["insufficient_training_pairs"]+=1
                continue
            grids=[_grid(m["score_matrix"]) for m in models]
            baseline={"full_score_grids":grids,"five_doubles":choices,"estimated_coverage":coverage}
            challenger=_arm([adjust(g,v,beta) for g,v in zip(grids,x)])
            pairs.append({"day":path.stem,"coefficient":beta,"training":len(prior),
                          "baseline":sources._metric(baseline,actual),
                          "rest_calibrated":sources._metric(challenger,actual)})
        except (KeyError,ValueError,TypeError,OSError,AttributeError,ArithmeticError) as exc:
            rejects["pair:"+ (str(exc) if isinstance(exc,ValueError) else type(exc).__name__)]+=1
    return {"study_type":"chronological_same_competition_rest_context_replay",
            "qualified_training_fixtures":len(history),
            "nonzero_feature_fixtures":sum(abs(r["feature"])>1e-12 for r in history),
            "counts":dict(counts),"rejections":dict(rejects),
            "single_comparisons":_summary(singles),"pair_comparisons":_summary(pairs,pair=True),
            "result_conflicts":len(conflicts),"result_rejections":dict(result_rejects),
            "external_requests":0,"automatic_promotion":False,
            "limitations":["Same-competition kickoff gaps are incomplete schedule proxies, not player fatigue.",
                           "Fixed coefficient candidates and prior are experimental design choices.",
                           "Retrospective evaluation does not prove prospective accuracy."],
            "rows":{"singles":singles,"pairs":pairs}}

if __name__=="__main__":
    print(json.dumps(evaluate(),indent=2,allow_nan=False))
