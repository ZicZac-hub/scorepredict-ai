"""Independent chronology-locked calibration for verified-xG shadow forecasts.

Input is an explicit collection of archived, pre-kickoff authenticated snapshots,
not a dataset fabricated from final results. A calendar cutoff is fixed before
inspection; all learned goal-band multipliers use only earlier cases.
"""
import argparse
from collections import defaultdict
import json
import math
from zoneinfo import ZoneInfo

import numpy as np

from dynamic_dixon_coles import _groups, fit_forecast, five_doubles
from engine import _date
from verified_xg_challenger import predict_shadow
from verified_xg_prediction_archive import validated_shadow

SYDNEY = ZoneInfo("Australia/Sydney")
MODELS = ("dynamic_baseline", "verified_xg", "verified_xg_calibrated")
EPS = 1e-12


def _case(row):
    if not isinstance(row,dict):
        return None
    fixture=row.get("fixture")
    history=row.get("history")
    snapshot=row.get("snapshot")
    actual=row.get("actual")
    if (not isinstance(fixture,dict) or not isinstance(history,list)
            or not isinstance(snapshot,dict) or not isinstance(actual,(list,tuple))
            or len(actual)!=2 or any(type(x) is not int or x<0 for x in actual)):
        return None
    when=_date(fixture.get("utcDate"))
    if when is None:
        return None
    if "frozen_shadow" in row:
        # Genuine archived cases are graded on matrices sealed at capture time.
        # Never silently recompute them with later changes to model code.
        pinned=validated_shadow(row["frozen_shadow"])
        if pinned is None:
            return None
        baseline_matrix,xg_matrix,lineup_status=pinned
    else:
        # Direct-input replay is retained for isolated synthetic research tests;
        # the production archive loader always supplies frozen_shadow.
        base=fit_forecast(fixture,history)
        xg=predict_shadow(fixture,history,snapshot)
        if base is None or xg is None:
            return None
        baseline_matrix,xg_matrix=base["matrix"],xg["matrix"]
        lineup_status=xg["lineup_status"]
    return {"fixture":fixture,"actual":tuple(actual),"when":when,
            "baseline":baseline_matrix,"xg":xg_matrix,
            "lineup_status":lineup_status}


def fit_bands(training, prior=20.):
    """Multinomial goal-band calibration, strictly training-only."""
    if len(training)<30:
        return [1.,1.,1.]
    obs=np.zeros(3)
    exp=np.zeros(3)
    for r in training:
        h,a=r["actual"]
        obs[0 if h+a<=1 else 2 if h+a>=4 else 1]+=1.
        x=r["xg"]
        exp+=np.bincount(_groups(x).ravel(),weights=x.ravel(),minlength=3)
    return np.clip((obs+prior)/(exp+prior),.6,1.7).tolist()


def _calibrated(matrix, factors):
    clone=np.array(matrix,copy=True)
    clone*=np.asarray(factors)[_groups(clone)]
    clone/=clone.sum()
    return clone


def _record(metric,matrix,actual):
    h,a=actual
    p=float(matrix[h,a]) if h<matrix.shape[0] and a<matrix.shape[1] else 0.
    top=np.unravel_index(matrix.argmax(),matrix.shape)
    metric["matches"]+=1
    metric["exact_top1_hits"]+=int(tuple(top)==actual)
    metric["score_log_loss"]-=math.log(max(p,EPS))
    i,j=np.indices(matrix.shape)
    outcome=np.array([matrix[i>j].sum(),matrix[i==j].sum(),matrix[i<j].sum()])
    actual_outcome=0 if h>a else 1 if h==a else 2
    metric["outcome_brier"]+=sum((outcome-np.eye(3)[actual_outcome])**2)


def _finish(metrics):
    result={}
    for key,v in metrics.items():
        n=v["matches"]
        result[key]={"evaluated":int(n),
                     "top1_hit_rate":v["exact_top1_hits"]/n if n else None,
                     "score_log_loss":v["score_log_loss"]/n if n else None,
                     "outcome_brier":v["outcome_brier"]/n if n else None}
    return result


def _paired_delta(metrics):
    """Identical admitted matches for both models; positive hit delta helps xG.

    Negative loss/Brier deltas help xG. None means the denominator is zero.
    These are descriptive differences, never a significance claim.
    """
    base, xg = metrics["dynamic_baseline"], metrics["verified_xg"]
    n = base["evaluated"]
    if n != xg["evaluated"]:
        raise ValueError("unpaired_model_denominators")
    def delta(key):
        left, right = base[key], xg[key]
        return right-left if n and left is not None and right is not None else None
    return {
        "matched_cases": n,
        "exact_score_top1_hit_rate_delta_xg_minus_dc": delta("top1_hit_rate"),
        "score_log_loss_delta_xg_minus_dc": delta("score_log_loss"),
        "outcome_brier_delta_xg_minus_dc": delta("outcome_brier"),
    }


def evaluate(cases,cutoff="2026-01-01"):
    prepared=[r for r in (_case(case) for case in cases) if r is not None]
    prepared.sort(key=lambda r:(r["when"],str(r["fixture"].get("id"))))
    # Do not use 2026 outcomes to fit any coefficient.
    before=[r for r in prepared if r["when"].date().isoformat()<cutoff]
    after=[r for r in prepared if r["when"].date().isoformat()>=cutoff]
    factors=fit_bands(before)
    stats={k:defaultdict(float) for k in MODELS}
    groups=defaultdict(list)
    strata=defaultdict(lambda:{k:defaultdict(float) for k in MODELS})
    for r in after:
        matrices={"dynamic_baseline":r["baseline"],"verified_xg":r["xg"],
                  "verified_xg_calibrated":_calibrated(r["xg"],factors)}
        for key,matrix in matrices.items():
            _record(stats[key],matrix,r["actual"])
            _record(strata[r["lineup_status"]][key],matrix,r["actual"])
        groups[r["when"].astimezone(SYDNEY).date()].append((r,matrices))
    doubles={k:{"pairs":0,"hits":0,"coverage":0.} for k in MODELS}
    for _, day_rows in sorted(groups.items()):
        # Deterministic fixed pairing; neither odds nor final scores select it.
        selected=None
        for i,(a,ma) in enumerate(day_rows):
            ta={a["fixture"]["homeTeam"]["id"],a["fixture"]["awayTeam"]["id"]}
            for b,mb in day_rows[i+1:]:
                tb={b["fixture"]["homeTeam"]["id"],b["fixture"]["awayTeam"]["id"]}
                if ta.isdisjoint(tb):
                    selected=((a,ma),(b,mb))
                    break
            if selected:
                break
        if selected is None:
            continue
        (a,ma),(b,mb)=selected
        for key in MODELS:
            picks=five_doubles(ma[key],mb[key])
            actual=(a["actual"],b["actual"])
            d=doubles[key]
            d["pairs"]+=1
            d["hits"]+=int(any((x,y)==actual for _,x,y in picks))
            d["coverage"]+=sum(x for x,_,_ in picks)
    for row in doubles.values():
        n=row["pairs"]
        row["hit_rate"]=row["hits"]/n if n else None
        row["mean_predicted_coverage"]=row["coverage"]/n if n else None
    finished = _finish(stats)
    lineup_metrics = {key:_finish(value) for key,value in sorted(strata.items())}
    base_pairs, xg_pairs = doubles["dynamic_baseline"], doubles["verified_xg"]
    if base_pairs["pairs"] != xg_pairs["pairs"]:
        raise ValueError("unpaired_five_double_denominators")
    n_pairs = base_pairs["pairs"]
    paired = {
        "all": _paired_delta(finished),
        "by_lineup_availability": {
            key:_paired_delta(value) for key,value in lineup_metrics.items()},
        "five_double": {
            "matched_pairs": n_pairs,
            "hit_rate_delta_xg_minus_dc": (
                xg_pairs["hit_rate"]-base_pairs["hit_rate"] if n_pairs else None),
            "mean_coverage_delta_xg_minus_dc": (
                xg_pairs["mean_predicted_coverage"]-
                base_pairs["mean_predicted_coverage"] if n_pairs else None),
        },
        "interpretation": (
            "Positive exact-score and five-double hit deltas favor xG; "
            "negative score-log-loss, outcome-Brier deltas favor xG. "
            "These are descriptive paired comparisons, not proof of improvement."),
    }
    return {
        "cutoff":cutoff,"received_cases":len(cases),
        "verified_cases":len(prepared),"training_cases":len(before),
        "holdout_cases":len(after),"factors_from_training_only":factors,
        "model_metrics":finished,"by_lineup_availability":lineup_metrics,
        "paired_differences":paired,
        "five_double":doubles,
        "caution":"Research-only. Requires independently archived pre-match snapshots; missing cases excluded. Historical replay is not prospective accuracy proof; calibration fitted only before cutoff. Missing lineups do not imply missing players."
    }


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("cases_json",help="JSON list of {fixture,history,snapshot,actual}")
    parser.add_argument("--cutoff",default="2026-01-01")
    parser.add_argument("--output",default="verified_xg_shadow_results.json")
    args=parser.parse_args()
    with open(args.cases_json,encoding="utf-8") as file:
        cases=json.load(file)
    report=evaluate(cases,args.cutoff)
    with open(args.output,"w",encoding="utf-8") as file:
        json.dump(report,file,indent=2,allow_nan=False)
    print(json.dumps({"training":report["training_cases"],
                      "holdout":report["holdout_cases"],
                      "model_metrics":report["model_metrics"],
                      "five_double":report["five_double"]},indent=2))
