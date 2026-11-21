"""Read-only calendar-date development comparison for goal distributions.

All methods use the same baseline goal-rate evidence, pair identities, and
conservative pre-match history cutoff. 2023-24 selects a model by proper
exact-score log loss; 2024-25 reports that frozen choice. These seasons
have already been explored; this is NOT untouched or prospective evidence.
"""
import json
import math
from datetime import timedelta

import numpy as np

from engine import _date, predict
from backtest_dynamic_dixon_coles import SYDNEY, evaluate, five_doubles
from run_dynamic_dc_pinned_replay import load_matches
from v21_goal_model_comparison import variants

METHODS=("baseline_v8","poisson_marginal","negative_binomial_r3",
         "bivariate_poisson_c012")
SPLIT="2024-08-01"
MIN_SELECTION_PAIRS=30


def grade_pair(matrices,actual):
    """Score identical two-match outcomes; never mistake probability for wins."""
    picks=five_doubles(*matrices)
    hit=int(any((s,t)==actual for _,s,t in picks))
    coverage=sum(p for p,_,_ in picks)
    loss=0.
    high_forecast=0.
    high_observed=0
    high_brier=0.
    for matrix,(h,g) in zip(matrices,actual):
        p=float(matrix[h,g]) if h<matrix.shape[0] and g<matrix.shape[1] else 0.
        loss-=math.log(max(p,1e-12))/2
        ix,iy=np.indices(matrix.shape)
        four_plus=float(matrix[ix+iy>=4].sum())
        observed=int(h+g>=4)
        high_forecast+=four_plus/2
        high_observed+=observed/2
        high_brier+=(four_plus-observed)**2/2
    return {"hit":hit,"coverage":coverage,"score_log_loss":loss,
            "four_plus_probability":high_forecast,
            "four_plus_observed":high_observed,
            "four_plus_brier":high_brier}


def select_model(validation):
    """Predeclare log-loss selection, retaining the unchanged baseline."""
    if validation["matched_pairs"]<MIN_SELECTION_PAIRS:
        return None
    return min(METHODS,key=lambda name:
               (validation["models"][name]["score_log_loss"],METHODS.index(name)))


def compare_phase(matches,reference,start,stop=None):
    index={str(m["id"]):m for m in matches}
    accum={name:{"hits":0,"sum_coverage":0.,"sum_loss":0.,
                 "sum_four_plus":0.,"sum_observed_four_plus":0.,
                 "sum_four_plus_brier":0.} for name in METHODS}
    eligible=0
    qualified=0
    excluded={"baseline_unavailable":0,"invalid_score_grid":0}
    for trial in reference["matched_trials"]:
        day=trial["sydney_date"]
        if day<start or stop is not None and day>=stop:
            continue
        eligible+=1
        pair=[index[str(key)] for key in trial["fixture_ids"]]
        cutoff=min(_date(m["utcDate"]) for m in pair)-timedelta(days=1)
        past=[m for m in matches if _date(m["utcDate"])<cutoff
              and _date(m["utcDate"]).astimezone(SYDNEY).date().isoformat()<day]
        # Remove the realized scores before passing fixtures to the model.
        fixtures=[{**m,"status":"SCHEDULED",
                   "score":{"fullTime":{"home":None,"away":None}}} for m in pair]
        original=[predict(f,past,variant="baseline") for f in fixtures]
        if any(f is None for f in original):
            excluded["baseline_unavailable"]+=1
            continue
        try:
            by_leg=[variants(f["matrix"]) for f in original]
        except (ValueError,TypeError,KeyError,FloatingPointError):
            excluded["invalid_score_grid"]+=1
            continue
        actual=tuple(tuple(score) for score in trial["actual_scores"])
        qualified+=1
        for name in METHODS:
            result=grade_pair([leg[name] for leg in by_leg],actual)
            dest=accum[name]
            dest["hits"]+=result["hit"]
            for src,dst in (("coverage","sum_coverage"),("score_log_loss","sum_loss"),
                            ("four_plus_probability","sum_four_plus"),
                            ("four_plus_observed","sum_observed_four_plus"),
                            ("four_plus_brier","sum_four_plus_brier")):
                dest[dst]+=result[src]
    models={}
    for name,row in accum.items():
        models[name]={"pairs":qualified,"hits":row["hits"],
                      "hit_rate":row["hits"]/qualified if qualified else None,
                      "mean_joint_five_coverage":row["sum_coverage"]/qualified if qualified else None,
                      "score_log_loss":row["sum_loss"]/qualified if qualified else None,
                      "predicted_four_plus":row["sum_four_plus"]/qualified if qualified else None,
                      "observed_four_plus":row["sum_observed_four_plus"]/qualified if qualified else None,
                      "four_plus_brier":row["sum_four_plus_brier"]/qualified if qualified else None}
    return {"eligible_reference_pairs":eligible,"matched_pairs":qualified,
            "excluded_pairs":eligible-qualified,"exclusions":excluded,"models":models}


def main():
    matches,missing=load_matches()
    # Remove any incomplete calendar slate equally for all methods.
    matches=[m for m in matches if m["utcDate"][:10] not in missing]
    reference=evaluate(matches,start="2023-08-01")
    validation=compare_phase(matches,reference,"2023-08-01",SPLIT)
    selected=select_model(validation)
    evaluation=compare_phase(matches,reference,SPLIT)
    print(json.dumps({"source_date_surrogate":True,"development_only":True,
                      "previously_explored_seasons":True,
                      "no_live_or_untouched_accuracy_claim":True,
                      "model_promotion":False,
                      "bookmaker_benchmark":"unavailable_no_timestamped_correct_score_odds",
                      "selection_rule":"lower prior season exact-score log loss; baseline included",
                      "selected_method":selected,"validation":validation,
                      "evaluation":evaluation,
                      "selected_evaluation":evaluation["models"].get(selected)},indent=2))


if __name__=="__main__":
    main()
