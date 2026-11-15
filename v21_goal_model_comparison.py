"""V21 fixed, independently evaluable exact-score model challengers.

All score grids share the original expected-goal means. Independent Poisson,
overdispersed negative-binomial and correlated bivariate Poisson are compared
WITHOUT fitting on matches whose results are being assessed. They are hypotheses,
not an automatic replacement for the existing frozen forecasts.
"""
import math
import numpy as np


MAX_GOALS = 10
MODEL_NAMES = ("baseline_v8", "poisson_marginal", "negative_binomial_r3",
               "bivariate_poisson_c012")


def _probabilities(grid):
    matrix = np.asarray(grid, dtype=float)
    if matrix.ndim != 2 or min(matrix.shape) < 5 or max(matrix.shape)>40:
        raise ValueError("Complete 2-D score grid required")
    if np.any(~np.isfinite(matrix)) or np.any(matrix < 0) or abs(matrix.sum()-1)>1e-5:
        raise ValueError("Invalid probability mass")
    return matrix/matrix.sum()


def _poisson(rate, length):
    values = np.zeros(length)
    values[0] = math.exp(-rate)
    for k in range(1,length):
        values[k] = values[k-1]*rate/k
    values[-1] += max(0, 1-float(values.sum()))
    return values/values.sum()


def _nb(rate, shape, length):
    """Gamma-Poisson mixture with E[goals]=rate and Var=rate+rate²/shape."""
    p = shape/(shape+rate)
    out = np.zeros(length)
    out[0] = p**shape
    for i in range(1,length):
        out[i] = out[i-1]*(i-1+shape)/i*(1-p)
    out[-1] += max(0,1-float(out.sum()))
    return out/out.sum()


def _correlated(lam_h, lam_a, length, covariance=.12):
    shared = min(max(0.,covariance),.8*min(lam_h,lam_a))
    h = _poisson(max(.0001,lam_h-shared),length)
    a = _poisson(max(.0001,lam_a-shared),length)
    z = _poisson(shared,length)
    out=np.zeros((length,length),dtype=float)
    for i in range(length):
        for j in range(length):
            out[i,j] = sum(z[k]*h[i-k]*a[j-k]
                           for k in range(min(i,j)+1))
    # Residual tails are tiny for capped normal-rate football models; the
    # final cell truncation is audited by mass rather than made-up scores.
    out/=out.sum()
    return out


def variants(original):
    matrix = _probabilities(original)
    h,a=matrix.shape
    ix=np.arange(h)[:,None]
    iy=np.arange(a)[None,:]
    mu_h=float((matrix*ix).sum())
    mu_a=float((matrix*iy).sum())
    if not (.15<=mu_h<=5 and .15<=mu_a<=5):
        raise ValueError("Observed original means outside reliable football modelling range")
    return {
        "baseline_v8":matrix,
        "poisson_marginal":np.outer(_poisson(mu_h,h),_poisson(mu_a,a)),
        "negative_binomial_r3":np.outer(_nb(mu_h,3.0,h),_nb(mu_a,3.0,a)),
        "bivariate_poisson_c012":_correlated(mu_h,mu_a,min(h,a))
    }


def top_scores(matrix, count=10):
    grid=_probabilities(matrix)
    choices=sorted(((i,j,float(grid[i,j])) for i in range(grid.shape[0])
                    for j in range(grid.shape[1])),
                   key=lambda x:(-x[2],x[0],x[1]))[:count]
    return [[int(a),int(b),p] for a,b,p in choices]


def inspect_prediction(matrix, actual_home, actual_away):
    grid=_probabilities(matrix)
    if type(actual_home) is not int or type(actual_away) is not int or min(actual_home,actual_away)<0:
        raise ValueError("Full-time regulation goals required")
    h,a=grid.shape
    p_score=float(grid[actual_home,actual_away]) if actual_home<h and actual_away<a else 0.0
    top=top_scores(grid,5)
    x=np.arange(h)[:,None]
    y=np.arange(a)[None,:]
    expected_h=float((grid*x).sum())
    expected_a=float((grid*y).sum())
    total_over_2_5=float(grid[(x+y>=3)].sum())
    total_over_3_5=float(grid[(x+y>=4)].sum())
    observed_over_2_5=int(actual_home+actual_away>=3)
    p_outcome={"home":float(np.tril(grid,-1).sum()),
               "draw":float(np.trace(grid)),
               "away":float(np.triu(grid,1).sum())}
    outcome={"home":int(actual_home>actual_away),
             "draw":int(actual_home==actual_away),
             "away":int(actual_home<actual_away)}
    targets = {(2,2): "2-2", (3,1): "3-1", (3,2): "3-2"}
    return {
        "exact_hit":int((actual_home,actual_away)==tuple(top[0][:2])),
        "top5_hit":int(any([actual_home,actual_away]==score[:2] for score in top)),
        "goal_mae":(abs(expected_h-actual_home)+abs(expected_a-actual_away))/2,
        "exact_log_loss":-math.log(max(p_score,1e-12)),
        "outcome_brier":sum((p_outcome[key]-outcome[key])**2 for key in outcome),
        "over_2_5_brier":(total_over_2_5-observed_over_2_5)**2,
        "predicted_over_2_5":total_over_2_5,
        "predicted_over_3_5":total_over_3_5,
        "observed_over_2_5":observed_over_2_5,
        "observed_over_3_5":int(actual_home+actual_away>=4),
        "high_score_probs": {label:float(grid[i,j]) if i<h and j<a else 0.0
                             for (i,j),label in targets.items()},
        "actual_high_score":int(actual_home+actual_away>=4),
        "predicted_high_score_mass":total_over_3_5,
        "actual_score":f"{actual_home}-{actual_away}",
        "most_likely_score":f"{top[0][0]}-{top[0][1]}"
    }


def compare_frozen(rows):
    """Read-only paired comparison; each row has ONE original pre-match grid.

    These fixed alternative transformations of the archived original are
    RETROSPECTIVE DIAGNOSTICS until independently frozen for new kickoffs.
    Never present a historical transformation as a genuinely frozen forecast.
    """
    cohorts={name:[] for name in MODEL_NAMES}
    for row in rows:
        try:
            variants_grid=variants(row["matrix"])
            for name in MODEL_NAMES:
                cohorts[name].append(inspect_prediction(
                    variants_grid[name],row["home_goals"],row["away_goals"]))
        except (ValueError,TypeError,KeyError):
            continue
    result={}
    for name,metrics in cohorts.items():
        count=len(metrics)
        numeric=("exact_hit","top5_hit","goal_mae","exact_log_loss",
                 "outcome_brier","over_2_5_brier","predicted_over_2_5",
                 "observed_over_2_5","actual_high_score","predicted_high_score_mass")
        result[name]={"verified":count,"exact_hits":sum(x["exact_hit"] for x in metrics),
                      "top5_hits":sum(x["top5_hit"] for x in metrics),
                      **{("mean_"+k): (sum(x[k] for x in metrics)/count if count else None)
                         for k in numeric},
                      "high_goal_miss_examples": [
                          {"actual":x["actual_score"],"predicted":x["most_likely_score"]}
                          for x in metrics if x["actual_high_score"] and not x["exact_hit"]][:10]}
    return {"kind":"retrospective_same_original_pre_match_matrix_not_prospective",
            "model_promotion":False, "evidence":result,
            "needs_independent_future_verification":True,
            "note":"A good retrospective number is NOT a verified prospective improvement."}


def walk_forward(finished_matches, forecast_fn, *, min_prior=45, max_eval=1000):
    """Chronological walk-forward replay, previously available results ONLY.

    forecast_fn(fixture_with_unset_result, prior_finished, as_of) -> full grid.
    Freeze features one hour before each kickoff; exclude recent fixtures
    whose 3-hour conservative result-availability delay has not elapsed.
    """
    from datetime import timedelta
    from copy import deepcopy
    from v10_prospective import utc
    games=[]
    for row in finished_matches:
        try:
            if row.get("status") != "FINISHED": continue
            t=utc(row["utcDate"])
            full=(row.get("score") or {}).get("fullTime") or {}
            if type(full.get("home")) is not int or type(full.get("away")) is not int:
                continue
            games.append((t,row))
        except (TypeError,KeyError,ValueError):
            continue
    games.sort(key=lambda x:(x[0],str(x[1].get("id"))))
    paired=[]
    for t,match in games:
        as_of=t-timedelta(hours=1)
        prior=[x for when,x in games if when+timedelta(hours=3) <= as_of]
        if len(prior)<min_prior:
            continue
        fixture=deepcopy(match)
        fixture["status"]="SCHEDULED"
        fixture["score"]={"fullTime":{"home":None,"away":None}}
        original=forecast_fn(fixture,prior,as_of)
        if original is None:
            continue
        goals=match["score"]["fullTime"]
        paired.append({"matrix":original,"home_goals":goals["home"],
                       "away_goals":goals["away"],"kickoff_utc":t.isoformat(),
                       "fixture_id":str(match.get("id"))})
        if len(paired)>=max_eval:
            break
    return {"matched_test_fixtures":len(paired),"comparison":compare_frozen(paired),
            "test_type":"historical_chronological_replay_not_live_prospective"}
