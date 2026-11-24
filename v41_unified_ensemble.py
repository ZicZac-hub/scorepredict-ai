"""V41 fixed grouped probability ensemble; not a proven accuracy increase.

Shared V8-derived methods are correlated hypotheses, not independent votes.
Each family gets a fixed budget before any new results. Optional xG and market
updates must qualify on BOTH legs at the common cutoff. Never fill missing
evidence with fabricated values; no model training, external calls or promotion.
"""
import numpy as np
from v21_goal_model_comparison import variants
from v35_continuous_double_trials import score_grid
from v36_tail_calibration_research import transform_score_grid as tail
from v38_margin_calibration import transform_score_grid as margin
from v22_evidence_models import combine_goal_evidence, calibrate_markets

WEIGHTS = {"v8": .40, "v25_anchor": .20, "v35_shape": .10,
           "v36_tail": .10, "v38_margin": .10, "v39_poisson": .10}
V35_METHODS = ("temperature_flatten_1_15", "temperature_sharpen_0_85",
               "high_goal_tilt_0_15", "low_goal_tilt_0_15")
V36_METHODS = ("tail_temperature_1_10", "tail_shift_0_10")


def normalized(grid, shape=None):
    p = np.asarray(grid, dtype=float)
    if (p.ndim != 2 or min(p.shape)<5 or not np.isfinite(p).all()
            or (p<0).any() or not np.isclose(p.sum(),1,rtol=0,atol=1e-6)
            or (shape is not None and p.shape != shape)):
        raise ValueError("invalid_unified_distribution")
    return p/p.sum()


def core(grid):
    base = normalized(grid)
    alternatives = variants(base)
    sensitivity = np.zeros_like(base)
    for name in ("poisson_marginal", "negative_binomial_r3", "bivariate_poisson_c012"):
        p = normalized(alternatives[name])
        if any(a>b for a,b in zip(p.shape,base.shape)):
            raise ValueError("incompatible_sensitivity_grid")
        sensitivity[:p.shape[0],:p.shape[1]] += p/3
    families = {
        "v8": base,
        "v25_anchor": .75*base+.25*sensitivity,
        "v35_shape": np.mean([score_grid({"grid":base},m) for m in V35_METHODS],axis=0),
        "v36_tail": np.mean([tail(base,m) for m in V36_METHODS],axis=0),
        "v38_margin": margin(base,"conditional_margin_125"),
        "v39_poisson": alternatives["poisson_marginal"],
    }
    return normalized(sum(WEIGHTS[k]*normalized(v,base.shape) for k,v in families.items()),base.shape)


def final_pair(grids, bundles):
    if len(grids)!=2 or len(bundles)!=2:
        raise ValueError("two_original_legs_required")
    current = [core(g) for g in grids]
    used = {"xg":False,"market":False}
    for kind in ("xg","market"):
        if not all(b.get(kind) is not None for b in bundles):
            continue
        if kind=="xg":
            updates=[combine_goal_evidence(p,b[kind]["value"],weight=.25)
                     for p,b in zip(current,bundles)]
        else:
            updates=[calibrate_markets(p,b[kind]["value"],strength=.35,max_move=.15)
                     for p,b in zip(current,bundles)]
        if all(x.get("ready") for x in updates):
            current=[normalized(x["matrix"],p.shape) for x,p in zip(updates,current)]
            used[kind]=True
    return current,used
