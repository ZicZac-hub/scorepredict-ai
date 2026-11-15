"""V14 Robust-Agreement Five Doubles — independent research challenger.

Research basis: calibrated score probabilities matter more than cosmetic
score diversity. This selector ranks five mutually exclusive exact-score
pairs by the lower *joint* probability assigned by V7 and V8, and can rank
the TWO fixtures by the sum of that lower envelope. It uses both complete,
genuine, normalized pre-match score grids; no AI opinions, bookmaker odds,
post-kickoff results, or invented scores. This LOWER ENVELOPE is a
conservative model criterion, not a guaranteed lower bound on reality.

V7 and V8 share statistical evidence and are NOT independent witnesses.
The 2026 historical data have been inspected during model development.
V14 is shadow / opt-in only until separate prospective validation.
"""
from datetime import datetime, timezone
from itertools import combinations
import numpy as np
from hybrid_five_v13 import valid_pair, one_date, VERSION as HYBRID_VERSION

VERSION = "ziczac-robust-agreement-v14-experimental"
MODELS = ("v7", "v8")
MIN_SCORE = 0.0
MIN_LEAD_SECONDS = 300


def grids(row):
    """Validate two complete score probability distributions, or abstain."""
    result = row.get("result") or {}
    raw = result.get("component_score_matrices") or {}
    versions = result.get("component_versions") or {}
    if not all(versions.get(name) and name in raw for name in MODELS):
        return None
    found = {}
    for name in MODELS:
        try:
            matrix = np.asarray(raw[name], dtype=float)
        except (TypeError, ValueError):
            return None
        if (matrix.ndim != 2 or matrix.shape[0] < 2 or
            matrix.shape[1] < 2 or not np.isfinite(matrix).all() or
            np.any(matrix < 0) or not np.isclose(matrix.sum(), 1.0, atol=1e-6)):
            return None
        found[name] = matrix
    if found["v7"].shape != found["v8"].shape:
        return None
    return found


def _joint(a, b):
    return np.outer(a.ravel(), b.ravel())


def robust_pair(a, b, *, now=None, min_coverage=MIN_SCORE):
    """Five exact pairs maximising the sum of min(P_v7, P_v8) per cell.

    This is the exact optimum of this robust criterion over the complete
    two-match score grids (not a top-ten truncation). Each selection is an
    exclusive two-match outcome. No odds or observed hit rates are implied.
    """
    if not valid_pair(a, b, now=now):
        return None
    first, second = grids(a), grids(b)
    if first is None or second is None:
        return None
    if first["v7"].shape != second["v7"].shape:
        return None
    p7 = _joint(first["v7"], second["v7"])
    p8 = _joint(first["v8"], second["v8"])
    envelope = np.minimum(p7, p8)
    # Each source grid is a full normalized probability distribution,
    # so all 5 pair options are drawn from physically possible score cells.
    ids = np.argsort(-envelope.ravel(), kind="stable")[:5]
    if len(ids) != 5:
        return None
    n1=first["v7"].shape[1]
    n2=second["v7"].shape[1]
    width=second["v7"].size
    picks=[]
    for idx in ids:
        left,right=divmod(int(idx),width)
        h,away=divmod(left,n1)
        h2,away2=divmod(right,n2)
        picks.append({
            "Score 1":f"{h}-{away}","Score 2":f"{h2}-{away2}",
            "Joint":float(envelope.ravel()[idx]),
            "V7 joint":float(p7.ravel()[idx]),
            "V8 joint":float(p8.ravel()[idx]),
        })
    coverage=sum(p["Joint"] for p in picks)
    if coverage < min_coverage:
        return None
    v8best=float(np.sort(p8.ravel())[-5:].sum())
    # Agreement is a model-distribution diagnostic, not skill or a confidence interval.
    total_variation={
        name:float(np.abs(first["v7"]-first["v8"]).sum()/2)
        if name=="first" else float(np.abs(second["v7"]-second["v8"]).sum()/2)
        for name in ("first","second")}
    return {
        "first":a,"second":b,"doubles":picks,
        "model_version":VERSION,
        "selection_mode":"v14_robust_min_joint",
        "estimated_joint_coverage":coverage,
        "model_coverage_v7":sum(p["V7 joint"] for p in picks),
        "model_coverage_v8":sum(p["V8 joint"] for p in picks),
        "v8_optimal_coverage_same_pair":v8best,
        "robust_lower_envelope":True,
        "conservative_not_a_calibrated_probability":True,
        "agreement_total_variation":total_variation,
        "four_distinct_teams":True,
        "sydney_date":one_date(a),
        "five_doubles_distinct":True,
        "original_forecasts_changed":False,
        "is_unvalidated_challenger":True,
        "research_basis":"Minimum full-grid two-leg probability of V7 and V8, no model training",
    }


def rank_robust_pairs(rows, *, now=None, limit=1, min_coverage=MIN_SCORE):
    """Choose at most limit highest robust-covered pairs on one Sydney day."""
    if limit<1:
        return []
    selected=[]
    for a,b in combinations(rows,2):
        option=robust_pair(a,b,now=now,min_coverage=min_coverage)
        if option is not None:
            selected.append(option)
    selected.sort(key=lambda item:(
        -item["estimated_joint_coverage"],
        -min(item["model_coverage_v7"],item["model_coverage_v8"]),
        item["sydney_date"],
        str((item["first"].get("fixture") or {}).get("id")),
        str((item["second"].get("fixture") or {}).get("id"))))
    return selected[:limit]
