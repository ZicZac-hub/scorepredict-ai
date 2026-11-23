"""V36 research-only prespecified score-tail challenger.

Transforms an authenticated immutable pre-match score grid. Does not issue
tickets, edit old forecasts, query providers, or promote a live model.
Parameters are fixed before any future prospective use.
"""
import numpy as np

VERSION = "v36-tail-redistribution-research-v1"
METHODS = ("baseline", "tail_temperature_1_10", "tail_shift_0_10")


def transform_score_grid(grid, method):
    """Return a normalized new grid without mutating its source.

    Tail shift emphasizes matches with at least four combined goals.
    This is a testable hypothesis, not a calibrated or validated model.
    """
    original = np.asarray(grid, dtype=float)
    if (original.ndim != 2 or min(original.shape) < 5
            or not np.isfinite(original).all()
            or np.any(original < 0)
            or not np.isclose(original.sum(), 1, rtol=0, atol=1e-6)):
        raise ValueError("invalid_original_score_grid")
    if method == "baseline":
        weights = original.copy()
    elif method == "tail_temperature_1_10":
        weights = original ** (1 / 1.10)
    elif method == "tail_shift_0_10":
        home, away = np.indices(original.shape)
        weights = original * np.exp(0.10 * np.minimum(home + away, 8))
    else:
        raise ValueError("unknown_prespecified_method")
    total = float(weights.sum())
    if not np.isfinite(total) or total <= 0:
        raise ValueError("invalid_challenger_grid")
    return (weights / total).tolist()


def evaluate_fixture(grid, actual):
    """Proper scores and high-goal diagnostics for a verified physical fixture.

    Report all methods on the SAME actual fixture; exclude pending results
    upstream. Do not use this function to retrospectively issue tickets.
    """
    h, a = actual
    if not isinstance(h, int) or not isinstance(a, int) or h < 0 or a < 0:
        raise ValueError("invalid_verified_final_score")
    out = {}
    for method in METHODS:
        p = np.asarray(transform_score_grid(grid, method))
        if h >= p.shape[0] or a >= p.shape[1]:
            # Explicitly mark out-of-grid observations; never silently clip.
            out[method] = {"outside_grid": True, "exact_log_loss": None,
                           "multiclass_brier": None, "over_3_5_brier": None}
            continue
        observed = np.zeros_like(p)
        observed[h, a] = 1.0
        home, away = np.indices(p.shape)
        high = float(p[(home + away) >= 4].sum())
        out[method] = {
            "outside_grid": False,
            "exact_log_loss": float(-np.log(max(p[h, a], 1e-15))),
            "multiclass_brier": float(np.sum((p - observed) ** 2)),
            "over_3_5_brier": float((high - int(h + a >= 4)) ** 2),
            "high_goal_probability": high,
        }
    return {"version": VERSION, "actual_score": [h, a], "methods": out,
            "research_only": True, "no_live_model_change": True}
