"""Pre-kickoff, write-once research probabilities for the paired xG study.

Only called while freezing a genuine source packet and its exact as-of history.
This module never updates a live model, fetches data, or learns from results.
The stored matrices are subsequently graded *as frozen*: changing model code
cannot silently alter a historical prediction after the outcome is known.
"""
import math

import numpy as np

from dynamic_dixon_coles import fit_forecast
from verified_xg_challenger import predict_shadow

SHADOW_POLICY = "dc-vs-verified-xg-frozen-probabilities-v1"
LINEUP_STATES = {"both_confirmed", "partial", "unavailable"}


def _matrix(value):
    try:
        matrix = np.asarray(value, dtype=float)
    except (TypeError, ValueError, OverflowError):
        return None
    if (matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]
            or not 2 <= matrix.shape[0] <= 25
            or not np.isfinite(matrix).all()
            or (matrix < 0).any()
            or not math.isclose(float(matrix.sum()), 1., abs_tol=1e-6)):
        return None
    return matrix


def freeze_shadow(fixture, history, snapshot):
    """Construct both model predictions once, without consulting actual goals.

    Abstentions are retained so that missing provider coverage does not
    disappear from the research denominator or become fabricated xG.
    """
    if not snapshot.get("verified_historical_fixture_xg"):
        return {"status": "abstained", "reason": "no_authenticated_historical_xg"}
    try:
        baseline = fit_forecast(fixture, history)
        if baseline is None:
            return {"status": "abstained", "reason": "baseline_insufficient_history"}
        challenger = predict_shadow(
            fixture, history, snapshot, as_of=snapshot["captured_at_utc"])
        if challenger is None:
            return {"status": "abstained", "reason": "verified_xg_not_eligible"}
        dc_matrix = _matrix(baseline["matrix"])
        xg_matrix = _matrix(challenger["matrix"])
        if dc_matrix is None:
            return {"status": "abstained", "reason": "invalid_baseline_probabilities"}
        if xg_matrix is None:
            return {"status": "abstained", "reason": "invalid_xg_probabilities"}
        # The shared goal-grid algorithm expands adaptively at higher rates.
        # Valid normalized matrices need not share dimensions to be paired.
        if challenger.get("lineup_status") not in LINEUP_STATES:
            return {"status": "abstained", "reason": "invalid_lineup_evidence_category"}
        return {
            "status": "ready", "policy": SHADOW_POLICY,
            "baseline_model_version": str(baseline["model_version"]),
            "xg_model_version": str(challenger["model_version"]),
            "lineup_status": challenger["lineup_status"],
            "lineup_numeric_weight": 0.0,
            "dynamic_baseline_matrix": dc_matrix.tolist(),
            "verified_xg_matrix": xg_matrix.tolist(),
        }
    except (KeyError, TypeError, ValueError, OverflowError, FloatingPointError):
        return {"status": "abstained", "reason": "shadow_calculation_failed"}


def validated_shadow(value):
    """Read archived probabilities only if their predeclared format is sound."""
    if not isinstance(value, dict) or value.get("status") != "ready":
        return None
    if (value.get("policy") != SHADOW_POLICY
            or value.get("lineup_status") not in LINEUP_STATES
            or value.get("lineup_numeric_weight") != 0.0
            or not all(isinstance(value.get(key), str) and value[key]
                       for key in ("baseline_model_version", "xg_model_version"))):
        return None
    baseline = _matrix(value.get("dynamic_baseline_matrix"))
    challenger = _matrix(value.get("verified_xg_matrix"))
    if baseline is None or challenger is None:
        return None
    return baseline, challenger, value["lineup_status"]
