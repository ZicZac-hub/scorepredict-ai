"""Research-only score accuracy diagnostics for V25 mathematical sensitivities.

Transforming an archived V8 distribution *after* a match does not create a
genuinely frozen V25 forecast. These retrospective comparisons must never be
used as independent evidence for promotion or reported as live V25 accuracy.
"""
import math

import numpy as np

from v21_goal_model_comparison import variants

SENSITIVITIES = (
    "baseline_v8",
    "poisson_marginal",
    "negative_binomial_r3",
    "bivariate_poisson_c012",
    "v25_fixed_equal_mixture_replay",
)


def describe_score(grid, actual):
    """Proper multiclass scores and prespecified goal-shape diagnostics."""
    matrix = np.asarray(grid, dtype=float)
    if (matrix.ndim != 2 or min(matrix.shape) < 5 or
            not np.isfinite(matrix).all() or (matrix < 0).any() or
            not np.isclose(matrix.sum(), 1, atol=1e-6)):
        raise ValueError("invalid_normalized_score_matrix")
    if (len(actual) != 2 or any(type(n) is not int or n < 0 for n in actual)):
        raise ValueError("invalid_verified_regulation_score")
    home, away = actual
    in_support = home < matrix.shape[0] and away < matrix.shape[1]
    score_probability = float(matrix[home, away]) if in_support else 0.0
    rank = np.argsort(-matrix.ravel(), kind="stable")
    cell = home * matrix.shape[1] + away if in_support else -1
    x, y = np.indices(matrix.shape)
    total = x + y
    expected_home = float((matrix * x).sum())
    expected_away = float((matrix * y).sum())
    high_probability = float(matrix[total >= 4].sum())
    low_probability = float(matrix[total <= 1].sum())
    observed_high = int(home + away >= 4)
    observed_low = int(home + away <= 1)
    return {
        "actual_score": f"{home}-{away}",
        "actual_total_goals": home + away,
        "exact_score_probability": score_probability,
        "top_one_hit": int(cell == int(rank[0])),
        "top_three_hit": int(cell in rank[:3]),
        "top_five_hit": int(cell in rank[:5]),
        "exact_score_log_loss": -math.log(max(score_probability, 1e-15)),
        "multiclass_score_brier": float(
            1 + np.square(matrix).sum() - 2 * score_probability
        ),
        "expected_total_goals": expected_home + expected_away,
        "goal_mae": (abs(expected_home - home) + abs(expected_away - away)) / 2,
        "over_3_5_probability": high_probability,
        "over_3_5_observed": observed_high,
        "over_3_5_brier": (high_probability - observed_high) ** 2,
        "under_1_5_probability": low_probability,
        "under_1_5_observed": observed_low,
        "under_1_5_brier": (low_probability - observed_low) ** 2,
    }


def summarize_scores(rows):
    """All rates use identical, verified leg slots; no missing-as-loss."""
    n = len(rows)
    keys = (
        "top_one_hit", "top_three_hit", "top_five_hit",
        "exact_score_log_loss", "multiclass_score_brier", "goal_mae",
        "expected_total_goals", "actual_total_goals",
        "over_3_5_probability", "over_3_5_observed", "over_3_5_brier",
        "under_1_5_probability", "under_1_5_observed", "under_1_5_brier",
    )
    means = {
        "mean_" + key: math.fsum(row[key] for row in rows) / n if n else None
        for key in keys
    }
    return {
        "verified_leg_slots": n,
        "top_one_exact_hits": sum(row["top_one_hit"] for row in rows),
        "top_three_exact_hits": sum(row["top_three_hit"] for row in rows),
        "top_five_exact_hits": sum(row["top_five_hit"] for row in rows),
        "high_goal_4_plus_observed": sum(row["over_3_5_observed"] for row in rows),
        "low_goal_0_or_1_observed": sum(row["under_1_5_observed"] for row in rows),
        "high_goal_probability_gap_observed_minus_predicted": (
            means["mean_over_3_5_observed"] - means["mean_over_3_5_probability"]
            if n else None
        ),
        "total_goals_bias_actual_minus_expected": (
            means["mean_actual_total_goals"] - means["mean_expected_total_goals"]
            if n else None
        ),
        **means,
    }


def replay_sensitivities(original_v8, actual):
    """Exploratory, *not independently frozen*, transformations of one V8 grid.

    No fitted weights, target tuning, betting-market priors, news, or provider calls.
    In particular, these historical scores cannot retrospectively validate
    prospective V25 selection accuracy.
    """
    base = np.asarray(original_v8, dtype=float)
    candidates = variants(base)
    mixture = np.zeros_like(base)
    for name in ("poisson_marginal", "negative_binomial_r3", "bivariate_poisson_c012"):
        component = candidates[name]
        mixture[:component.shape[0], :component.shape[1]] += component / 3
    mixture /= mixture.sum()
    candidates["v25_fixed_equal_mixture_replay"] = mixture
    return {
        name: describe_score(candidates[name], actual)
        for name in SENSITIVITIES
    }
