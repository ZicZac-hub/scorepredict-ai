"""Predeclared, future-only comparison of the original V8, V38 and Poisson.

Use exactly the Poisson marginal transformation investigated in V34.
No fitting, post-result selection, external requests or live promotion.
The original grid's finite support and last-cell tail convention are retained.
"""
from v21_goal_model_comparison import variants
from v38_margin_calibration import transform_score_grid as margin_transform

METHODS = ("baseline", "conditional_margin_125", "poisson_marginal")


def transform_score_grid(grid, method):
    if method in ("baseline", "conditional_margin_125"):
        return margin_transform(grid, method)
    if method == "poisson_marginal":
        return variants(grid)["poisson_marginal"].tolist()
    raise ValueError("unknown_prespecified_method")
