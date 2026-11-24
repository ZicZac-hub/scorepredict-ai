"""Fixed, research-only conditional winning-margin calibration.

Motivation: archived V8-near total-goal calibration was nearly unbiased,
while margins >=3 were underrepresented. Preserve each total-goal band's
original mass exactly; test a predeclared 25% relative weight for margins
>=3 within that band. No fitting, provider calls or live changes.
"""
import numpy as np

METHODS = ("baseline", "conditional_margin_125")


def _grid(grid):
    p = np.asarray(grid, dtype=float)
    if (p.ndim != 2 or min(p.shape) < 5 or not np.isfinite(p).all()
            or np.any(p < 0) or not np.isclose(p.sum(), 1, rtol=0, atol=1e-6)):
        raise ValueError("invalid_original_score_grid")
    return p / p.sum()


def transform_score_grid(grid, method):
    p = _grid(grid)
    if method == "baseline":
        return p.tolist()
    if method != "conditional_margin_125":
        raise ValueError("unknown_prespecified_method")
    h, a = np.indices(p.shape)
    total = h + a
    weighted = p * np.where(np.abs(h-a) >= 3, 1.25, 1.0)
    out = np.zeros_like(p)
    for goals in range(sum(p.shape)-1):
        mask = total == goals
        mass = float(p[mask].sum())
        if mass:
            out[mask] = weighted[mask] * (mass / float(weighted[mask].sum()))
    return out.tolist()


def margin_brier(grid, actual):
    h, a = actual
    if type(h) is not int or type(a) is not int or min(h, a) < 0:
        raise ValueError("invalid_verified_final_score")
    p = _grid(grid)
    i, j = np.indices(p.shape)
    probability = float(p[np.abs(i-j) >= 3].sum())
    return (probability - int(abs(h-a) >= 3)) ** 2
