"""V42 preregistered score-shape hypotheses. Research only, never live promotion."""
import numpy as np
from v41_unified_ensemble import normalized

def hypotheses(grid):
    p = normalized(grid)
    h, a = np.indices(p.shape)
    total = h + a
    # Small fixed perturbations, not fitted to the first V41 loss.
    def tilt(weight):
        q = p * weight
        return normalized(q / q.sum(), p.shape)
    low = tilt(np.where(total <= 1, 1.25, 1.0))
    high = tilt(np.where(total >= 4, 1.25, 1.0))
    wide = tilt(np.where(np.abs(h-a) >= 3, 1.25, 1.0))
    # Reverse a small amount of decisive-outcome mass without changing draws.
    upset = p.copy()
    if p.shape[0] == p.shape[1]:
        upset = .95*p + .05*p.T
    return {"low_goal":low, "high_goal":high, "wide_margin":wide,
            "upset_sensitivity":normalized(upset,p.shape)}

def ensemble(grid):
    p = normalized(grid)
    arms = hypotheses(p)
    return normalized(.60*p + .10*sum(arms.values()),p.shape)
