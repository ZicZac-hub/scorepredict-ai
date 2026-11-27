"""Measure five exact-score doubles against a target without changing forecasts."""
import math

from five_double_accuracy import _validated_grid, wilson
from v25_five_double_frontier import frontier

TARGET = 0.70
CHECKPOINTS = (5, 10, 20, 50, 100, 150, 200, 500)


def _validate_target(target):
    if type(target) not in (float, int) or not math.isfinite(target) or not 0 < target <= 1:
        raise ValueError("The target must be a probability greater than zero and at most one.")


def model_feasibility(grids, *, target=TARGET):
    """Exact ranked joint mass under independence and a bound given the marginals.

    Five joint outcomes touch at most five scores for either match. The
    smaller of the two top-five marginal sums is therefore an upper bound
    for *any* dependence structure with those same marginal distributions.
    Both results concern the supplied forecasts and need calibration.
    """
    if len(grids) != 2:
        raise ValueError("Two full score distributions are required.")
    first, second = [_validated_grid(grid) for grid in grids]
    ranked = frontier(first, second, target=target, checkpoints=CHECKPOINTS)
    marginal = [
        min(1.0, math.fsum(sorted((p for row in grid for p in row), reverse=True)[:5]))
        for grid in (first, second)
    ]
    return {
        "target": target,
        "maximum_five_model_mass": ranked["model_max_five_probability_mass"],
        "minimum_combinations_for_target_model_mass": ranked["minimum_joint_combinations_for_target_model_mass"],
        "five_reaches_target_model_mass": ranked["target_met_by_five_model_choices"],
        "target_count_status": ranked["target_count_status"],
        "checkpoints_model_mass": ranked["checkpoints_model_mass"],
        "single_match_top_five_masses": marginal,
        "marginal_top_five_upper_bound": min(marginal),
        "requires_independent_match_model": True,
        "observed_accuracy_established": False,
    }


def validation_gate(hits, settled, *, target=TARGET):
    """Describe a fixed-sample Wilson criterion for independent settled pairs.

    For a real performance claim, use a prospectively specified selection
    policy and one observation per independent unit. Reviewing overlapping
    historical pairs or repeatedly checking the result does not validate it.
    """
    _validate_target(target)
    if type(hits) is not int or type(settled) is not int or not 0 <= hits <= settled:
        raise ValueError("Hits and settled pairs must be nonnegative counts, with hits no greater than settled.")
    interval = wilson(hits, settled)
    perfect = wilson(settled, settled)
    minimum = None
    if perfect and perfect[0] >= target:
        low, high = 0, settled
        while low < high:
            middle = (low + high) // 2
            if wilson(middle, settled)[0] >= target:
                high = middle
            else:
                low = middle + 1
        minimum = low
    return {
        "target": target,
        "hits": hits,
        "settled": settled,
        "observed_hit_rate": hits / settled if settled else None,
        "wilson_95": interval,
        "minimum_hits_for_wilson_lower_bound": minimum,
        "wilson_lower_bound_reaches_target": interval[0] >= target if interval else False,
        "requires_independent_settled_pairs": True,
    }
