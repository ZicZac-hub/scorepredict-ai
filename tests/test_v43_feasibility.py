import math

import pytest

from five_double_accuracy import wilson
from v43_feasibility import model_feasibility, validation_gate


def test_exact_top_five_mass_and_minimum_width_use_normalized_joint_scores():
    first = [[.4, .3], [.2, .1]]
    second = [[.5, .25], [.15, .1]]
    joint = sorted((a * b for row in first for a in row for other in second for b in other), reverse=True)
    report = model_feasibility([first, second])
    assert report["maximum_five_model_mass"] == pytest.approx(math.fsum(joint[:5]))
    count = report["minimum_combinations_for_target_model_mass"]
    assert math.fsum(joint[:count - 1]) < .70 <= math.fsum(joint[:count])
    assert report["five_reaches_target_model_mass"] is False


def test_fixed_marginal_bound_blocks_70_even_with_arbitrary_dependence():
    first, second = [[.1] * 10], [[.7, .1, .1, .1]]
    report = model_feasibility([first, second])
    assert report["single_match_top_five_masses"] == pytest.approx([.50, 1.])
    assert report["marginal_top_five_upper_bound"] == pytest.approx(.50)


def test_mass_can_reach_target_for_a_concentrated_distribution():
    report = model_feasibility([[[.9, .1]], [[.9, .1]]])
    assert report["minimum_combinations_for_target_model_mass"] == 1
    assert report["five_reaches_target_model_mass"] is True
    assert report["observed_accuracy_established"] is False


def test_interval_threshold_requires_more_than_seventy_observed_hits_at_100():
    report = validation_gate(70, 100)
    minimum = report["minimum_hits_for_wilson_lower_bound"]
    assert minimum == 79
    assert wilson(minimum - 1, 100)[0] < .7 <= wilson(minimum, 100)[0]
    assert report["wilson_lower_bound_reaches_target"] is False
    assert validation_gate(minimum, 100)["wilson_lower_bound_reaches_target"] is True


def test_unsettled_and_small_samples_do_not_establish_the_target():
    assert validation_gate(0, 0)["observed_hit_rate"] is None
    assert validation_gate(0, 0)["wilson_95"] is None
    assert validation_gate(2, 2)["minimum_hits_for_wilson_lower_bound"] is None
    with pytest.raises(ValueError):
        validation_gate(2, 1)
    with pytest.raises(ValueError):
        model_feasibility([[[.9]]])
