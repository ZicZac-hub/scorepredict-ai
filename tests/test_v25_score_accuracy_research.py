"""V25 score-distribution research must not rewrite any pre-kickoff prediction."""
import math

import numpy as np
import pytest

from v25_score_accuracy_research import (
    SENSITIVITIES, describe_score, replay_sensitivities, summarize_scores,
)


def test_exact_hit_and_high_goal_calibration_are_mathematically_consistent():
    matrix = np.zeros((5, 5))
    matrix[3, 1] = .8
    matrix[0, 0] = .2
    observed = describe_score(matrix, (3, 1))
    assert observed["top_one_hit"] == 1
    assert observed["top_three_hit"] == 1
    assert observed["top_five_hit"] == 1
    assert observed["exact_score_log_loss"] == pytest.approx(-math.log(.8))
    assert observed["multiclass_score_brier"] == pytest.approx(.08)
    assert observed["over_3_5_probability"] == pytest.approx(.8)
    assert observed["over_3_5_brier"] == pytest.approx(.04)
    assert observed["under_1_5_probability"] == pytest.approx(.2)
    assert observed["under_1_5_observed"] == 0
    assert observed["expected_total_goals"] == pytest.approx(3.2)
    assert observed["goal_mae"] == pytest.approx(.4)


def test_rare_high_score_is_a_miss_not_a_rewritten_prediction():
    grid = np.zeros((5, 5))
    grid[0, 0] = 1
    first = describe_score(grid, (5, 2))
    assert first["top_one_hit"] == 0
    assert first["top_five_hit"] == 0
    assert first["exact_score_probability"] == 0
    assert math.isfinite(first["exact_score_log_loss"])
    assert first["over_3_5_observed"] == 1
    assert first["over_3_5_probability"] == 0
    summary = summarize_scores([first])
    assert summary["verified_leg_slots"] == 1
    assert summary["high_goal_probability_gap_observed_minus_predicted"] == 1


def test_no_verified_legs_means_unavailable_accuracy():
    summary = summarize_scores([])
    assert summary["verified_leg_slots"] == 0
    assert summary["mean_exact_score_log_loss"] is None
    assert summary["high_goal_probability_gap_observed_minus_predicted"] is None


def test_same_source_retrospective_transform_preserves_input():
    a = np.array([.2, .3, .25, .15, .1])
    b = np.array([.32, .35, .18, .1, .05])
    matrix = np.outer(a, b)
    original = matrix.copy()
    replay = replay_sensitivities(matrix, (3, 1))
    assert tuple(replay) == SENSITIVITIES
    assert np.array_equal(matrix, original)
    for item in replay.values():
        assert 0 <= item["exact_score_probability"] <= 1
        assert 0 <= item["over_3_5_probability"] <= 1
        assert math.isfinite(item["exact_score_log_loss"])
        assert item["actual_score"] == "3-1"


@pytest.mark.parametrize("bad_score", [(-1, 0), (1.0, 2), (0, True)])
def test_reject_invalid_observed_scores(bad_score):
    with pytest.raises(ValueError, match="invalid_verified_regulation_score"):
        describe_score(np.ones((5, 5)) / 25, bad_score)
