import numpy as np
import pytest
from v36_tail_calibration_research import METHODS, transform_score_grid, evaluate_fixture


def test_v36_normalized_and_immutable():
    grid = np.full((6, 6), 1 / 36)
    snapshot = grid.copy()
    for name in METHODS:
        result = np.asarray(transform_score_grid(grid, name))
        assert result.shape == (6, 6)
        assert np.isclose(result.sum(), 1)
        assert (result >= 0).all()
    assert np.array_equal(grid, snapshot)


def test_v36_tail_hypothesis_and_proper_scores():
    grid = np.full((6, 6), 1 / 36)
    report = evaluate_fixture(grid, (3, 2))
    assert report["research_only"]
    assert report["methods"]["tail_shift_0_10"]["high_goal_probability"] > report["methods"]["baseline"]["high_goal_probability"]
    assert all(report["methods"][m]["exact_log_loss"] is not None for m in METHODS)


def test_v36_rejects_invalid_and_does_not_clip():
    with pytest.raises(ValueError):
        transform_score_grid(np.full((5, 5), -1), "baseline")
    with pytest.raises(ValueError):
        transform_score_grid(np.full((5, 5), 1 / 25), "unregistered")
    report = evaluate_fixture(np.full((5, 5), 1 / 25), (5, 1))
    assert all(report["methods"][m]["outside_grid"] for m in METHODS)
    assert all(report["methods"][m]["exact_log_loss"] is None for m in METHODS)
