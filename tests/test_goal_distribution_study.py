"""Synthetic-only mechanics tests; not observed match accuracy."""
import math
import numpy as np
from engine import _grid
from v21_goal_model_comparison import variants
from run_goal_distribution_study import METHODS, grade_pair, select_model

def test_same_actual_scores_and_valid_probability_grids():
    source=_grid(1.5,1.1)
    saved=source.copy()
    choices=variants(source)
    assert tuple(choices)==METHODS
    np.testing.assert_array_equal(source,saved)
    for method,matrix in choices.items():
        assert np.isclose(matrix.sum(),1.)
        assert np.isfinite(matrix).all()
        assert matrix.min()>=0
        result=grade_pair([matrix,matrix],((1,0),(1,0)))
        assert result["hit"] in (0,1)
        assert 0 < result["coverage"] <= 1.
        assert math.isfinite(result["score_log_loss"])
        assert 0. <= result["four_plus_probability"] <= 1.
        assert result["four_plus_observed"]==0.
        assert np.isclose(result["four_plus_brier"],result["four_plus_probability"]**2)

def test_model_selection_prefers_log_loss_not_retrospective_hit_count():
    sample={"matched_pairs":40,"models":{
        name:{"score_log_loss":3.5+i/100,"hits":10+i} for i,name in enumerate(METHODS)}}
    assert select_model(sample)=="baseline_v8"
    sample["models"]["bivariate_poisson_c012"]["score_log_loss"]=3.0
    assert select_model(sample)=="bivariate_poisson_c012"
    sample["matched_pairs"]=29
    assert select_model(sample) is None

def test_actual_score_beyond_grid_is_not_invented():
    matrix=_grid(1.2,.8)
    outcome=grade_pair([matrix,matrix],((99,0),(1,0)))
    assert outcome["hit"]==0
    assert outcome["score_log_loss"]>=-math.log(1e-12)/2
