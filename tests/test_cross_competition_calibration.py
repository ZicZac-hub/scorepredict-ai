import numpy as np
import pytest
from cross_competition_evaluate import score_metrics, paired_metrics


def test_proper_score_metrics_same_actual_event():
    control=np.array([[.25,.25],[.25,.25]])
    challenger=np.array([[.10,.10],[.70,.10]])
    a=score_metrics(control,(1,0))
    b=score_metrics(challenger,(1,0))
    assert b["exact_log_loss"] < a["exact_log_loss"]
    assert b["outcome_brier"] < a["outcome_brier"]
    assert b["exact_multiclass_brier"] < a["exact_multiclass_brier"]
    report=paired_metrics([{"control":a,"challenger":b}])
    assert report["same_verified_fixtures"]==1
    assert report["paired_differences"]["exact_log_loss"]["mean_control_minus_challenger"]>0
    assert report["paired_differences"]["exact_log_loss"]["descriptive_normal_95_interval"] is None
    assert not report["minimum_review_count_reached"]
    assert not report["automatic_promotion"]


def test_empty_and_invalid_evidence_not_counted_as_loss():
    r=paired_metrics([])
    assert r["same_verified_fixtures"]==0
    assert r["methods"]["challenger"]["mean_exact_log_loss"] is None
    with pytest.raises(ValueError):
        score_metrics([[1.,1.]],(0,0))
    with pytest.raises(ValueError):
        score_metrics([[1.]],(-1,0))


def test_outside_grid_score_is_not_an_exact_hit():
    r=score_metrics([[1.]],(7,2))
    assert not r["top_one_hit"] and not r["top_five_hit"]
    assert r["exact_log_loss"]>30 and r["outcome_brier"]==2
