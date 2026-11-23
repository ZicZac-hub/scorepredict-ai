"""V28 original-decision feeder is visible without creating fake successes."""
from datetime import datetime, timezone

from continuous_accuracy_bot import assemble


def test_v28_issuance_report_is_not_a_verified_hit():
    report = assemble(
        None, None, None, None, None,
        now=datetime(2035, 1, 1, tzinfo=timezone.utc),
        v28={
            "freeze": {"created": [{"day": "2035-01-02", "policy": "FROZEN"}],
                       "source_ineligibility": {"outside_research_freeze_window": 12}},
            "verified_comparison": {
                "original_frozen_decisions": 3, "verified_paired_days": 1,
                "pending_days": 2,
            },
        },
    )
    feed = report["v28_upstream_capture"]
    assert feed["new_original_decisions_last_run"] == 1
    assert feed["original_frozen_decisions"] == 3
    assert feed["verified_paired_days"] == 1
    assert feed["pending_days"] == 2
    assert feed["source_ineligibility"]["outside_research_freeze_window"] == 12
    assert report["goal_verified"] is False
    assert report["production_model_changed_by_bot"] is False


def test_missing_v28_feeder_does_not_invent_new_decisions():
    report = assemble(
        None, None, None, None, None,
        now=datetime(2035, 1, 1, tzinfo=timezone.utc),
    )
    assert report["v28_upstream_capture"]["new_original_decisions_last_run"] == 0
    assert report["v28_upstream_capture"]["verified_paired_days"] == 0
    assert report["v28_upstream_capture"]["pending_days"] == 0
