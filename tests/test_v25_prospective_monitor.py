"""Prospective V25 evidence must be frozen before results; old sets stay old."""
from copy import deepcopy

import pytest

from v25_prospective_monitor import PROBABILITY_REQUIRED_FROM, build_monitor


def audit(rows=None, *, archived=None, excluded=None, probability_pairs=0):
    rows = rows or []
    return {
        "archived_sets_found": len(rows) if archived is None else archived,
        "source_valid_sets": len(rows),
        "source_exclusions": {} if excluded is None else excluded,
        "matched_probability_calibration": {"verified_pairs": probability_pairs},
        "rows": rows,
    }


def legacy(day="2026-11-19", status="verified"):
    return {
        "sydney_date": day,
        "selection_time_utc": "2026-11-17T23:11:34+00:00",
        "status": status,
        "v25_probability_status": "unavailable",
        "v25_mixture_five_status": "unavailable",
    }


def new(day="2026-11-24", status="pending"):
    record = {
        "sydney_date": day,
        "selection_time_utc": "2026-11-23T14:30:00+00:00",
        "status": status,
        "v25_probability_status": "frozen",
        "v25_mixture_five_status": "frozen",
    }
    if status == "verified":
        record.update(
            v25_score_probability_diagnostics=[{}, {}],
            v25_mixture_five_hit=False,
        )
    return record


def test_old_sets_never_receive_retrospective_frozen_probabilities():
    report = build_monitor(audit([legacy()]))
    assert report["legacy_sets_not_backfilled"] == 1
    assert report["prospective_sets_since_rollout"] == 0
    assert report["status"] == "awaiting_new_frozen_sets"
    assert report["all_eligible_verified_probability_pairs"] == 0
    assert report["check_passed"] is True
    assert report["old_probability_grids_reconstructed"] is False
    assert report["production_model_promoted"] is False


def test_pending_new_probability_is_eligible_but_not_a_loss():
    report = build_monitor(audit([legacy(), new()]))
    assert report["prospective_sets_since_rollout"] == 1
    assert report["new_sets_with_both_valid_grids_and_mixture"] == 1
    assert report["new_pending_pairs"] == 1
    assert report["new_verified_pairs"] == 0
    assert report["status"] == "awaiting_verified_two_leg_results"
    assert report["integrity_issues"] == []


def test_verified_same_pair_counts_after_both_scores():
    report = build_monitor(audit([legacy(), new(status="verified")], probability_pairs=1))
    assert report["new_verified_pairs"] == 1
    assert report["new_pending_pairs"] == 0
    assert report["all_eligible_verified_probability_pairs"] == 1
    assert report["initial_100_pair_evaluation"]["remaining"] == 99
    assert report["initial_100_pair_evaluation"]["goal_reached"] is False
    assert report["status"] == "collecting_verified_probabilities"


@pytest.mark.parametrize("field,value,reason", [
    ("v25_probability_status", "unavailable", "missing_or_invalid_frozen_v25_grids"),
    ("v25_probability_status", "invalid", "missing_or_invalid_frozen_v25_grids"),
    ("v25_mixture_five_status", "invalid", "missing_or_invalid_frozen_mixture_choices"),
])
def test_new_set_missing_prospective_freeze_fails_integrity(field, value, reason):
    packet = new()
    packet[field] = value
    report = build_monitor(audit([packet]))
    assert report["status"] == "integrity_failure"
    assert report["check_passed"] is False
    assert any(item["reason"] == reason for item in report["integrity_issues"])


def test_new_verified_without_genuine_graded_score_diagnostics_fails():
    record = new(status="verified")
    del record["v25_mixture_five_hit"]
    record["v25_score_probability_diagnostics"] = [{}]
    report = build_monitor(audit([record], probability_pairs=1))
    issues = {item["reason"] for item in report["integrity_issues"]}
    assert issues == {"missing_same_leg_score_metrics", "missing_verified_mixture_grade"}


def test_source_invalid_and_duplicate_day_fail_even_if_legacy():
    report = build_monitor(audit([legacy(), legacy()], archived=3, excluded={
        "source_timestamp_mismatch": 1,
    }))
    issues = {item["reason"] for item in report["integrity_issues"]}
    assert "duplicate_or_missing_day" in issues
    assert "invalid_original_source_archives" in issues


def test_prospective_cutoff_must_be_stable_and_timezone_aware():
    assert PROBABILITY_REQUIRED_FROM.endswith("+00:00")
    early = new()
    early["selection_time_utc"] = "2026-11-22T05:17:26+00:00"
    result = build_monitor(audit([early]))
    assert result["legacy_sets_not_backfilled"] == 1
    assert result["prospective_sets_since_rollout"] == 0


def test_invalid_denominator_rejected():
    with pytest.raises(ValueError, match="invalid_source_valid_set_count"):
        build_monitor({"archived_sets_found": 2, "source_valid_sets": 2, "rows": [legacy()]})
    with pytest.raises(ValueError, match="invalid_verified_probability_pair_count"):
        build_monitor(audit([new()], probability_pairs=None))
