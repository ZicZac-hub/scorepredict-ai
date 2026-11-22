"""V34 regression: no invented hits, no overlap inflation, no post-match picks."""
import copy

import numpy as np
import pytest

from v34_score_error_research import VERSION, classify, score, summarize

PICKS = [
    (("0-0", "0-0")), (("1-1", "1-1")),
    (("2-0", "2-0")), (("0-1", "0-1")), (("1-0", "1-0")),
]
A = ("API-Football", "1")
B = ("API-Football", "2")
C = ("API-Football", "3")
GRID = np.outer([.22, .31, .23, .14, .10],
                [.27, .31, .20, .13, .09])


def packet(fixture_keys, joint):
    return {
        "fixture_keys": [list(x) for x in fixture_keys],
        "five_original_v25_robust_scores": [
            {"score1": a, "score2": b} for a, b in joint
        ],
        "five_original_v8_optimal_scores": [
            {"score1": a, "score2": b} for a, b in joint
        ],
    }


def sample():
    first = packet((A, B), [
        ("0-0", "1-1"), ("1-1", "2-0"), ("2-0", "0-1"),
        ("0-1", "0-0"), ("1-0", "1-0"),
    ])
    # Both actual leg scores appear individually, but never on ONE ticket.
    second = packet((A, C), [
        ("0-0", "0-0"), ("1-1", "4-1"), ("2-0", "2-0"),
        ("0-1", "0-1"), ("1-0", "1-0"),
    ])
    v30 = {
        "overlapping_pairs_are_not_independent_samples": True,
        "invalid_or_tampered_frozen_days_excluded": {},
        "verified_result_conflicts_excluded": 0,
        "days": [{
            "sydney_day": "2035-04-10",
            "original_policy": "FROZEN",
            "candidate_count": 2,
            "candidate_pairs_verified": 2,
            "candidate_pairs_pending": 0,
            "complete_slate": True,
            "candidate_rows": [
                {"fixture_keys": [list(A), list(B)],
                 "actual_scores": ["0-0", "1-1"],
                 "robust_five_hit": True, "v8_baseline_five_hit": True},
                {"fixture_keys": [list(A), list(C)],
                 "actual_scores": ["0-0", "4-1"],
                 "robust_five_hit": False, "v8_baseline_five_hit": False},
            ],
        }],
    }
    return v30, {"2035-04-10": [first, second]}, {
        "2035-04-10": {A: GRID, B: GRID, C: GRID}
    }


def test_all_five_ticket_error_categories():
    picks = [((score(a)), (score(b))) for a, b in PICKS]
    assert classify(picks, (score("1-1"), score("1-1"))) == "hit"
    assert classify(picks, (score("0-0"), score("1-1"))) == "scores_present_but_wrong_joint_ticket"
    assert classify(picks, (score("7-0"), score("1-1"))) == "one_leg_score_absent"
    assert classify(picks, (score("7-0"), score("8-1"))) == "both_leg_scores_absent"
    with pytest.raises(ValueError):
        classify(picks[:4], (score("0-0"), score("0-0")))


def test_verified_pair_overlap_does_not_inflate_independent_fixture_count():
    report = summarize(*sample())
    assert report["version"] == VERSION
    assert report["verified_overlapping_candidate_pair_rows"] == 2
    assert report["unique_verified_physical_fixtures"] == 3
    assert report["original_v8_unique_fixture_score_diagnostics"]["verified_leg_slots"] == 3
    assert report["original_v8_verified_goal_bands"]["high_4_plus"] == 1
    f = report["frozen_ticket_failure_attribution"]["v25_robust_original"]
    assert f["hit"] == 1
    assert f["scores_present_but_wrong_joint_ticket"] == 1
    assert report["initial_review_ready"] is False
    assert report["no_live_model_changes"] is True
    assert report["probability_replays_are_not_real_v25_frozen_forecasts"] is True
    for model in report["POST_HOC_sensitivity_replays_only"].values():
        assert model["verified_leg_slots"] == 3


def test_pending_original_pairs_never_scored_as_losses():
    v30, catalogs, grids = sample()
    v30["days"][0]["candidate_rows"].pop()
    v30["days"][0]["candidate_pairs_verified"] = 1
    v30["days"][0]["candidate_pairs_pending"] = 1
    v30["days"][0]["complete_slate"] = False
    report = summarize(v30, catalogs, grids)
    assert report["verified_overlapping_candidate_pair_rows"] == 1
    assert report["unique_verified_physical_fixtures"] == 2
    assert report["days"][0]["pending_pair_rows"] == 1
    assert report["fully_verified_slate_days"] == 0


def test_tampered_hit_is_rejected():
    v30, catalogs, grids = sample()
    v30["days"][0]["candidate_rows"][1]["robust_five_hit"] = True
    with pytest.raises(ValueError, match="grade_disagrees"):
        summarize(v30, catalogs, grids)


def test_inconsistent_finals_on_overlapping_fixture_are_rejected():
    v30, catalogs, grids = sample()
    v30["days"][0]["candidate_rows"][1]["actual_scores"][0] = "3-1"
    with pytest.raises(ValueError, match="conflicting_verified_scores"):
        summarize(v30, catalogs, grids)


def test_original_catalog_and_missing_grid_rejected():
    v30, catalogs, grids = sample()
    catalogs["2035-04-10"].pop()
    with pytest.raises(ValueError, match="v30_verified_counts_disagree"):
        summarize(v30, catalogs, grids)
    v30, catalogs, grids = sample()
    del grids["2035-04-10"][C]
    with pytest.raises(ValueError, match="score_grid_missing"):
        summarize(v30, catalogs, grids)


def test_no_verified_fixture_does_not_claim_zero_percent_hit_rate():
    v30, catalogs, grids = sample()
    v30["days"][0]["candidate_rows"] = []
    v30["days"][0]["candidate_pairs_verified"] = 0
    v30["days"][0]["candidate_pairs_pending"] = 2
    v30["days"][0]["complete_slate"] = False
    report = summarize(v30, catalogs, grids)
    assert report["unique_verified_physical_fixtures"] == 0
    assert report["original_v8_unique_fixture_score_diagnostics"]["mean_exact_score_log_loss"] is None
    assert report["verified_overlapping_candidate_pair_rows"] == 0


def test_missing_frozen_source_overlap_guardrail_is_fatal():
    v30, catalogs, grids = sample()
    v30["overlapping_pairs_are_not_independent_samples"] = False
    with pytest.raises(ValueError, match="overlap_guardrail"):
        summarize(v30, catalogs, grids)
