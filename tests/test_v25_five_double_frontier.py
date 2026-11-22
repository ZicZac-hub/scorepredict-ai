"""Exact five-double joint-score coverage frontier and no-hindsight tests."""
from datetime import datetime, timedelta, timezone
import json

import pytest

from five_double_accuracy import joint_geometry
from v25_five_double_frontier import frontier, build_frontier_report


def test_uniform_score_grid_needs_far_more_than_five_combinations():
    grid = [[1 / 25 for _ in range(5)] for _ in range(5)]
    out = frontier(grid, grid)
    assert out["model_max_five_probability_mass"] == pytest.approx(5 / 625)
    assert out["minimum_joint_combinations_for_target_model_mass"] == 438
    assert out["checkpoints_model_mass"]["100"] == pytest.approx(100 / 625)
    assert out["target_met_by_five_model_choices"] is False
    assert out["not_a_verified_hit_rate"] is True


def test_nearly_certain_model_can_exceed_target_but_not_prove_accuracy():
    grid = [[0.0] * 5 for _ in range(5)]
    grid[0][0] = 1.0
    out = frontier(grid, grid)
    assert out["model_max_five_probability_mass"] == pytest.approx(1.0)
    assert out["minimum_joint_combinations_for_target_model_mass"] == 1
    assert out["target_met_by_five_model_choices"] is True
    assert out["higher_probability_model_does_not_prove_better_accuracy"] is True


def test_five_double_frontier_matches_exact_original_geometry():
    a = [[.30, .25, .1, 0, 0], [.2, .1, 0, 0, 0],
         [.05, 0, 0, 0, 0], [0] * 5, [0] * 5]
    b = [[.1, .25, .15, 0, 0], [.2, .1, 0, 0, 0],
         [.2, 0, 0, 0, 0], [0] * 5, [0] * 5]
    assert sum(map(sum, a)) == pytest.approx(1)
    assert sum(map(sum, b)) == pytest.approx(1)
    result = frontier(a, b)
    original = joint_geometry(a, b)
    assert result["model_max_five_probability_mass"] == pytest.approx(
        original["optimal_five_model_coverage"]
    )
    assert result["minimum_joint_combinations_for_target_model_mass"] == (
        original["minimum_combinations_for_target_model_coverage"]
    )


@pytest.mark.parametrize("target", [0.0, 1.01, -0.2, float("nan"), float("inf")])
def test_bad_targets_never_produce_fake_70pct_model_coverage(target):
    grid = [[1 / 25 for _ in range(5)] for _ in range(5)]
    with pytest.raises(ValueError, match="invalid_target"):
        frontier(grid, grid, target)


def test_work_limit_reports_unknown_not_invented_target():
    grid = [[1 / 25 for _ in range(5)] for _ in range(5)]
    result = frontier(grid, grid, max_work=10, checkpoints=(5,))
    assert result["minimum_joint_combinations_for_target_model_mass"] is None
    assert result["target_count_status"] == "not_reached_within_work_limit"


def test_identical_verified_pair_fixture_does_not_rewrite_saved_picks(tmp_path):
    from v8_v25_matched_audit import _selection_mass
    now = datetime(2035, 1, 1, 19, tzinfo=timezone.utc)
    kickoff = now - timedelta(hours=6)
    selection = kickoff - timedelta(hours=2)
    capture = selection - timedelta(hours=1)
    root, prematch, sets = (tmp_path / sub for sub in ("v8", "finals", "sets"))
    grid = [[.30, .25, .1, 0, 0], [.2, .1, 0, 0, 0],
            [.05, 0, 0, 0, 0], [0] * 5, [0] * 5]
    original = []
    for fid, h, a in (("101", "1", "2"), ("102", "3", "4")):
        row = {
            "provider": "API-Football", "fixture_id": fid,
            "forecast_id": "fixed" + fid, "home_team_id": h,
            "away_team_id": a, "model_key": "v8", "model_version": "v8",
            "input_digest": "known-input-digest", "history_matches": 50,
            "mode": "prospective", "score_basis": "regulation",
            "kickoff_utc": kickoff.isoformat(),
            "captured_at_utc": capture.isoformat(),
            "evidence_as_of_utc": capture.isoformat(),
            "score_matrix": grid,
        }
        path = root / "forecasts" / row["provider"] / fid
        path.mkdir(parents=True)
        (path / (row["forecast_id"] + ".json")).write_text(json.dumps(row))
        original.append(row)
    options = joint_geometry(grid, grid)["optimal_five"]
    baseline = [{"score1": a["score1"], "score2": a["score2"]}
                for a in options]
    choices = [(tuple(int(z) for z in p["score1"].split("-")),
                tuple(int(z) for z in p["score2"].split("-"))) for p in baseline]
    mass = _selection_mass(choices, grid, grid)
    frozen = {
        "decision": "FROZEN_RESEARCH_ONLY",
        "model_version": "pro-analyst-v25-robust-sensitivity-shadow",
        "lower_envelope_not_a_calibrated_probability": True,
        "sydney_date": kickoff.astimezone(
            __import__("zoneinfo").ZoneInfo("Australia/Sydney")).date().isoformat(),
        "selected_at_utc": selection.isoformat(),
        "first": {k: original[0][k] for k in (
            "provider", "fixture_id", "forecast_id", "home_team_id",
            "away_team_id", "model_key", "model_version", "input_digest",
            "kickoff_utc", "captured_at_utc")},
        "second": {k: original[1][k] for k in (
            "provider", "fixture_id", "forecast_id", "home_team_id",
            "away_team_id", "model_key", "model_version", "input_digest",
            "kickoff_utc", "captured_at_utc")},
        "baseline_five": baseline, "five_doubles": baseline,
        "v8_optimal_five_model_mass": mass,
        "robust_five_v8_model_mass": mass,
    }
    sets.mkdir(parents=True)
    saved = sets / (frozen["sydney_date"] + ".json")
    saved.write_text(json.dumps(frozen))
    original_bytes = saved.read_bytes()
    report = build_frontier_report(root=root, prematch=prematch, sets=sets, now=now)
    assert report["original_archived_sets"] == 1
    assert report["pending_original_pairs"] == 1
    assert report["separately_frozen_v25_probability_pairs"] == 0
    assert report["real_70pct_five_double_hit_rate_proven"] is False
    assert report["rows"][0]["original_v8"]["model_max_five_probability_mass"] == pytest.approx(mass)
    assert saved.read_bytes() == original_bytes
