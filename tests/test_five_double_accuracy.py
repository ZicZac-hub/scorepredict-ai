"""Audit evidence integrity and exact Cartesian probability diagnostics."""
from datetime import datetime, timedelta, timezone
from itertools import product
import json

import pytest

from forecast_ledger import append_file_record, read_file_records
from five_double_accuracy import build_report, joint_geometry, wilson


NOW = datetime(2035, 1, 1, 0, tzinfo=timezone.utc)
GRID = [[.35, .20, .10], [.10, .10, .05], [.04, .04, .02]]


def archive(tmp_path):
    for number in (1, 2):
        row = {"mode": "prospective", "provider": "API-Football", "fixture_id": str(number),
               "model_key": "v8", "model_version": "test-v8", "input_digest": "original-input",
               "home_team_id": str(number*2), "away_team_id": str(number*2+1),
               "home_team": f"Home {number}", "away_team": f"Away {number}",
               "fixture_name": f"Home {number} vs Away {number}", "score_matrix": GRID,
               "captured_at_utc": NOW.isoformat(), "evidence_as_of_utc": NOW.isoformat(),
               "kickoff_utc": (NOW+timedelta(hours=2+number)).isoformat()}
        append_file_record(row, root=tmp_path)
    models = read_file_records(root=tmp_path)
    packet = {"mode": "prospective", "status": "frozen_pre_kickoff", "sydney_date": "2035-01-01",
              "captured_at_utc": (NOW+timedelta(minutes=1)).isoformat(),
              "first": models[0], "second": models[1]}
    geometry = joint_geometry(GRID, GRID)
    packet["five_doubles"] = geometry["optimal_five"]
    packet["estimated_coverage"] = geometry["optimal_five_model_coverage"]
    directory = tmp_path/"doubles"
    directory.mkdir()
    file = directory/"2035-01-01.json"
    file.write_text(json.dumps(packet))
    finals = {}
    for model in models:
        finals[model["fixture_id"]] = {k: model[k] for k in (
            "provider", "fixture_id", "home_team_id", "away_team_id", "kickoff_utc")}
        finals[model["fixture_id"]].update({"evidence_version": 2, "score_basis": "regulation",
            "status": "FT", "verified_at_utc": (NOW+timedelta(hours=8)).isoformat(),
            "home_goals": 0, "away_goals": 0})
    return packet, file, finals


def test_geometry_matches_brute_force_entire_rectangular_support():
    left = [[.4, .3], [.2, .1]]
    right = [[.5, .2, .1], [.1, .05, .05]]
    actual = joint_geometry(left, right)
    products = sorted((p*q for row in left for p in row for row2 in right for q in row2), reverse=True)
    cumulative = 0
    for expected, value in enumerate(products, 1):
        cumulative += value
        if cumulative >= .70-1e-12:
            break
    assert actual["optimal_five_model_coverage"] == pytest.approx(sum(products[:5]))
    assert actual["minimum_combinations_for_target_model_coverage"] == expected
    assert actual["full_joint_support_cells"] == 24


@pytest.mark.parametrize("invalid", [[], [[1], [0, 0]], [[float("nan")]], [[-1, 2]], [[True]], [[.3]]])
def test_public_geometry_rejects_invalid_probability_sources(invalid):
    with pytest.raises(ValueError):
        joint_geometry(invalid, GRID)


def test_geometry_work_limit_reports_no_invented_minimum(monkeypatch):
    monkeypatch.setattr("five_double_accuracy.MAX_COVERAGE_CELLS", 5)
    result = joint_geometry([[.25, .25], [.25, .25]], [[.25, .25], [.25, .25]])
    assert result["minimum_combinations_for_target_model_coverage"] is None
    assert result["target_computation_status"] == "work_limit_reached"
    assert result["optimal_five_model_coverage"] == pytest.approx(5/16)


def test_verified_hit_pending_and_model_probability_are_separate(tmp_path):
    packet, file, finals = archive(tmp_path)
    original = file.read_bytes()
    report = build_report(tmp_path, tmp_path/"other", results=finals, now=NOW+timedelta(hours=9))
    assert report["observed"]["verified_sets"] == 1
    assert report["observed"]["hits"] == 1
    assert report["observed"]["hit_rate"] == 1
    assert report["target_70_percent_proven"] is False
    assert report["observed"]["minimum_100_sets_and_lower_interval_at_target"] is False
    assert report["sets"][0]["selected_five_model_coverage"] < .70
    assert file.read_bytes() == original
    finals.pop("2")
    pending = build_report(tmp_path, tmp_path/"other", results=finals, now=NOW+timedelta(hours=9))
    assert pending["observed"]["pending_sets"] == 1
    assert pending["observed"]["verified_sets"] == 0
    assert pending["observed"]["hit_rate"] is None
    assert pending["observed"]["wilson_95"] is None


@pytest.mark.parametrize("field,value", [("provider", "wrong-provider"),
    ("home_team_id", "wrong-team"), ("score_basis", "extra_time"), ("status", "NS"),
    ("home_goals", True), ("verified_at_utc", (NOW-timedelta(hours=1)).isoformat()),
    ("verified_at_utc", (NOW+timedelta(days=2)).isoformat())])
def test_unqualified_results_never_enter_denominator(tmp_path, field, value):
    _, _, finals = archive(tmp_path)
    finals["2"][field] = value
    report = build_report(tmp_path, tmp_path/"other", results=finals, now=NOW+timedelta(hours=9))
    assert report["observed"]["verified_sets"] == 0
    assert report["observed"]["pending_sets"] == 1


def test_actual_score_inside_both_legs_is_not_automatically_double_hit(tmp_path):
    packet, _, finals = archive(tmp_path)
    # Pick a Cartesian combination of displayed leg scores missing as a row.
    covered = [{p[f"score{i}"] for p in packet["five_doubles"]} for i in (1, 2)]
    selected = {(p["score1"], p["score2"]) for p in packet["five_doubles"]}
    absent = next(p for p in product(*covered) if p not in selected)
    for i, score in enumerate(absent, 1):
        finals[str(i)]["home_goals"], finals[str(i)]["away_goals"] = map(int, score.split("-"))
    report = build_report(tmp_path, tmp_path/"other", results=finals, now=NOW+timedelta(hours=9))
    assert report["observed"]["misses"] == 1
    attribution = report["sets"][0]["verified_result_attribution"]
    assert attribution["classification"] == "leg_scores_present_but_not_in_same_combination"


def test_later_source_snapshot_cannot_backfill_original_pair(tmp_path):
    packet, file, finals = archive(tmp_path)
    packet["captured_at_utc"] = (NOW-timedelta(minutes=1)).isoformat()
    file.write_text(json.dumps(packet))
    report = build_report(tmp_path, tmp_path/"other", results=finals, now=NOW+timedelta(hours=9))
    assert report["source_valid_sets"] == 0
    assert report["source_rejections"] == {"source_created_after_pair_selection": 1}


def test_tampered_probability_and_duplicate_combination_are_rejected(tmp_path):
    packet, file, finals = archive(tmp_path)
    packet["five_doubles"][0]["joint_probability"] += .1
    file.write_text(json.dumps(packet))
    rejected = build_report(tmp_path, tmp_path/"other", results=finals, now=NOW+timedelta(hours=9))
    assert rejected["source_rejections"] == {"saved_joint_probability_disagrees_with_source_grid": 1}
    packet["five_doubles"][1] = packet["five_doubles"][0]
    file.write_text(json.dumps(packet))
    rejected = build_report(tmp_path, tmp_path/"other", results=finals, now=NOW+timedelta(hours=9))
    assert rejected["source_rejections"] == {"five_distinct_combinations_required": 1}


def test_conflicting_scores_stay_pending_instead_of_choosing_convenient_result(tmp_path):
    _, _, finals = archive(tmp_path)
    (tmp_path/"verified_results.json").write_text(json.dumps(finals))
    other = tmp_path/"other"
    other.mkdir()
    finals["2"]["home_goals"] = 1
    (other/"verified_results.json").write_text(json.dumps(finals))
    result = build_report(tmp_path, other, now=NOW+timedelta(hours=9))
    assert result["conflicting_result_identities"] == 1
    assert result["observed"]["pending_sets"] == 1


def test_malformed_final_metadata_does_not_abort_other_records(tmp_path):
    _, _, finals = archive(tmp_path)
    finals["2"]["kickoff_utc"] = "invalid"
    (tmp_path/"verified_results.json").write_text(json.dumps(finals))
    result = build_report(tmp_path, tmp_path/"other", now=NOW+timedelta(hours=9))
    assert result["result_source_rejections"] == {"invalid_result_kickoff": 1}
    assert result["observed"]["pending_sets"] == 1


def test_no_data_never_means_seventy_percent(tmp_path):
    report = build_report(tmp_path, tmp_path, now=NOW)
    assert report["source_valid_sets"] == 0
    assert report["observed"]["hit_rate"] is None
    assert report["diagnosis"]["model_coverage_range"] is None
    assert report["target_70_percent_proven"] is False
    assert wilson(0, 1)[1] == pytest.approx(.7934506856)
