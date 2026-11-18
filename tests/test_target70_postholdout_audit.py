"""Read-only probability scoring, paired policies and missing-slate accounting."""
from copy import deepcopy
import json

import pytest

from target70_postholdout_audit import (calibration, policy_comparisons, source_audit,
                                      validate_day_records, validate_source_results, _summary, file_hash)


def pair(*, ids=("1", "2"), hit=False, coverage=.10):
    return {"match_ids": list(ids), "fixtures": [
        {"league": "en.1", "date": "2024-08-16", "home": "A", "away": "B"},
        {"league": "en.1", "date": "2024-08-16", "home": "C", "away": "D"}],
        "fit_month_starts": ["2024-08-01"]*2,
        "last_training_source_dates": ["2024-07-30"]*2,
        "estimated_coverage": coverage,
        "five_doubles": [{"score1": [i//3, 0], "score2": [i % 3, 0],
                          "joint_probability": coverage/5} for i in range(5)],
        "actual_scores": [[0, 0], [0, 0]] if hit else [[9, 9], [9, 9]], "hit": hit}


def test_binary_brier_reliability_and_validation_constant_are_correct():
    report = calibration([(.1, False), (.1, True), (.2, False)], constant_probability=.1)
    assert report["binary_brier"] == pytest.approx((.01+.81+.04)/3)
    assert report["mean_model_coverage"] == pytest.approx(.4/3)
    assert report["observed_hit_rate"] == pytest.approx(1/3)
    assert report["observed_minus_predicted"] == pytest.approx(.2)
    assert sum(b["issued"] for b in report["reliability_bins"]) == 3
    assert report["validation_prevalence_constant_brier"] == pytest.approx((.01+.81+.01)/3)
    assert report["calibration_fitted_or_applied"] is False


def test_boundary_probabilities_are_counted_once_and_empty_is_unknown():
    result = calibration([(0., False), (.10, True), (1., True)])
    assert sum(b["issued"] for b in result["reliability_bins"]) == 3
    assert result["reliability_bins"][-1]["issued"] == 1
    empty = calibration([])
    assert empty["binary_brier"] is None
    assert empty["observed_hit_rate"] is None
    assert empty["wilson_95_descriptive"] is None


@pytest.mark.parametrize("observations", [[(float("nan"), True)], [(-.1, False)],
    [(1.1, False)], [(.2, 1)], [(True, False)]])
def test_invalid_probability_or_unverified_hit_type_is_rejected(observations):
    with pytest.raises(ValueError):
        calibration(observations)


def test_abstentions_remain_opportunities_and_have_no_probability_error():
    days = [{"models": {"m": pair(hit=True)}}, {"models": {"m": None}}]
    result = _summary(days, "m")
    assert result["issued_sets"] == 1
    assert result["source_date_opportunities"] == 2
    assert result["wins_per_source_date_opportunity"] == .5
    assert result["binary_brier"] == pytest.approx(.81)
    strict = _summary(days, "m", .70)
    assert strict["issued_sets"] == 0
    assert strict["observed_hit_rate"] is None
    assert strict["abstained_dates"] == 2


def test_whole_row_hit_cannot_be_replaced_by_separate_leg_membership():
    row = pair(hit=True)
    row["actual_scores"] = [[1, 0], [2, 0]]
    with pytest.raises(ValueError, match="actual_same_row"):
        validate_day_records({"date_records": [{"source_date": "2024-08-16", "models": {"m": row}}]})


def test_saved_cutoff_and_probability_tampering_fail_validation():
    row = pair()
    season = {"date_records": [{"source_date": "2024-08-16", "models": {"m": row}}]}
    validate_day_records(season)
    row["last_training_source_dates"][0] = "2024-07-31"
    with pytest.raises(ValueError, match="training_cutoff"):
        validate_day_records(season)
    row["last_training_source_dates"][0] = "2024-07-30"
    row["estimated_coverage"] += .1
    with pytest.raises(ValueError, match="coverage_disagrees"):
        validate_day_records(season)


def test_actual_results_are_bound_to_pinned_raw_fixture_identity():
    row = pair(hit=True)
    season = {"date_records": [{"source_date": "2024-08-16", "models": {"m": row}}]}
    source = [{"source_match_id": ident, "competition": fixture["league"],
               "source_date": fixture["date"], "home_team": fixture["home"],
               "away_team": fixture["away"], "home_goals": 0, "away_goals": 0}
              for ident, fixture in zip(row["match_ids"], row["fixtures"])]
    validate_source_results(season, source)
    source[0]["home_goals"] = 2
    with pytest.raises(ValueError, match="actual_score_disagrees"):
        validate_source_results(season, source)
    source[0]["home_team"] = "Unrelated team"
    with pytest.raises(ValueError, match="fixture_not_in_pinned"):
        validate_source_results(season, source)


def test_different_policy_pairs_are_not_mislabelled_same_pair_evidence():
    season = {"single_match_metrics": {"a": {}, "b": {}}, "date_records": [
        {"models": {"a": pair(hit=True), "b": pair(ids=("3", "4"), hit=False)}},
        {"models": {"a": pair(hit=False), "b": pair(hit=True)}},
        {"models": {"a": None, "b": pair(hit=False)}}]}
    original = deepcopy(season)
    result = policy_comparisons(season)[0]
    assert result["different_pair_dates_among_joint_issued"] == 1
    assert result["both_policies_issued"]["paired_dates"] == 2
    assert result["same_two_matches_subgroup"]["paired_dates"] == 1
    assert result["same_two_matches_subgroup"]["second_only_wins"] == 1
    assert result["all_source_dates_abstention_counts_as_no_winning_set"]["paired_dates"] == 3
    assert season == original


def test_raw_missing_fixtures_and_dates_are_visible_without_inventing_results(tmp_path):
    raw = {"matches": [
        {"date": "2025-05-23", "score": {}},
        {"date": "2025-05-25", "score": {}},
        {"date": "2025-05-25", "score": {"ft": [1, 0]}}]}
    source = tmp_path/"pinned"/"2024-25"/"es.1.json"
    source.parent.mkdir(parents=True)
    source.write_text(json.dumps(raw))
    manifest = {"source_files": [{"cache_relative_path": "pinned/2024-25/es.1.json",
        "source_url": "https://example.test/pinned/es.1.json", "sha256": file_hash(source),
        "season": "2024-25", "competition": "es.1", "normalized_played_rows": 1,
        "rejected_rows": {"missing_full_time_score": 2}}]}
    original = source.read_bytes()
    report, all_dates, incomplete = source_audit(manifest, tmp_path, "2024-25")
    assert report["holdout_missing_full_time_rows"] == 2
    assert report["normalized_played_rows"] == 1
    assert report["holdout_raw_scheduled_source_dates"] == 2
    assert all_dates == incomplete == {"2025-05-23", "2025-05-25"}
    assert source.read_bytes() == original
    source.write_text(json.dumps({"matches": []}))
    with pytest.raises(ValueError, match="hash_mismatch"):
        source_audit(manifest, tmp_path, "2024-25")
