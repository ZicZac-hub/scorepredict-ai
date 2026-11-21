"""Regression tests: verified exact-score audit never uses post-kickoff information."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json

import pytest

from scoreline_calibration_audit import _events, build_report

KICKOFF = datetime(2035, 1, 1, 14, tzinfo=timezone.utc)
CAPTURE = KICKOFF - timedelta(hours=2)
NOW = KICKOFF + timedelta(hours=5)


def matrix(probability=.12):
    return [
        [1 - .35 - .25 - probability, 0, 0, 0],
        [.35, 0, 0, 0],
        [.25, 0, 0, 0],
        [probability, 0, 0, 0],
    ]


def forecast(fid="1234", family="v8_near", snapshot="fresh", capture=CAPTURE):
    return {
        "provider": "API-Football", "fixture_id": fid, "forecast_id": snapshot,
        "home_team_id": "10", "away_team_id": "20",
        "fixture_name": "Test home vs Test away",
        "mode": "prospective", "score_basis": "regulation",
        "input_digest": "original-immutable-source", "model_key": family,
        "captured_at_utc": capture.isoformat(),
        "evidence_as_of_utc": (capture - timedelta(minutes=5)).isoformat(),
        "kickoff_utc": KICKOFF.isoformat(),
        "score_matrix": matrix(),
    }


def result(fid="1234", goals=(3, 0)):
    return {
        "provider": "API-Football", "fixture_id": fid,
        "home_team_id": "10", "away_team_id": "20",
        "kickoff_utc": KICKOFF.isoformat(),
        "status": "FT", "evidence_version": 2, "score_basis": "regulation",
        "verified_at_utc": (KICKOFF + timedelta(hours=3)).isoformat(),
        "home_goals": goals[0], "away_goals": goals[1],
    }


def save_forecast(root, row):
    target = root / "forecasts" / row["provider"] / row["fixture_id"]
    target.mkdir(parents=True, exist_ok=True)
    (target / (row["forecast_id"] + ".json")).write_text(json.dumps(row))


def save_result(root, row):
    root.mkdir(parents=True, exist_ok=True)
    (root / "verified_results.json").write_text(json.dumps({row["fixture_id"]: row}))


def test_three_nil_is_not_four_plus_goals():
    out = _events(matrix(), (3, 0))
    assert out["exact_3_0"]["predicted_probability"] == pytest.approx(.12)
    assert out["exact_3_0"]["observed"] == 1
    assert out["at_least_four_total_goals"]["observed"] == 0
    assert out["at_least_four_total_goals"]["predicted_probability"] == 0


def test_near_pre_match_source_and_verified_result_are_counted_once(tmp_path):
    root = tmp_path / "v10"
    prematch = tmp_path / "prematch"
    older = forecast(snapshot="older", capture=CAPTURE - timedelta(days=1))
    older["score_matrix"] = matrix(.02)
    newest = forecast()
    save_forecast(root, older)
    save_forecast(root, newest)
    save_result(prematch, result())
    before = (root / "forecasts/API-Football/1234/fresh.json").read_bytes()
    report = build_report(root, prematch, now=NOW)
    assert report["aggregates"]["v8_near"]["verified_fixtures"] == 1
    assert report["aggregates"]["v8"]["verified_fixtures"] == 0
    event = report["aggregates"]["v8_near"]["events"]["exact_3_0"]
    assert event["observed_events"] == 1
    assert event["expected_events"] == pytest.approx(.12)
    assert event["brier"] == pytest.approx((1 - .12) ** 2)
    assert report["production_model_promoted"] is False
    assert (root / "forecasts/API-Football/1234/fresh.json").read_bytes() == before


def test_early_and_near_models_not_counted_as_independent_fixtures(tmp_path):
    root, prematch = tmp_path / "v10", tmp_path / "prematch"
    save_forecast(root, forecast(family="v8_near", snapshot="near"))
    save_forecast(root, forecast(family="v8", snapshot="early"))
    save_result(prematch, result())
    report = build_report(root, prematch, now=NOW)
    assert report["aggregates"]["v8"]["verified_fixtures"] == 1
    assert report["aggregates"]["v8_near"]["verified_fixtures"] == 1
    assert report["model_keys_separate"] == ["v8", "v8_near"]


@pytest.mark.parametrize("alter", ["post_kickoff", "less_than_30_minutes", "future_evidence",
                                    "wrong_team", "wrong_kickoff", "late_verification",
                                    "wrong_filename", "invalid_grid"])
def test_rejects_unqualified_forecast_or_final(tmp_path, alter):
    root, prematch = tmp_path / "v10", tmp_path / "prematch"
    row = forecast()
    final = result()
    if alter == "post_kickoff":
        row["captured_at_utc"] = (KICKOFF + timedelta(minutes=1)).isoformat()
    elif alter == "less_than_30_minutes":
        row["captured_at_utc"] = (KICKOFF - timedelta(minutes=10)).isoformat()
    elif alter == "future_evidence":
        row["evidence_as_of_utc"] = (KICKOFF + timedelta(minutes=1)).isoformat()
    elif alter == "wrong_team":
        final["away_team_id"] = "wrong"
    elif alter == "wrong_kickoff":
        final["kickoff_utc"] = (KICKOFF + timedelta(hours=1)).isoformat()
    elif alter == "late_verification":
        final["verified_at_utc"] = (NOW + timedelta(hours=1)).isoformat()
    elif alter == "wrong_filename":
        row["forecast_id"] = "different"
    elif alter == "invalid_grid":
        row["score_matrix"][0][0] = -1
    save_forecast(root, row)
    if alter == "wrong_filename":
        folder = root / "forecasts/API-Football/1234"
        (folder / "different.json").rename(folder / "fake.json")
    save_result(prematch, final)
    report = build_report(root, prematch, now=NOW)
    assert report["aggregates"]["v8_near"]["verified_fixtures"] == 0
    assert report["aggregates"]["v8_near"]["events"]["exact_3_0"]["observed_rate"] is None


def test_conflicting_verified_sources_exclude_fixture(tmp_path):
    root, prematch = tmp_path / "v10", tmp_path / "prematch"
    save_forecast(root, forecast())
    save_result(root, result(goals=(3, 0)))
    save_result(prematch, result(goals=(1, 0)))
    report = build_report(root, prematch, now=NOW)
    assert report["result_conflicts_excluded"] == 1
    assert report["aggregates"]["v8_near"]["verified_fixtures"] == 0


def test_no_results_has_unavailable_rates_not_false_confidence(tmp_path):
    report = build_report(tmp_path / "none", tmp_path / "prematch", now=NOW)
    assert report["aggregates"]["v8"]["events"]["exact_3_0"]["observed_rate"] is None
    assert report["improvement_claimed"] is False


def test_archived_lazio_case_keeps_real_3_0_as_three_total_goals():
    from pathlib import Path
    repo = Path(__file__).resolve().parents[1]
    source = repo / (
        "data/v10_prospective/forecasts/API-Football/1550202/"
        "6b3f81042ebecce9e6df174ccdb319823ed397dcfbc43277c59ae8c8862dc65e.json"
    )
    record = json.loads(source.read_text(encoding="utf-8"))
    finals = json.loads((repo / "data/prematch/verified_results.json").read_text(encoding="utf-8"))
    final = finals["1550202"]
    assert record["fixture_id"] == "1550202" == str(final["fixture_id"])
    assert record["home_team_id"] == str(final["home_team_id"])
    assert record["away_team_id"] == str(final["away_team_id"])
    assert record["model_key"] == "v8_near"
    assert datetime.fromisoformat(record["captured_at_utc"].replace("Z", "+00:00")) < KICKOFF.replace(year=2026, month=11, day=21)
    actual = (final["home_goals"], final["away_goals"])
    assert actual == (3, 0)
    metric = _events(record["score_matrix"], actual)
    assert metric["exact_3_0"]["predicted_probability"] == pytest.approx(.06955151006996509)
    assert metric["actual_exact_score_rank_min"] == 5
    assert metric["at_least_four_total_goals"]["observed"] == 0
