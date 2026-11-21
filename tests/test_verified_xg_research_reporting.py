"""Synthetic tests for descriptive paired xG reporting; no skill claim."""
import json

from test_verified_xg_challenger import evidence, seal
from verified_xg_calibration import evaluate
from verified_xg_evidence_pipeline import collection_health, report


def test_empty_holdout_has_null_deltas_and_no_promotion(tmp_path):
    evaluation = evaluate([], cutoff="2026-12-15")
    paired = evaluation["paired_differences"]
    assert paired["all"]["matched_cases"] == 0
    assert paired["all"]["exact_score_top1_hit_rate_delta_xg_minus_dc"] is None
    assert paired["all"]["score_log_loss_delta_xg_minus_dc"] is None
    assert paired["five_double"]["matched_pairs"] == 0
    assert paired["five_double"]["hit_rate_delta_xg_minus_dc"] is None
    status = report(root=tmp_path / "frozen",
                    provider_root=tmp_path / "provider",
                    results_path=tmp_path / "no-results.json",
                    sportmonks_status=tmp_path / "no-sportmonks.json")
    assert status["evidence_shortfall"] == {
        "training_cases_needed": 30,
        "holdout_cases_needed": 50,
        "paired_days_needed": 20,
    }
    assert status["live_prediction_model_changed"] is False
    assert status["automatic_promotion"] is False
    assert status["accuracy_improvement_verified"] is False


def test_paired_lineup_strata_remain_descriptive():
    fixture, history, snapshot = evidence()
    case = {"fixture": fixture, "history": history, "snapshot": seal(snapshot),
            "actual": [1, 0]}
    rows = [case]
    # A second synthetic fixture verifies the alternate pre-kickoff XI stratum.
    import copy
    both = copy.deepcopy(case)
    both["fixture"]["id"] = 9002
    both["snapshot"]["fixture_id"] = "9002"
    both["snapshot"]["confirmed_starting_xi"] = [
        {"team_id": str(team), "provider_reported": True,
         "starting_xi": [f"{team}-player-{number}" for number in range(11)]}
        for team in (1, 2)]
    both["snapshot"].pop("snapshot_sha256")
    both["snapshot"] = seal(both["snapshot"])
    rows.append(both)
    r = evaluate(rows, cutoff="2025-01-01")
    paired = r["paired_differences"]
    assert r["holdout_cases"] == 2
    assert paired["all"]["matched_cases"] == 2
    assert paired["by_lineup_availability"]["unavailable"]["matched_cases"] == 1
    assert paired["by_lineup_availability"]["both_confirmed"]["matched_cases"] == 1
    assert paired["all"]["score_log_loss_delta_xg_minus_dc"] is not None


def test_collector_status_is_strictly_whitelisted(tmp_path):
    provider = tmp_path / "provider"
    provider.mkdir()
    (provider / "status.json").write_text(json.dumps({
        "reason": "no_new_source_evidence", "eligible_legs": 69,
        "API_FOOTBALL_KEY": "DO_NOT_LEAK", "provider_raw_response": "PRIVATE"
    }))
    sport = tmp_path / "sport.json"
    sport.write_text(json.dumps({
        "reason": "daily_request_budget_exhausted",
        "daily_requests_used": 4, "daily_request_limit": 4,
        "last_provider_observation": {
            "reason": "provider_access_or_subscription",
            "raw_response": "PRIVATE"
        },
        "SPORTMONKS_API_TOKEN": "DO_NOT_LEAK"
    }))
    out = collection_health(provider_root=provider, sportmonks_status=sport)
    assert out["api_football"]["eligible_legs"] == 69
    assert out["sportmonks_last_provider_reason"] == "provider_access_or_subscription"
    rendered = json.dumps(out)
    assert "DO_NOT_LEAK" not in rendered
    assert "PRIVATE" not in rendered
