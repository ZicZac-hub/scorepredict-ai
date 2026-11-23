"""Cautious V25 gate and read-only retrospective goal diagnostics."""
from copy import deepcopy
from pathlib import Path
import json

from v25_upcoming import automatic_quality, best_pair
from v25_goal_validation import historical_reality_check


def candidate_report(mass=0.15, *, age=False, market=True):
    return {"status":"ready",
            "v8_optimal_five_estimated_mass":mass,
            "early_frozen_source":[age,False],
            "source_evidence_audit":[{"verified_pre_match_market":market},
                                     {"verified_pre_match_market":market}]}


def test_auto_quality_requires_both_real_market_legs_and_no_stale_forecast():
    ready=candidate_report()
    before=deepcopy(ready)
    assert automatic_quality(ready)["passed"] is True
    assert automatic_quality(candidate_report(market=False))["passed"] is False
    assert "no_source_matched_fresh_market_on_both_legs" in automatic_quality(
        candidate_report(market=False))["reasons"]
    assert "original_source_over_72_hours_old" in automatic_quality(
        candidate_report(age=True))["reasons"]
    assert "low_model_estimated_five_score_coverage" in automatic_quality(
        candidate_report(mass=.04))["reasons"]
    assert ready==before


def test_old_auto_remains_exploratory_and_empty_catalog_is_not_win():
    result=best_pair([], __import__("datetime").datetime.now(__import__("datetime").timezone.utc),
                     cautious=True)
    assert result["status"] == "no_pairs"


def test_reality_check_does_not_reweight_or_promote_historical_model(tmp_path):
    record={"kind":"retrospective_walk_forward_replay_not_genuine_future_accuracy",
            "paired_v8_v20_same_fixtures":{
                "original_v8_on_eligible_fixtures":{
                    "evaluated":100,"actual_high_score_rate":.32,
                    "predicted_high_score_mass":.31,"exact_hits":14}},
            "same_pair_exact_five":{"methods":{
                "baseline_v8":{"issued":35,"wins":1},
                "negative_binomial_r3":{"issued":35,"wins":2}}}}
    target=tmp_path/"replay.json"
    target.write_text(json.dumps(record))
    out=historical_reality_check(target)
    assert out["status"]=="ready"
    assert out["high_goal_calibration_gap_observed_minus_predicted"]>0
    assert out["retrospective_five_double_wins"] == 1
    assert out["high_score_multiplier_applied"] is False
    assert out["safe_to_claim_new_model_is_better"] is False
    assert json.loads(target.read_text()) == record


def test_invalid_replay_never_claims_accuracy(tmp_path):
    target=tmp_path/"fake.json"
    target.write_text(json.dumps({"kind":"live_prospective","foo":"bar"}))
    assert historical_reality_check(target)["status"]=="unavailable"


def test_abstained_auto_reports_only_unissued_research_candidate(monkeypatch):
    import numpy as np
    import v25_upcoming as mod
    from datetime import datetime,timezone
    matrix=np.full((5,5),1/25)
    item=lambda fid:{"key":("API-Football",str(fid)),"alternatives":{"baseline_v8":matrix}}
    report={"status":"ready","v8_optimal_five_estimated_mass":.1,
            "v25_lower_envelope_criterion":.04,
            "early_frozen_source":[False,False],
            "source_evidence_audit":[{"verified_pre_match_market":False}]*2}
    monkeypatch.setattr(mod,"generate",lambda *args:dict(report))
    result=mod.best_pair([(item(1),item(2))],datetime.now(timezone.utc),cautious=True)
    assert result["status"]=="abstained_low_evidence"
    alt=result["research_alternative"]
    assert alt["auto_issued"] is False
    assert alt["verified_success_probability"] is None
    assert "no_source_matched_fresh_market_on_both_legs" in alt["failed_checks"]
    assert mod.best_pair([(item(1),item(2))],datetime.now(timezone.utc),
                         cautious=False)["status"]=="ready"
