"""Synthetic regression tests only; never count these as real xG results."""
import json

import pytest

from test_verified_xg_challenger import seal
from test_verified_xg_evidence_pipeline import setup_case
from verified_xg_evidence_pipeline import freeze_history, load_cases
from verified_xg_calibration import evaluate
from verified_xg_prediction_archive import validated_shadow


def test_archived_models_are_frozen_before_result_and_used_without_refitting(tmp_path, monkeypatch):
    fixture, inputs, _, path, root, source, results = setup_case(tmp_path)
    frozen = freeze_history(fixture, inputs, path, root=root, provider_root=source)
    record = json.loads(frozen.read_text())
    assert record["schema"] == 2
    assert record["frozen_shadow"]["status"] == "ready"
    assert validated_shadow(record["frozen_shadow"]) is not None
    cases, exclusions = load_cases(root=root, provider_root=source, results_path=results)
    assert len(cases) == 1 and not exclusions
    # If either refit is attempted after the score is known, the audit fails.
    import verified_xg_calibration as calibration
    def forbidden(*args, **kwargs):
        raise AssertionError("post-match model recalculation is forbidden")
    monkeypatch.setattr(calibration, "fit_forecast", forbidden)
    monkeypatch.setattr(calibration, "predict_shadow", forbidden)
    report = evaluate(cases, cutoff="2025-01-01")
    assert report["holdout_cases"] == 1
    assert report["paired_differences"]["all"]["matched_cases"] == 1


def test_missing_xg_still_freezes_inputs_but_abstains_from_accuracy(tmp_path):
    fixture, inputs, wrapper, path, root, source, results = setup_case(tmp_path)
    snapshot = wrapper["provider_snapshot"]
    snapshot["verified_historical_fixture_xg"] = []
    snapshot.pop("snapshot_sha256", None)
    wrapper["provider_snapshot"] = seal(snapshot)
    path.write_text(json.dumps(wrapper))
    frozen = freeze_history(fixture, inputs, path, root=root, provider_root=source)
    record = json.loads(frozen.read_text())
    assert record["frozen_shadow"] == {
        "status": "abstained", "reason": "no_authenticated_historical_xg"
    }
    assert record["history_input_digest"] == inputs["input_digest"]
    cases, exclusions = load_cases(root=root, provider_root=source, results_path=results)
    assert cases == []
    assert exclusions["pre_kickoff_shadow_abstained_or_invalid"] == 1


def test_tampering_with_frozen_probabilities_invalidates_replay(tmp_path):
    fixture, inputs, _, path, root, source, results = setup_case(tmp_path)
    frozen = freeze_history(fixture, inputs, path, root=root, provider_root=source)
    record = json.loads(frozen.read_text())
    record["frozen_shadow"]["verified_xg_matrix"][0][0] = 0.99
    frozen.write_text(json.dumps(record))
    cases, exclusions = load_cases(root=root, provider_root=source, results_path=results)
    assert not cases
    assert exclusions["invalid_history_or_source_seal"] == 1


def test_archive_rejects_malformed_probabilities():
    bad = {
        "status": "ready",
        "policy": "dc-vs-verified-xg-frozen-probabilities-v1",
        "baseline_model_version": "baseline",
        "xg_model_version": "xg",
        "lineup_status": "unavailable",
        "lineup_numeric_weight": 0.0,
        "dynamic_baseline_matrix": [[0.5, 0.5], [0.0, 0.0]],
        "verified_xg_matrix": [[0.5, float("nan")], [0.5, 0.0]],
    }
    assert validated_shadow(bad) is None
