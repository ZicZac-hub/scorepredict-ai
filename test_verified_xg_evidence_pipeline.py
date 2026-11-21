"""Synthetic integrity tests: these are NOT real xG accuracy evidence."""
from datetime import timedelta
import json

import pytest

from history_pipeline import build_evidence
from test_verified_xg_challenger import evidence, seal
from v10_prospective import utc
from verified_xg_evidence_pipeline import (
    audit_packets, freeze_history, load_cases, report,
)


def setup_case(tmp_path):
    fixture, history, packet = evidence()
    sealed = seal(packet)
    as_of = sealed["captured_at_utc"]
    inputs = build_evidence(fixture, history, as_of)
    root, provider_root = tmp_path / "archive", tmp_path / "source"
    provider_root.mkdir()
    path = provider_root / "packet.json"
    wrapper = {
        "source_forecast": {
            "provider": "API-Football",
            "fixture_id": str(fixture["id"]),
            "home_team_id": str(fixture["homeTeam"]["id"]),
            "away_team_id": str(fixture["awayTeam"]["id"]),
            "kickoff_utc": fixture["utcDate"],
        },
        "provider_snapshot": sealed,
        "supplemental_history_digest": inputs["input_digest"],
    }
    path.write_text(json.dumps(wrapper))
    result = {
        "provider": "API-Football", "fixture_id": fixture["id"],
        "home_team_id": fixture["homeTeam"]["id"],
        "away_team_id": fixture["awayTeam"]["id"],
        "kickoff_utc": fixture["utcDate"], "status": "FT",
        "score_basis": "regulation", "home_goals": 1, "away_goals": 0,
        "verified_at_utc": (utc(fixture["utcDate"]) + timedelta(hours=3)).isoformat(),
    }
    results = tmp_path / "verified_results.json"
    results.write_text(json.dumps({str(fixture["id"]): result}))
    return fixture, inputs, wrapper, path, root, provider_root, results


def test_frozen_inputs_join_only_authenticated_independent_results(tmp_path):
    fixture, inputs, wrapper, path, root, source, results = setup_case(tmp_path)
    written = freeze_history(fixture, inputs, path, root=root, provider_root=source)
    assert written.exists()
    assert freeze_history(fixture, inputs, path, root=root, provider_root=source) == written
    cases, excluded = load_cases(root=root, provider_root=source, results_path=results)
    assert len(cases) == 1
    assert cases[0]["actual"] == [1, 0]
    assert not excluded
    status = report(root=root, provider_root=source, results_path=results)
    assert status["authenticated_archive"]["authenticated_packets"] == 1
    assert status["independently_settled_candidate_cases"] == 1
    assert status["status"] == "insufficient_matched_authenticated_evidence"
    assert status["accuracy_improvement_verified"] is False
    assert status["automatic_promotion"] is False


def test_tampered_history_and_snapshot_are_rejected(tmp_path):
    fixture, inputs, wrapper, path, root, source, results = setup_case(tmp_path)
    frozen = freeze_history(fixture, inputs, path, root=root, provider_root=source)
    changed = json.loads(frozen.read_text())
    changed["history"][0]["score"]["fullTime"]["home"] = 99
    frozen.write_text(json.dumps(changed))
    cases, excluded = load_cases(root=root, provider_root=source, results_path=results)
    assert not cases
    assert excluded["invalid_history_or_source_seal"] == 1
    # Restore a clean record, then corrupt the independently sealed source.
    frozen.unlink()
    freeze_history(fixture, inputs, path, root=root, provider_root=source)
    wrapper["provider_snapshot"]["verified_historical_fixture_xg"][0]["home_xg"] = 14
    path.write_text(json.dumps(wrapper))
    cases, excluded = load_cases(root=root, provider_root=source, results_path=results)
    assert not cases
    assert excluded["invalid_history_or_source_seal"] == 1


def test_postkickoff_and_changed_results_fail_closed(tmp_path):
    fixture, inputs, wrapper, path, root, source, results = setup_case(tmp_path)
    freeze_history(fixture, inputs, path, root=root, provider_root=source)
    altered = json.loads(results.read_text())
    altered[str(fixture["id"])]["home_team_id"] = 999
    results.write_text(json.dumps(altered))
    cases, excluded = load_cases(root=root, provider_root=source, results_path=results)
    assert not cases
    assert excluded["missing_verified_final"] == 1
    wrapper["provider_snapshot"]["captured_at_utc"] = fixture["utcDate"]
    path.write_text(json.dumps(wrapper))
    with pytest.raises(ValueError, match="unverified_or_mismatched_capture"):
        freeze_history(fixture, inputs, path, root=root, provider_root=source)


def test_history_frozen_after_source_tamper_cannot_be_rewritten(tmp_path):
    fixture, inputs, wrapper, path, root, source, results = setup_case(tmp_path)
    destination = freeze_history(fixture, inputs, path, root=root, provider_root=source)
    altered = json.loads(destination.read_text())
    altered["history_input_digest"] = "fabricated"
    destination.write_text(json.dumps(altered))
    with pytest.raises(ValueError, match="immutable_history_conflict"):
        freeze_history(fixture, inputs, path, root=root, provider_root=source)


def test_empty_archive_does_not_claim_an_accuracy_rate(tmp_path):
    root, src = tmp_path / "empty", tmp_path / "src"
    results = tmp_path / "none.json"
    assert audit_packets(provider_root=src) == {"unique_fixtures_with_authenticated_packets": 0}
    summary = report(root=root, provider_root=src, results_path=results)
    assert summary["comparison"]["holdout_cases"] == 0
    assert summary["comparison"]["model_metrics"]["verified_xg"]["top1_hit_rate"] is None
    assert summary["comparison"]["five_double"]["verified_xg"]["hit_rate"] is None
    assert not summary["evidence_sufficient"]
