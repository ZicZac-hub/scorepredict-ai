"""End-to-end synthetic capture test; never interpreted as real match evidence."""
from copy import deepcopy
from datetime import timedelta
import json

import pytest

from api_football import LEAGUES
from history_pipeline import build_evidence
from test_verified_xg_challenger import evidence, seal
from v10_prospective import utc
import v21_capture_sources as capture
from verified_xg_evidence_pipeline import load_cases, report


def test_scheduled_collector_freezes_real_asof_shape_without_result_or_network(tmp_path, monkeypatch):
    fixture, history, snapshot = evidence()
    # The synthetic fixture must use the actual competition code-to-ID mapping
    # applied by the real capture_pair fixture constructor.
    fixture["competition"]["id"] = LEAGUES["PL"]
    for row in history:
        row["competition"]["id"] = LEAGUES["PL"]
    as_of = utc(snapshot["captured_at_utc"])
    inputs = build_evidence(fixture, history, as_of)
    assert len(inputs["history"]) >= 100
    snapshot = deepcopy(snapshot)
    snapshot["available_sources"] = {"historical_xg": True}
    snapshot = seal(snapshot)
    leg = {
        "provider": "API-Football", "fixture_id": str(fixture["id"]),
        "forecast_id": "synthetic-unique-prematch", "input_digest": "original-v8-digest",
        "home_team_id": str(fixture["homeTeam"]["id"]),
        "away_team_id": str(fixture["awayTeam"]["id"]),
        "home_team": "Home", "away_team": "Away", "competition": "PL",
        "kickoff_utc": fixture["utcDate"],
    }
    monkeypatch.setattr(capture, "current_pair", lambda root, now: [leg])
    monkeypatch.setattr(capture, "targeted_pair_models",
                        lambda **kwargs: [])
    monkeypatch.setattr(capture, "load_evidence", lambda *a, **k: inputs)
    monkeypatch.setattr(capture, "capture", lambda *a, **k: snapshot)
    def no_network(*a, **k):
        pytest.fail("synthetic capture called external provider")
    import requests
    monkeypatch.setattr(requests, "get", no_network)
    provider_root = tmp_path / "packets"
    archive_root = tmp_path / "archive"
    outcome = capture.capture_pair(
        root=provider_root, source_root=tmp_path / "original",
        pair_root=tmp_path / "pairs", archive_root=archive_root,
        enabled=True, api_key="synthetic-no-provider-key",
        now=as_of, clock=lambda: as_of,
    )
    assert outcome["attempted_legs"] == 1
    assert outcome["frozen_shadow_ready"] == 1, outcome
    assert len(outcome["frozen_history_inputs"]) == 1
    records = list((archive_root / "history").glob("*.json"))
    assert len(records) == 1
    frozen = json.loads(records[0].read_text())
    assert frozen["frozen_shadow"]["status"] == "ready"
    assert utc(frozen["captured_at_utc"]) < utc(fixture["utcDate"])

    # Future truth is independently attached only after the original sealed
    # matrices already exist; the frozen record remains byte-identical.
    before = records[0].read_bytes()
    final = {
        "provider": "API-Football", "fixture_id": fixture["id"],
        "home_team_id": fixture["homeTeam"]["id"],
        "away_team_id": fixture["awayTeam"]["id"],
        "kickoff_utc": fixture["utcDate"], "status": "FT",
        "score_basis": "regulation", "home_goals": 1, "away_goals": 0,
        "verified_at_utc": (utc(fixture["utcDate"]) + timedelta(hours=3)).isoformat(),
    }
    results_path = tmp_path / "verified_results.json"
    results_path.write_text(json.dumps({str(fixture["id"]): final}))
    cases, errors = load_cases(root=archive_root, provider_root=provider_root,
                               results_path=results_path)
    assert len(cases) == 1 and not errors
    assert cases[0]["frozen_shadow"]["status"] == "ready"
    state = report(root=archive_root, provider_root=provider_root,
                   results_path=results_path, sportmonks_status=tmp_path / "none.json")
    assert state["frozen_history_records"] == 1
    assert state["independently_settled_candidate_cases"] == 1
    assert state["automatic_promotion"] is False
    assert state["accuracy_improvement_verified"] is False
    assert records[0].read_bytes() == before
