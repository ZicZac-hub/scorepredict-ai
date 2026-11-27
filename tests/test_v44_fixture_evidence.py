import json
from datetime import datetime, timezone

import v44_fixture_evidence as audit


def _write(root, forecast_id, captured):
    path = root / "forecasts" / "p" / "1" / (forecast_id + ".json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "model_key": "v8", "forecast_id": forecast_id, "input_digest": "test",
        "provider": "p", "fixture_id": "1", "home_team_id": "a", "away_team_id": "b",
        "home_team": "A", "away_team": "B", "captured_at_utc": captured,
        "kickoff_utc": "2025-01-01T12:00:00Z"}))


def test_counts_matched_fixture_once_and_uses_earliest_capture(tmp_path, monkeypatch):
    _write(tmp_path, "first", "2025-01-01T06:00:00Z")
    _write(tmp_path, "later", "2025-01-01T10:00:00Z")
    monkeypatch.setattr(audit, "_forecast", lambda row: row)
    monkeypatch.setattr(audit.sources, "_components", lambda *a, **k: {"xg": {}, "market": None, "lineup": None})
    final = {"provider":"p", "fixture_id":"1", "home_team_id":"a", "away_team_id":"b",
             "kickoff_utc":"2025-01-01T12:00:00Z", "verified_at_utc":"2025-01-01T15:00:00Z",
             "home_goals":2, "away_goals":1, "score_basis":"regulation",
             "evidence_version":2, "status":"FT"}
    (tmp_path / "verified_results.json").write_text(json.dumps({"p:1": final}))
    report = audit.evaluate(source_root=tmp_path, prematch_root=tmp_path/"other",
                            now=datetime(2025,1,2,tzinfo=timezone.utc))
    assert report["matched_verified_fixtures"] == 1
    assert report["rows"][0]["forecast_id"] == "first"
    assert report["exclusions"]["duplicate_fixture_capture_or_event"] == 1
    assert report["component_availability_counts"]["xg"] == 1
    assert report["double_pairs_created"] == 0


def test_missing_final_not_counted_as_verified(tmp_path, monkeypatch):
    _write(tmp_path, "first", "2025-01-01T06:00:00Z")
    monkeypatch.setattr(audit, "_forecast", lambda row: row)
    report = audit.evaluate(source_root=tmp_path, prematch_root=tmp_path/"other",
                            now=datetime(2025,1,2,tzinfo=timezone.utc))
    assert report["past_without_verified_final"] == 1
    assert report["matched_verified_fixtures"] == 0
