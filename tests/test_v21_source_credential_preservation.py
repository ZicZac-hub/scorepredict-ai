"""Regression: successful history archival must not clobber the next API credential."""
import json

import pytest

import v21_capture_sources_v2 as collector
import verified_xg_evidence_pipeline as pipeline
from test_v21_matched_refresh import NOW, fixture_sources, packet


@pytest.mark.parametrize("shadow_status", ["ready", "abstained"])
def test_two_legs_keep_provider_credential_after_history_archive(tmp_path, monkeypatch, shadow_status):
    source, pairs, legs = fixture_sources(tmp_path)
    root = tmp_path / "packets"
    archive = tmp_path / "archive"
    archive.mkdir()
    receipt = archive / "synthetic.json"
    receipt.write_text(json.dumps({"frozen_shadow": {"status": shadow_status}}))
    calls = []
    sentinel = "synthetic-test-credential"

    def history(fixture, **kwargs):
        calls.append(("history", str(fixture["id"]), kwargs["api_key"]))
        return {"history": [], "input_digest": "synthetic", "as_of_utc": NOW.isoformat()}

    def capture(fixture, history, key, **kwargs):
        calls.append(("capture", str(fixture["id"]), key))
        leg = next(row for row in legs if row["fixture_id"] == str(fixture["id"]))
        return packet(leg, captured=NOW)

    monkeypatch.setattr(collector, "current_pair", lambda *a, **k: list(legs))
    monkeypatch.setattr(collector, "load_evidence", history)
    monkeypatch.setattr(collector, "capture", capture)
    monkeypatch.setattr(pipeline, "freeze_history", lambda *a, **k: receipt)
    monkeypatch.setenv("V21_DAILY_CAPTURE_ATTEMPTS", "6")
    status = collector.capture_pair(root=root, source_root=source, pair_root=pairs,
        archive_root=archive, enabled=True, api_key=sentinel, now=NOW, clock=lambda: NOW)
    assert status["new_packets"] == [leg["fixture_id"] for leg in legs]
    assert status["frozen_shadow_" + shadow_status] == 2
    assert status["attempted_legs"] == status["daily_attempts_used"] == 2
    assert calls == [(stage, leg["fixture_id"], sentinel)
                     for leg in legs for stage in ("history", "capture")]
    assert sentinel not in (root / "status.json").read_text()


def test_common_pair_deadline_stops_late_second_leg_attempt(tmp_path, monkeypatch):
    from datetime import timedelta
    source, pairs, legs = fixture_sources(tmp_path)
    root = tmp_path / "packets"
    current = [NOW]
    seen = []
    def history(fixture, **kwargs):
        return {"history": [], "input_digest": "synthetic"}
    def capture(fixture, history, key, **kwargs):
        leg = next(row for row in legs if row["fixture_id"] == str(fixture["id"]))
        seen.append(leg["fixture_id"])
        current[0] = NOW + timedelta(minutes=53)
        return packet(leg, captured=current[0])
    monkeypatch.setattr(collector, "current_pair", lambda *a, **k: list(legs))
    monkeypatch.setattr(collector, "load_evidence", history)
    monkeypatch.setattr(collector, "capture", capture)
    monkeypatch.setenv("V21_DAILY_CAPTURE_ATTEMPTS", "6")
    status = collector.capture_pair(root=root, source_root=source, pair_root=pairs,
        enabled=True, api_key="synthetic", now=NOW, clock=lambda: current[0])
    assert seen == [legs[0]["fixture_id"]]
    assert status["attempted_legs"] == status["daily_attempts_used"] == 1
    assert legs[1]["fixture_id"] + ":common_pair_window_closed" in status["skipped"]
