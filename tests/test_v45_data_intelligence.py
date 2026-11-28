from datetime import datetime, timedelta, timezone

import pytest

import v45_data_intelligence as v45


NOW = datetime(2026, 11, 28, 10, tzinfo=timezone.utc)
MODEL = {"captured_at_utc": NOW - timedelta(hours=1),
         "kickoff_utc": NOW + timedelta(minutes=30)}


def test_quality_uses_fixed_weights_and_only_cross_checked_lineups(monkeypatch):
    monkeypatch.setattr(v45, "history_status", lambda *args: "ready")
    monkeypatch.setattr(v45, "_components", lambda *args, **kwargs: {
        "xg": {"reference": {"captured_at_utc": NOW.isoformat()}}, "market": None})
    monkeypatch.setattr(v45, "status_for", lambda *args, **kwargs: {"status": "confirmed"})
    report = v45.fixture_quality(MODEL, cutoff=NOW)
    assert report["score"] == 60
    assert report["statuses"]["lineup"] == "confirmed"
    assert report["points"]["lineup"] == 0
    monkeypatch.setattr(v45, "status_for", lambda *args, **kwargs: {"status": "verified"})
    assert v45.fixture_quality(MODEL, cutoff=NOW)["score"] == 80


def test_lineup_waiting_and_post_kickoff_cannot_score(monkeypatch):
    monkeypatch.setattr(v45, "history_status", lambda *args: "missing_exact_history_receipt")
    monkeypatch.setattr(v45, "_components", lambda *args, **kwargs: {"xg": None, "market": None})
    monkeypatch.setattr(v45, "status_for", lambda *args, **kwargs: {"status": "missing"})
    earlier = NOW - timedelta(minutes=45)
    assert v45.fixture_quality(MODEL, cutoff=earlier)["statuses"]["lineup"] == "window_not_open"
    assert v45.fixture_quality(MODEL, cutoff=NOW)["statuses"]["lineup"] == "missing"
    with pytest.raises(ValueError, match="before kickoff"):
        v45.fixture_quality(MODEL, cutoff=MODEL["kickoff_utc"])
