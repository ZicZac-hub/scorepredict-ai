"""Regression: generated tracking summaries are not immutable forecast evidence."""
import pytest
import reconcile_capture_conflicts as reconciliation


def test_concurrent_tracking_report_prefers_published_upstream(monkeypatch):
    calls = []

    def fake_git(*args):
        calls.append(args)
        if args == ("diff", "--name-only", "--diff-filter=U"):
            return "data/tracking/status.json\n"
        return ""

    monkeypatch.setattr(reconciliation, "_git", fake_git)
    reconciliation.reconcile()

    assert calls == [
        ("diff", "--name-only", "--diff-filter=U"),
        ("checkout", "--ours", "--", "data/tracking/status.json"),
        ("add", "--", "data/tracking/status.json"),
    ]


def test_original_forecast_conflict_still_stops_publication(monkeypatch):
    calls = []

    def fake_git(*args):
        calls.append(args)
        if args == ("diff", "--name-only", "--diff-filter=U"):
            return "data/v10_prospective/forecasts/immutable.json\n"
        raise AssertionError("Original forecast should not be modified")

    monkeypatch.setattr(reconciliation, "_git", fake_git)
    with pytest.raises(RuntimeError, match="Immutable forecast conflicts need manual review"):
        reconciliation.reconcile()
    assert calls == [("diff", "--name-only", "--diff-filter=U")]
