from datetime import datetime, timezone
import v44_late_evidence_audit as audit


def test_select_preserves_original_reference_and_rejects_future(monkeypatch, tmp_path):
    cutoff = datetime(2025,1,1,12,tzinfo=timezone.utc)
    models = [{"forecast_id":"old"}, {"forecast_id":"new"}, {"forecast_id":"future"}]
    def components(model, **kwargs):
        stamps = {"old":"2025-01-01T10:00:00Z",
                  "new":"2025-01-01T11:00:00Z",
                  "future":"2025-01-01T13:00:00Z"}
        key = model["forecast_id"]
        return {"market":{"reference":{"captured_at_utc":stamps[key],
                                       "snapshot_sha256":key},"value":{}}}
    monkeypatch.setattr(audit.sources,"_components",components)
    refs, errors = audit._select(models,tmp_path,cutoff)
    assert refs["market"]["source_forecast_id"] == "new"
    assert refs["xg"] is None
    assert errors == 1


def test_fixed_cutoff_does_not_use_later_forecast(monkeypatch, tmp_path):
    monkeypatch.setattr(audit.inventory,"evaluate",lambda **kw: {
        "matched_verified_fixtures":1,
        "rows":[{"kickoff_utc":"2025-01-01T12:00:00Z",
                 "decision_utc":"2025-01-01T11:50:00Z"}]})
    report = audit.evaluate(source_root=tmp_path,prematch_root=tmp_path,
                            provider_root=tmp_path)
    assert report["gaps"]["no_original_capture_before_fixed_cutoff"] == 1
    assert not report["rows"]
    assert report["forecasts_created"] == 0


def test_missing_components_not_invented(monkeypatch,tmp_path):
    monkeypatch.setattr(audit.sources,"_components",lambda *a,**kw: {})
    refs, errors = audit._select([{"forecast_id":"one"}],tmp_path,
                                datetime(2025,1,1,tzinfo=timezone.utc))
    assert refs == {"market":None,"lineup":None,"xg":None}
    assert errors == 0
