from datetime import datetime,timedelta,timezone
import json
import v44_collection_health as h

NOW=datetime(2026,11,27,14,tzinfo=timezone.utc)

def write(root,path,obj):
    p=root/path
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(obj))

def test_old_publication_is_not_a_failed_job(tmp_path):
    write(tmp_path,"prospective_ledger/capture_coverage.json",{
        "collector_started_at_utc":(NOW-timedelta(hours=1)).isoformat(),
        "coverage_complete":True})
    write(tmp_path,"prospective_ledger/cadence.json",{
        "last_heavy_at_utc":(NOW-timedelta(hours=8)).isoformat()})
    write(tmp_path,"v44_fresh_history_study/protocol.json",{
        "registered_at_utc":(NOW-timedelta(minutes=20)).isoformat()})
    r=h.evaluate(root=tmp_path,now=NOW)
    assert r["capture"]["status"]=="overdue"
    assert r["heavy_completion"]["status"]=="overdue"
    assert r["coverage_complete"] is True
    assert r["published_capture_started_after_registration"] is False
    assert r["published_diagnostics"]["v44_recovery_reserved"] is None
    assert "recovery_execution_not_confirmed_by_report" in r["actions_to_investigate"]

def test_published_zero_is_not_missing(tmp_path):
    write(tmp_path,"prospective_ledger/capture_coverage.json",{
        "collector_started_at_utc":(NOW-timedelta(minutes=10)).isoformat(),
        "v44_recovery_reserved":0,"v44_pair_refresh_reserved":0})
    write(tmp_path,"prospective_ledger/cadence.json",{
        "last_heavy_at_utc":(NOW-timedelta(hours=3)).isoformat()})
    write(tmp_path,"v44_fresh_history_study/protocol.json",{
        "registered_at_utc":(NOW-timedelta(hours=1)).isoformat()})
    write(tmp_path,"v44_fresh_history_study/readiness.json",{})
    r=h.evaluate(root=tmp_path,now=NOW)
    assert r["capture"]["status"]=="within_grace"
    assert r["published_capture_started_after_registration"] is True
    assert r["published_diagnostics"]["v44_recovery_reserved"]==0
    assert r["actions_to_investigate"]==[]

def test_missing_and_future_timestamps_are_distinguished(tmp_path):
    r=h.evaluate(root=tmp_path,now=NOW)
    assert r["capture"]["status"]=="missing_or_unreadable"
    assert r["published_capture_started_after_registration"] is None
    assert not list(tmp_path.iterdir())
    write(tmp_path,"prospective_ledger/capture_coverage.json",{
        "collector_started_at_utc":(NOW+timedelta(seconds=1)).isoformat(),
        "v44_recovery_reserved":False})
    write(tmp_path,"prospective_ledger/cadence.json",{"last_heavy_at_utc":"2026-11-27T13:00:00"})
    r=h.evaluate(root=tmp_path,now=NOW)
    assert r["capture"]["status"]=="future_timestamp"
    assert r["heavy_completion"]["status"]=="invalid_timestamp"
    assert r["published_diagnostics"]["v44_recovery_reserved"] is None

def test_grace_boundary_and_no_mutations(tmp_path):
    write(tmp_path,"prospective_ledger/capture_coverage.json",{
        "collector_started_at_utc":(NOW-h.CAPTURE_GRACE).isoformat()})
    before={p:p.read_bytes() for p in tmp_path.rglob("*.json")}
    assert h.evaluate(root=tmp_path,now=NOW)["capture"]["status"]=="within_grace"
    assert h.evaluate(root=tmp_path,now=NOW+timedelta(seconds=1))["capture"]["status"]=="overdue"
    assert before=={p:p.read_bytes() for p in tmp_path.rglob("*.json")}
