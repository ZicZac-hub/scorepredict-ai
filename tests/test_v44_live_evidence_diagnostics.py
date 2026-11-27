from datetime import datetime,timedelta,timezone
import json
from pathlib import Path
import v44_live_evidence_diagnostics as d

NOW=datetime(2025,2,1,10,tzinfo=timezone.utc)
MODEL={"provider":"API-Football","fixture_id":"1","home_team_id":"10","away_team_id":"20",
       "kickoff_utc":(NOW+timedelta(hours=2)).isoformat(),"input_digest":"original"}

def receipt(root,when=NOW):
    event=["API-Football","1",MODEL["kickoff_utc"],"10","20"]
    path=root/"receipts"/(d.early.receipt_key(event,"original")+".json")
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps({"captured_at_utc":when.isoformat()}))
    return path

def test_missing_and_late_receipts(tmp_path):
    assert d.history_status(MODEL,NOW,tmp_path)=="missing_exact_history_receipt"
    path=receipt(tmp_path,NOW+timedelta(seconds=1))
    assert d.history_status(MODEL,NOW,tmp_path)=="receipt_after_pair_cutoff"
    assert path.is_file()

def test_invalid_and_mismatched_receipts(tmp_path,monkeypatch):
    path=receipt(tmp_path)
    def invalid(*a,**k):
        raise ValueError("invalid receipt")
    monkeypatch.setattr(d.early,"load",invalid)
    assert d.history_status(MODEL,NOW,tmp_path)=="invalid_history_receipt"
    monkeypatch.setattr(d.early,"load",lambda *a,**k:{})
    monkeypatch.setattr(d,"qualifies",lambda *a:False)
    assert d.history_status(MODEL,NOW,tmp_path)=="receipt_does_not_match_original_forecast"

def test_qualified_history_readiness(tmp_path,monkeypatch):
    receipt(tmp_path)
    monkeypatch.setattr(d.early,"load",lambda *a,**k:{"fixture":{},"history":[],"as_of_utc":NOW.isoformat()})
    monkeypatch.setattr(d,"qualifies",lambda *a:True)
    monkeypatch.setattr(d.ratings,"predict",lambda *a:{"status":"abstained"})
    assert d.history_status(MODEL,NOW,tmp_path)=="insufficient_team_or_league_history"
    monkeypatch.setattr(d.ratings,"predict",lambda *a:{"status":"ready"})
    assert d.history_status(MODEL,NOW,tmp_path)=="ready"

def test_no_protocol_does_not_create_files(tmp_path):
    report=d.evaluate(root=tmp_path/"trial",history_root=tmp_path/"history",
                      pair_root=tmp_path/"pairs",now=NOW)
    assert not report["registered"]
    assert list(tmp_path.iterdir())==[]


def test_future_original_pair_preflight_counts_missing_legs(tmp_path,monkeypatch):
    root=tmp_path/"trial"; root.mkdir()
    (root/"protocol.json").write_text("{}")
    pair_root=tmp_path/"pairs"; (pair_root/"sets").mkdir(parents=True)
    (pair_root/"sets"/"2025-02-01.json").write_text("{}")
    later=NOW+timedelta(hours=5)
    models=[{**MODEL,"kickoff_utc":later.isoformat()},
            {**MODEL,"fixture_id":"2","kickoff_utc":later.isoformat()}]
    monkeypatch.setattr(d.trial,"_protocol",lambda *a:{"registered_at_utc":NOW.isoformat()})
    monkeypatch.setattr(d.sources,"_pair",lambda *a,**k:({"predicted_at_utc":NOW.isoformat()},models))
    monkeypatch.setattr(d,"history_status",lambda m,*a:"ready" if m["fixture_id"]=="1" else "missing_exact_history_receipt")
    result=d.evaluate(root=root,pair_root=pair_root,history_root=tmp_path/"history",now=NOW)
    assert result["counts"]["future_pair_opportunities"]==1
    assert result["counts"]["future_one_history_ready"]==1
    assert result["counts"]["future_with_trial_file"]==0
    assert result["leg_statuses"]["future:missing_exact_history_receipt"]==1
    assert not (root/"trials").exists()
