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
