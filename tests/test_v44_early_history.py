from datetime import datetime,timedelta,timezone
import json
import pytest

from history_pipeline import build_evidence
import v44_early_history as early
from v44_early_history_audit import qualifies

NOW=datetime(2025,2,1,10,tzinfo=timezone.utc)
FIXTURE={"provider":"API-Football","id":999,"utcDate":"2025-02-01T12:00:00Z",
         "homeTeam":{"id":1},"awayTeam":{"id":2},"competition":{"id":39,"season":2024}}


def evidence():
    past={**FIXTURE,"id":10,"utcDate":"2025-01-20T12:00:00Z","status":"FINISHED",
          "score":{"fullTime":{"home":2,"away":1},"duration":"REGULAR"}}
    return build_evidence(FIXTURE,[past],NOW-timedelta(minutes=1))


def test_roundtrip_immutable_receipt(tmp_path):
    e=evidence()
    assert early.freeze(FIXTURE,e,root=tmp_path,clock=lambda:NOW)=="written"
    path=next((tmp_path/"receipts").glob("*.json"))
    before=path.read_bytes()
    r=early.load(path,root=tmp_path,now=NOW)
    assert r["history_input_digest"]==e["input_digest"]
    assert early.freeze(FIXTURE,e,root=tmp_path,clock=lambda:NOW)=="original_preserved"
    assert path.read_bytes()==before
    model={"provider":"API-Football","fixture_id":"999","home_team_id":"1","away_team_id":"2",
           "kickoff_utc":FIXTURE["utcDate"],"evidence_as_of_utc":e["as_of_utc"],
           "input_digest":e["input_digest"],"captured_at_utc":NOW.isoformat()}
    assert qualifies(r,model,NOW)
    assert not qualifies(r,model,NOW-timedelta(seconds=1))
    assert not qualifies(r,{**model,"input_digest":"wrong"},NOW)


def test_tampered_blob_rejected(tmp_path):
    early.freeze(FIXTURE,evidence(),root=tmp_path,clock=lambda:NOW)
    blob=next((tmp_path/"blobs").glob("*.json"))
    rows=json.loads(blob.read_text())
    rows[0]["score"]["fullTime"]["home"]=9
    blob.write_text(json.dumps(rows))
    with pytest.raises(ValueError,match="invalid_history_blob"):
        early.load(next((tmp_path/"receipts").glob("*.json")),root=tmp_path,now=NOW)


def test_late_capture_and_wrong_digest(tmp_path):
    assert early.freeze(FIXTURE,evidence(),root=tmp_path,clock=lambda:NOW+timedelta(hours=3))=="outside_pre_match_window"
    assert not list(tmp_path.glob("receipts/*.json"))
    with pytest.raises(ValueError,match="history_input_digest_mismatch"):
        early.freeze(FIXTURE,{**evidence(),"input_digest":"wrong"},root=tmp_path,clock=lambda:NOW)


def test_content_addressed_history_reused(tmp_path):
    e=evidence()
    early.freeze(FIXTURE,e,root=tmp_path,clock=lambda:NOW)
    second={**FIXTURE,"id":1000}
    e2=build_evidence(second,e["history"],NOW-timedelta(minutes=1))
    assert early.freeze(second,e2,root=tmp_path,clock=lambda:NOW)=="written"
    assert len(list(tmp_path.glob("receipts/*.json")))==2
    assert len(list(tmp_path.glob("blobs/*.json")))==1
