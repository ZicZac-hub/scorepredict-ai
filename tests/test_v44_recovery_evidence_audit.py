"""Synthetic orchestration checks; no fixture is a real accuracy sample."""
from datetime import datetime,timedelta,timezone
import json
import numpy as np
import pytest

from forecast_ledger import append_file_record,read_file_records
from history_pipeline import build_evidence
import v44_recovery_capture as recovery
import v44_recovery_evidence_audit as audit

NOW=datetime(2025,2,1,10,tzinfo=timezone.utc)
FIXTURE={"provider":"API-Football","id":1,"utcDate":(NOW+timedelta(hours=2)).isoformat(),
         "homeTeam":{"id":10,"name":"Home"},"awayTeam":{"id":20,"name":"Away"},
         "competition":{"id":39,"season":2024,"code":"PL"}}

@pytest.fixture
def archive(tmp_path,monkeypatch):
    source=tmp_path/"source";root=tmp_path/"recovery"
    original={"provider":"API-Football","fixture_id":"1","home_team_id":"10","away_team_id":"20",
              "home_team":"Home","away_team":"Away","competition":"PL",
              "kickoff_utc":FIXTURE["utcDate"],"model_key":"v8_near","model_version":"prior-near",
              "horizon":"near","score_basis":"regulation","score_matrix":[[.04]*5 for _ in range(5)],
              "input_digest":"missing-original-input",
              "captured_at_utc":(NOW-timedelta(hours=3)).isoformat(),
              "evidence_as_of_utc":(NOW-timedelta(hours=3,minutes=1)).isoformat()}
    assert append_file_record(original,root=source)
    original=read_file_records(root=source)[0]
    past={**FIXTURE,"id":11,"utcDate":"2025-01-20T12:00:00Z","status":"FINISHED",
          "score":{"fullTime":{"home":2,"away":1},"duration":"REGULAR"}}
    evidence=build_evidence(FIXTURE,[past],NOW-timedelta(minutes=1))
    key=recovery.fixture_key(FIXTURE)
    monkeypatch.setattr(recovery,"targets",lambda **kw:{key:key[2]})
    g=np.full((5,5),.04)
    result={"matrix":g,"scores":[(0,0,.04)],"home_xg":2.,"away_xg":2.,
            "model_version":"test-v8","component_weights":{"a":.5,"b":.5}}
    monkeypatch.setattr(recovery,"research_predict",lambda *a,**kw:result)
    monkeypatch.setattr(audit,"research_predict",lambda *a,**kw:result)
    assert recovery.capture(FIXTURE,evidence,root=root,source_root=source,
                            history_root=tmp_path/"old_history",now=NOW,
                            clock=lambda:NOW)["status"]=="recovery_written"
    monkeypatch.setattr(audit,"_original_pairs",lambda *a:iter([[original]]))
    path=next((root/"archive"/"forecasts").glob("*/*/*.json"))
    kwargs={"root":root,"source_root":source,"pair_root":tmp_path/"pairs",
            "prematch_root":tmp_path/"prematch"}
    return kwargs,path,original

def test_reproduction_and_independent_final_grade(archive):
    kw,path,original=archive
    before={p:p.read_bytes() for p in kw["root"].rglob("*.json")}
    pending=audit.evaluate(**kw,now=NOW)
    assert pending["counts"]=={"authenticated":1,"pending_or_unverified":1}
    assert pending["comparisons"]["later_recovery"]["hit_rate"] is None
    final={k:original[k] for k in ("provider","fixture_id","home_team_id","away_team_id","kickoff_utc")}
    final.update({"status":"FT","evidence_version":2,"score_basis":"regulation",
                  "home_goals":0,"away_goals":0,
                  "verified_at_utc":(NOW+timedelta(hours=5)).isoformat()})
    (kw["source_root"]/"verified_results.json").write_text(json.dumps({"1":final}))
    # The verified result is still future information at NOW.
    assert audit.evaluate(**kw,now=NOW)["counts"]["pending_or_unverified"]==1
    report=audit.evaluate(**kw,now=NOW+timedelta(hours=6))
    assert report["counts"]=={"authenticated":1,"verified":1}
    assert report["comparisons"]["later_recovery"]["hits"]==1
    assert not report["rejections"] and not report["automatic_promotion"]
    assert before=={p:p.read_bytes() for p in kw["root"].rglob("*.json")}

def test_tampered_receipt_link_is_rejected(archive):
    kw,path,_=archive
    row=json.loads(path.read_text());row["recovery_receipt_sha256"]="changed"
    path.write_text(json.dumps(row))
    r=audit.evaluate(**kw,now=NOW)
    assert r["rejections"]=={"recovery_receipt_mismatch":1}
    assert r["comparisons"]["later_recovery"]["fixtures"]==0

def test_reproduction_failure_and_missing_original_pair(archive,monkeypatch):
    kw,path,_=archive
    grid=np.zeros((5,5));grid[0,0]=1.
    monkeypatch.setattr(audit,"research_predict",lambda *a,**k:{
        "matrix":grid,"model_version":"test-v8","component_weights":{"a":.5,"b":.5}})
    assert audit.evaluate(**kw,now=NOW)["rejections"]=={"recovery_grid_not_reproduced":1}
    monkeypatch.setattr(audit,"_original_pairs",lambda *a:iter([]))
    assert audit.evaluate(**kw,now=NOW)["rejections"]=={"no_original_pair_at_capture":1}

def test_empty_archive_is_not_zero_accuracy(tmp_path):
    r=audit.evaluate(root=tmp_path/"absent",source_root=tmp_path/"source",
                     pair_root=tmp_path/"pairs",prematch_root=tmp_path/"prematch",now=NOW)
    assert r["archive_files"]==0
    assert r["comparisons"]["later_recovery"]["hit_rate"] is None
    assert not (tmp_path/"absent").exists()
