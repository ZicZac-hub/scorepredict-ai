from datetime import datetime,timedelta,timezone
import numpy as np
from forecast_ledger import append_file_record,read_file_records
from history_pipeline import build_evidence
import v44_recovery_capture as r

NOW=datetime(2025,2,1,10,tzinfo=timezone.utc)
FIXTURE={"provider":"API-Football","id":1,"utcDate":(NOW+timedelta(hours=2)).isoformat(),
         "homeTeam":{"id":10,"name":"Home"},"awayTeam":{"id":20,"name":"Away"},
         "competition":{"id":39,"season":2024,"code":"PL"}}

def setup(tmp_path,monkeypatch):
    source=tmp_path/"source";root=tmp_path/"recovery"
    original={"provider":"API-Football","fixture_id":"1","home_team_id":"10","away_team_id":"20",
              "home_team":"Home","away_team":"Away","competition":"PL",
              "kickoff_utc":FIXTURE["utcDate"],"model_key":"v8_near","model_version":"prior-near",
              "horizon":"near","score_basis":"regulation","score_matrix":[[.04]*5 for _ in range(5)],
              "input_digest":"missing-original-input",
              "captured_at_utc":(NOW-timedelta(hours=3)).isoformat(),
              "evidence_as_of_utc":(NOW-timedelta(hours=3,minutes=1)).isoformat()}
    assert append_file_record(original,root=source)
    past={**FIXTURE,"id":11,"utcDate":"2025-01-20T12:00:00Z","status":"FINISHED",
          "score":{"fullTime":{"home":2,"away":1},"duration":"REGULAR"}}
    evidence=build_evidence(FIXTURE,[past],NOW-timedelta(minutes=1))
    key=r.fixture_key(FIXTURE)
    monkeypatch.setattr(r,"targets",lambda **kw:{key:key[2]})
    # Test archive orchestration with a deterministic model result; no
    # prediction accuracy or genuine historical model skill is inferred.
    g=np.full((5,5),.04)
    monkeypatch.setattr(r,"research_predict",lambda *a,**kw:{
        "matrix":g,"scores":[(0,0,.04)],"home_xg":2.,"away_xg":2.,
        "model_version":"test-v8","component_weights":{"a":.5,"b":.5}})
    kwargs=dict(root=root,source_root=source,history_root=tmp_path/"original_history",
                now=NOW,clock=lambda:NOW)
    return source,root,evidence,kwargs

def test_recovery_is_isolated_and_repeat_preserves_bytes(tmp_path,monkeypatch):
    source,root,evidence,kwargs=setup(tmp_path,monkeypatch)
    before={p:p.read_bytes() for p in source.rglob("*.json")}
    result=r.capture(FIXTURE,evidence,**kwargs)
    assert result["status"]=="recovery_written"
    assert result["external_requests"]==0
    records=read_file_records(root=root/"archive",kind="forecasts")
    assert len(records)==1 and records[0]["model_key"]==r.MODEL_KEY
    assert records[0]["input_digest"]==evidence["input_digest"]
    assert records[0]["recovery_receipt_sha256"]==result["receipt_sha256"]
    frozen={p:p.read_bytes() for p in root.rglob("*.json")}
    assert r.capture(FIXTURE,evidence,**kwargs)["status"]=="recovery_preserved"
    assert frozen=={p:p.read_bytes() for p in root.rglob("*.json")}
    assert before=={p:p.read_bytes() for p in source.rglob("*.json")}
    assert not kwargs["history_root"].exists()
    assert r.read_latest(root/"archive")=={}

def test_recovery_rejects_old_evidence_and_nonmatching_pair(tmp_path,monkeypatch):
    source,root,evidence,kwargs=setup(tmp_path,monkeypatch)
    older={**evidence,"as_of_utc":(NOW-timedelta(hours=4)).isoformat()}
    assert r.capture(FIXTURE,older,**kwargs)["status"]=="history_not_genuinely_newer"
    monkeypatch.setattr(r,"targets",lambda **kw:{})
    assert r.capture(FIXTURE,evidence,**kwargs)["status"]=="not_authenticated_original_pair_leg"
    assert not root.exists()

def test_late_or_future_observations_never_archive(tmp_path,monkeypatch):
    source,root,evidence,kwargs=setup(tmp_path,monkeypatch)
    future={**evidence,"as_of_utc":(NOW+timedelta(seconds=1)).isoformat()}
    assert r.capture(FIXTURE,future,**kwargs)["status"]=="history_not_genuinely_newer"
    too_late=NOW+timedelta(hours=1,minutes=30)
    assert r.capture(FIXTURE,evidence,**{**kwargs,"now":too_late,"clock":lambda:too_late})["status"]=="outside_refresh_window"
    assert not root.exists()
