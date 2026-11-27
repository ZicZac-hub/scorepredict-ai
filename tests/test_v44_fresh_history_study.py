from datetime import datetime,timedelta,timezone
import json
import numpy as np
import pytest
import v44_fresh_history_study as s
from forecast_ledger import _forecast

NOW=datetime(2025,2,1,10,tzinfo=timezone.utc)
def model(i="1",captured=NOW,kind="v8_near"):
    return _forecast({"provider":"API-Football","fixture_id":i,"home_team_id":i+"1","away_team_id":i+"2",
        "home_team":"H"+i,"away_team":"A"+i,"kickoff_utc":(NOW+timedelta(hours=2)).isoformat(),
        "captured_at_utc":captured.isoformat(),"evidence_as_of_utc":(captured-timedelta(minutes=1)).isoformat(),
        "input_digest":"digest-"+captured.isoformat(),"model_key":kind,"model_version":"test-"+kind,
        "horizon":"near","score_basis":"regulation","score_matrix":[[.04]*5 for _ in range(5)]})

def save(root,m):
    p=root/"forecasts"/m["provider"]/m["fixture_id"]/(m["forecast_id"]+".json")
    p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(m))

def receipt(root,m):
    key=s.early.receipt_key(list(s.event(m)),m["input_digest"])
    p=root/"receipts"/(key+".json");p.parent.mkdir(parents=True,exist_ok=True)
    row={"event":list(s.event(m)),"history_input_digest":m["input_digest"],
         "as_of_utc":m["evidence_as_of_utc"],"captured_at_utc":m["evidence_as_of_utc"],
         "fixture":{},"history":[],"receipt_sha256":key}
    p.write_text(json.dumps(row));return p

def test_fresh_capture_does_not_repair_original_and_excludes_future(tmp_path,monkeypatch):
    source=tmp_path/"source";history=tmp_path/"history"
    original=model(captured=NOW-timedelta(hours=4),kind="v8")
    fresh=model(captured=NOW-timedelta(minutes=1))
    future=model(captured=NOW+timedelta(seconds=1))
    for m in (original,fresh,future):save(source,m)
    for m in (fresh,future):receipt(history,m)
    monkeypatch.setattr(s.early,"load",lambda path,**kw:json.loads(path.read_text()))
    chosen,r,key=s.source_for(original,NOW,source,history)
    assert chosen["forecast_id"]==fresh["forecast_id"]
    assert r["history_input_digest"]!=original["input_digest"]
    assert not (history/"receipts"/(s.early.receipt_key(list(s.event(original)),original["input_digest"])+".json")).exists()

def test_missing_history_abstains(tmp_path):
    m=model();save(tmp_path,m)
    with pytest.raises(ValueError,match="missing_fresh_exact_history"):
        s.source_for(m,NOW,tmp_path,tmp_path/"history")

def test_unregistered_read_only_evaluation(tmp_path,monkeypatch):
    monkeypatch.setattr(s,"_result_index",lambda *a:({},set(),{}))
    r=s.evaluate(root=tmp_path/"study",now=NOW)
    assert not r["registered"]
    assert r["comparisons"]["fresh_blend"]["hit_rate"] is None
    assert not (tmp_path/"study").exists()

def test_matched_arms_normalized(monkeypatch):
    models=[model("1"),model("2")]
    monkeypatch.setattr(s,"source_for",lambda m,*a:(m,{"fixture":{},"history":[],"as_of_utc":m["evidence_as_of_utc"],"receipt_sha256":"hash"},"key"))
    monkeypatch.setattr(s.ratings,"predict",lambda *a:{"status":"ready","grid":np.full((5,5),.04)})
    refs,arms=s.calculate(models,NOW,None,None)
    assert len(refs)==2
    for a in arms.values():
        assert len(a["five_doubles"])==5
        assert all(np.isclose(np.sum(g),1.) for g in a["full_score_grids"])


def test_readiness_checks_future_without_issuing_or_registering(tmp_path,monkeypatch):
    pairs=tmp_path/"pairs";(pairs/"sets").mkdir(parents=True)
    (pairs/"sets"/"2025-02-01.json").write_text("{}")
    models=[model("1"),model("2")]
    for m in models:
        m["kickoff_utc"]=(NOW+timedelta(hours=5)).isoformat()
    monkeypatch.setattr(s.sources,"_pair",lambda *a,**k:({"predicted_at_utc":NOW.isoformat()},models))
    def get(m,*a):
        if m["fixture_id"]=="2":
            raise ValueError("missing_fresh_exact_history")
        return m,{"fixture":{},"history":[],"as_of_utc":NOW.isoformat()},"key"
    monkeypatch.setattr(s,"source_for",get)
    monkeypatch.setattr(s.ratings,"predict",lambda *a:{"status":"ready"})
    result=s.readiness(root=tmp_path/"study",pair_root=pairs,now=NOW)
    assert result["counts"]["future_pairs_with_1_ready_legs"]==1
    assert result["leg_statuses"]["future:missing_fresh_exact_history"]==1
    assert not (tmp_path/"study").exists()
