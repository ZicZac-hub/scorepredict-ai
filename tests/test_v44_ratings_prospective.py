from datetime import datetime,timedelta,timezone
import json
import numpy as np

from history_pipeline import build_evidence
import v44_early_history as early
import v44_ratings_prospective as trial
from v44_pair_transfer_research import _arm

NOW=datetime(2025,2,1,10,tzinfo=timezone.utc)
FIXTURE={"provider":"API-Football","id":999,"utcDate":"2025-02-01T12:00:00Z",
         "homeTeam":{"id":1},"awayTeam":{"id":2},"competition":{"id":39,"season":2024}}


def receipt(root,fixture,cutoff):
    past={**fixture,"id":10,"utcDate":"2025-01-20T12:00:00Z","status":"FINISHED",
          "score":{"fullTime":{"home":2,"away":1},"duration":"REGULAR"}}
    e=build_evidence(fixture,[past],cutoff)
    assert early.freeze(fixture,e,root=root,clock=lambda:cutoff+timedelta(seconds=1))=="written"
    return {"provider":fixture["provider"],"fixture_id":str(fixture["id"]),
            "home_team_id":str(fixture["homeTeam"]["id"]),"away_team_id":str(fixture["awayTeam"]["id"]),
            "home_team":"H"+str(fixture["id"]),"away_team":"A"+str(fixture["id"]),
            "kickoff_utc":fixture["utcDate"],"evidence_as_of_utc":e["as_of_utc"],
            "input_digest":e["input_digest"],"captured_at_utc":(cutoff+timedelta(seconds=2)).isoformat(),
            "score_matrix":np.full((5,5),1/25).tolist()}


def test_exact_digest_multiple_captures(tmp_path):
    a=receipt(tmp_path,FIXTURE,NOW-timedelta(minutes=10))
    b=receipt(tmp_path,FIXTURE,NOW-timedelta(minutes=5))
    assert len(list(tmp_path.glob("receipts/*.json")))==2
    assert trial.receipt_for(a,NOW,tmp_path)[2]["history_input_digest"]==a["input_digest"]
    assert trial.receipt_for(b,NOW,tmp_path)[2]["history_input_digest"]==b["input_digest"]
    assert trial.receipt_for({**b,"input_digest":"wrong"},NOW,tmp_path) is None


def test_pair_freeze_and_settle(tmp_path,monkeypatch):
    history=tmp_path/"history"; root=tmp_path/"trial"; pair_root=tmp_path/"pairs"
    a=receipt(history,FIXTURE,NOW-timedelta(minutes=5))
    b=receipt(history,{**FIXTURE,"id":1000,"homeTeam":{"id":3},"awayTeam":{"id":4}},
              NOW-timedelta(minutes=5))
    models=[a,b]
    original=_arm([np.asarray(m["score_matrix"]) for m in models])
    pair={"record_sha256":"original","selection":{"choices":original["five_doubles"],
                     "estimated_coverage":original["estimated_coverage"]}}
    p=pair_root/"sets"/"2025-02-01.json"
    p.parent.mkdir(parents=True);p.write_text("{}")
    monkeypatch.setattr(trial.sources,"_pair",lambda *a,**k:(pair,models))
    monkeypatch.setattr(trial.ratings,"predict",lambda *a,**k:
                        {"status":"ready","grid":np.full((5,5),1/25)})
    monkeypatch.setattr(trial,"_result_index",lambda *a:({},set(),{}))
    monkeypatch.setattr(trial,"_final",lambda *a:None)
    assert trial.freeze(root=root,history_root=history,pair_root=pair_root,
                        now=NOW,clock=lambda:NOW)["issued"]==1
    saved=(root/"trials"/p.name).read_bytes()
    trial.freeze(root=root,history_root=history,pair_root=pair_root,now=NOW,clock=lambda:NOW)
    assert (root/"trials"/p.name).read_bytes()==saved
    assert trial.evaluate(root=root,history_root=history,pair_root=pair_root,now=NOW)["counts"]["pending"]==1
    monkeypatch.setattr(trial,"_final",lambda *a:[0,0])
    result=trial.evaluate(root=root,history_root=history,pair_root=pair_root,now=NOW+timedelta(hours=4))
    assert result["comparisons"]["original_v8"]["pairs"]==1
    assert result["counts"]["authenticated_issued"]==1


def test_missing_second_leg_abstains(tmp_path):
    a=receipt(tmp_path,FIXTURE,NOW-timedelta(minutes=5))
    import pytest
    with pytest.raises(ValueError,match="insufficient_qualified_team_history|missing_exact_early_history"):
        trial._arms({},[a,{**a,"fixture_id":"missing"}],NOW,tmp_path)
