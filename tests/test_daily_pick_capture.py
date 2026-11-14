"""Scheduled daily feature: pre-kickoff, separate from original forecast ledger."""
from datetime import datetime,timedelta,timezone
from zoneinfo import ZoneInfo
import json

import daily_pick_capture
import daily_pick_ui

SYDNEY=ZoneInfo("Australia/Sydney")


def prepared(now,ident=101,confidence=.16):
    tomorrow=now+timedelta(hours=3)
    model={
        "scores":[(1,0,confidence),(2,1,.10)],
        "home_win":.6,"draw":.24,"away_win":.16,
        "home_xg":1.7,"away_xg":.8,
        "home_last10":[{}]*10,"away_last10":[{}]*10,
        "home_home10":[{}]*5,"away_away10":[{}]*5,
        "h2h_last10":[{}]*2,
        "component_weights":{"preserved_v6":.55,"opponent_dixon_coles_v7":.45},
        "component_count":2,"component_top_score_agreement":1,
        "model_version":daily_pick_ui.MODEL}
    fixture={"id":ident,"provider":"API-Football","utcDate":tomorrow.isoformat(),
             "homeTeam":{"id":10,"name":"Home Club"},
             "awayTeam":{"id":20,"name":"Away Club"}}
    evidence={"input_digest":"sha256-frozen-before-kickoff","policy_version":"asof",
              "history":[],"league_history":[]}
    return fixture,evidence,model


def test_capture_model_result_is_frozen_before_kickoff(monkeypatch):
    import research_ensemble_v8
    now=datetime(2026,11,14,11,0,tzinfo=SYDNEY).astimezone(timezone.utc)
    fixture,evidence,forecast=prepared(now)
    monkeypatch.setattr(research_ensemble_v8,"predict",lambda *args,**kwargs:forecast)
    selection=daily_pick_capture.candidate(fixture,evidence,now,"PL")
    assert selection["model"]==daily_pick_ui.MODEL
    assert selection["status"]=="frozen_in_capture_workflow"
    assert selection["exact_score"]==[1,0]
    assert selection["evidence_input_digest"]=="sha256-frozen-before-kickoff"
    assert selection["captured_at_utc"]<selection["kickoff_utc"]
    assert daily_pick_capture.candidate(fixture,evidence,now+timedelta(hours=4),"PL") is None


def test_write_once_archives_shared_selection_across_new_sessions(monkeypatch,tmp_path):
    import research_ensemble_v8
    now=datetime(2026,11,14,11,0,tzinfo=SYDNEY).astimezone(timezone.utc)
    fixture,evidence,forecast=prepared(now)
    monkeypatch.setattr(research_ensemble_v8,"predict",lambda *args,**kwargs:forecast)
    pick=daily_pick_capture.candidate(fixture,evidence,now,"PL")
    saved=daily_pick_capture.save_for_dates([pick],root=tmp_path,now=now)
    assert saved==["2026-11-14"]
    target=tmp_path/"2026-11-14.json"
    original=target.read_bytes()
    edited={**pick,"exact_score":[4,3],"probability":.75}
    assert daily_pick_capture.save_for_dates([edited],root=tmp_path,now=now)==[]
    assert target.read_bytes()==original
    loaded=daily_pick_ui.choose_daily_snapshot([],{},as_of=now,snapshot_dir=tmp_path)
    assert loaded["exact_score"]==[1,0]
    assert loaded["fixture_identity"]==["API-Football","101"]
    assert loaded["status"]=="frozen_in_capture_workflow"
    assert daily_pick_ui.choose_daily_snapshot([],{},as_of=now+timedelta(hours=4),snapshot_dir=tmp_path)["exact_score"]==[1,0]


def test_invalid_shared_or_future_result_is_rejected(tmp_path):
    now=datetime(2026,11,14,11,0,tzinfo=SYDNEY).astimezone(timezone.utc)
    fixture,evidence,forecast=prepared(now)
    base=daily_pick_ui.build_snapshot(
        {"fixture":fixture,"league":"PL","kickoff":fixture["utcDate"],"result":forecast},
        now,forecast)
    bad={**base,"status":"frozen_in_capture_workflow",
         "captured_at_utc":(now+timedelta(hours=4)).isoformat()}
    (tmp_path/"2026-11-14.json").write_text(json.dumps(bad))
    assert daily_pick_ui.choose_daily_snapshot([],{},as_of=now,snapshot_dir=tmp_path) is None


def test_capture_does_not_claim_unverified_sparse_history(monkeypatch):
    import research_ensemble_v8
    now=datetime(2026,11,14,11,0,tzinfo=SYDNEY).astimezone(timezone.utc)
    fixture,evidence,forecast=prepared(now)
    sparse={**forecast,"home_last10":[{}]*3}
    monkeypatch.setattr(research_ensemble_v8,"predict",lambda *args,**kwargs:sparse)
    assert daily_pick_capture.candidate(fixture,evidence,now,"PL") is None
