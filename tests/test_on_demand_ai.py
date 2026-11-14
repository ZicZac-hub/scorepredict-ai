"""Selected-match AI calls: no secrets in UI, no after-kickoff or fake evidence."""
import json
from datetime import datetime, timedelta, timezone

import pytest

from on_demand_ai import (
    _verified_history, briefing, fixture_key, frozen_for_fixture,
    private_gate_ready, run_selected, server_settings
)


def match(start="2031-11-20T12:00:00Z"):
    return {"provider": "API-Football", "id": 991,
            "utcDate": start,
            "homeTeam": {"id": 11, "name": "Home Eleven"},
            "awayTeam": {"id": 12, "name": "Away United"},
            "competition": {"name": "Friendly Test League"}}


def history():
    return [
        {"status": "FINISHED", "utcDate": "2031-11-18T09:00:00Z",
         "homeTeam": {"id": 11, "name": "Home Eleven"},
         "awayTeam": {"id": 44, "name": "Rival"},
         "score": {"fullTime": {"home": 2, "away": 0}}},
        {"status": "FINISHED", "utcDate": "2031-11-21T09:00:00Z",
         "homeTeam": {"id": 12, "name": "Away United"},
         "awayTeam": {"id": 44, "name": "Rival"},
         "score": {"fullTime": {"home": 5, "away": 0}}},
        {"status": "SCHEDULED", "utcDate": "2031-11-18T10:00:00Z",
         "homeTeam": {"id": 11, "name": "Home Eleven"},
         "awayTeam": {"id": 12, "name": "Away United"},
         "score": {"fullTime": {"home": 0, "away": 0}}},
    ]


def test_pre_match_prompt_includes_real_history_only():
    now=datetime(2031,11,20,8,tzinfo=timezone.utc)
    prompt=briefing(match(),history(),{
        "model_version":"v8-test",
        "scores":[[1,0,.17],[1,1,.15]],
        "home_xg":1.2,"away_xg":0.9,
        "outcome_probabilities":{"home":0.47,"draw":0.30,"away":0.23}
    },now=now)
    assert "2, 0" in prompt
    assert "5, 0" not in prompt
    assert "unavailable" not in prompt.lower() or "not_provided" in prompt
    assert "REPLACE-WITH-SECRET" not in prompt
    assert len(_verified_history(match(),history(),now))==1


@pytest.mark.parametrize("start,now",[
    ("2031-11-20T12:00:00Z","2031-11-20T12:00:00Z"),
    ("2031-11-20T12:00:00Z","2031-11-20T11:57:00Z"),
    ("2031-11-20T12:00:00Z","2031-11-20T13:00:00Z"),
])
def test_no_external_api_calls_near_or_after_kickoff(start,now):
    def forbid(*args,**kwargs):
        raise AssertionError("No API requests allowed near kickoff")
    result=run_selected(match(start),history(),{}, {},{
        "openai":("fake-key","test-model")},
        now=datetime.fromisoformat(now.replace("Z","+00:00")),request=forbid)
    assert result["status"]=="kickoff_too_close"
    assert result["votes"]=={}


def test_requires_both_server_side_key_and_model():
    class Fake:
        secrets={"OPENAI_API_KEY":"not-real",
                 "OPENAI_MODEL":"gpt-4o-mini",
                 "GEMINI_API_KEY":"not-real"}
    configured=server_settings(Fake(),environ={})
    assert list(configured)==["openai"]
    assert configured["openai"]==("not-real","gpt-4o-mini")


def test_selected_match_consults_two_configured_providers_without_leaking_secrets():
    requests_seen=[]
    vote={"home_win":.5,"draw":.3,"away_win":.2,
          "exact_home":2,"exact_away":1,"factors":["verified historical form"]}
    def stub(url, **kwargs):
        requests_seen.append((url,kwargs))
        class Response:
            def raise_for_status(self): pass
            def json(self):
                return {"choices":[{"message":{"content":json.dumps(vote)}}]}
        return Response()
    result=run_selected(match(),history(),{
        "model_version":"historical",
        "scores":[[2,1,.14]],
        "outcome_probabilities":{"home":.5,"draw":.25,"away":.25},
        "home_xg":1.5, "away_xg":1.1,
    },{},{
        "openai":("supersecret-openai","test-model"),
        "groqcloud":("supersecret-groq","test-model"),
    },now=datetime(2031,11,20,8,tzinfo=timezone.utc),request=stub)
    assert set(result["votes"])=={"openai","groqcloud"}
    assert result["consensus"]["suggested_score"]==[2,1]
    assert len(requests_seen)==2
    assert result["ephemeral"] is True
    assert result["evaluated_as_frozen_five_double"] is False
    assert "supersecret-" not in str(result)
    assert "supersecret-" not in str([x[1]["json"] for x in requests_seen])
    assert fixture_key(match())==("API-Football","991","2031-11-20T12:00:00+00:00")


def test_no_unavailable_credentials_masquerading_as_live_ai():
    result=run_selected(match(),history(),{}, {},{},
                        now=datetime(2031,11,20,8,tzinfo=timezone.utc))
    assert result["status"]=="no_streamlit_api_credentials"
    assert result["votes"]=={}


def test_unsupported_credentials_dont_leak_into_errors():
    def fail(url,**kwargs):
        raise ValueError("PRIVATE_KEY_IN_ERROR")
    result=run_selected(match(),history(),{}, {},{
        "openai":("secret-string","test-model")},
        now=datetime(2031,11,20,8,tzinfo=timezone.utc),request=fail)
    assert "openai" in result["errors"]
    assert "PRIVATE_KEY_IN_ERROR" not in str(result)
    assert "secret-string" not in str(result)

def test_excludes_unrelated_league_results_and_120_minute_scores():
    rows=history()+[
        {"status":"FINISHED","utcDate":"2031-11-18T07:00:00Z",
         "homeTeam":{"id":83,"name":"Unrelated"},
         "awayTeam":{"id":84,"name":"Another"},
         "score":{"fullTime":{"home":9,"away":0}}},
        {"status":"FINISHED","utcDate":"2031-11-17T07:00:00Z",
         "homeTeam":{"id":11,"name":"Home Eleven"},
         "awayTeam":{"id":15,"name":"Cup Rival"},
         "score":{"duration":"EXTRA_TIME","fullTime":{"home":3,"away":2},
                  "regularTime":{"home":1,"away":1}}}
    ]
    now=datetime(2031,11,20,8,tzinfo=timezone.utc)
    output=_verified_history(match(),rows,now)
    assert len(output)==2
    assert all(x["home"]!="Unrelated" for x in output)
    assert next(x for x in output if x["away"]=="Cup Rival")["full_time_90min"]==[1,1]


def test_matches_existing_git_hub_ai_records_by_exact_fixture_and_pre_kickoff_time(tmp_path):
    from pathlib import Path
    folder=Path(tmp_path)/"forecasts"
    folder.mkdir()
    baseline={
        "research_only": True, "provider": "API-Football", "fixture_id":"991",
        "kickoff_utc":"2031-11-20T12:00:00Z",
        "captured_at_utc":"2031-11-20T06:00:00Z",
        "home_team_id":"11","away_team_id":"12",
        "votes":{"gemini":{"score":[1,1],"home":.4,"draw":.3,"away":.3}}
    }
    (folder/"a.json").write_text(json.dumps(baseline))
    assert frozen_for_fixture(match(),root=tmp_path)["votes"]["gemini"]["score"]==[1,1]
    # A previous freeze is never replaced by a later, more flattering vote.
    revised=dict(baseline, captured_at_utc="2031-11-20T08:00:00Z",
                 votes={"openai":{"score":[2,1],"home":.6,"draw":.2,"away":.2}})
    (folder/"b.json").write_text(json.dumps(revised))
    assert frozen_for_fixture(match(),root=tmp_path)["votes"]["gemini"]["score"]==[1,1]
    # The stored API-Football ID must never be treated as the same id
    # from a different fixture data provider.
    wrong_provider=dict(match(),provider="football-data.org")
    assert frozen_for_fixture(wrong_provider,root=tmp_path) is None
    wrong_team=dict(match(),homeTeam={"id":999,"name":"Wrong Team"})
    assert frozen_for_fixture(wrong_team,root=tmp_path) is None
    wrong_kickoff=match("2031-11-20T13:00:00Z")
    assert frozen_for_fixture(wrong_kickoff,root=tmp_path) is None


def test_frozen_ai_records_must_have_original_capture_before_kickoff(tmp_path):
    folder=tmp_path/"forecasts"
    folder.mkdir()
    record={
        "research_only":True,"provider":"API-Football","fixture_id":"991",
        "kickoff_utc":"2031-11-20T12:00:00Z",
        "captured_at_utc":"2031-11-20T12:00:01Z",
        "home_team_id":"11","away_team_id":"12",
        "votes":{"gemini":{"score":[1,1],"home":.4,"draw":.3,"away":.3}}
    }
    (folder/"late.json").write_text(json.dumps(record))
    assert frozen_for_fixture(match(),root=tmp_path) is None
    record["captured_at_utc"]="2031-11-20T11:00:00Z"
    record["votes"]={}
    (folder/"late.json").write_text(json.dumps(record))
    assert frozen_for_fixture(match(),root=tmp_path) is None


def test_paid_model_calls_require_real_private_password_and_login():
    class DummyApp:
        def __init__(self, secret, logged_in):
            self.secrets={"ZICZAC_ACCESS_PASSWORD": secret}
            self.session_state={"ziczac_access_granted": logged_in}
    assert private_gate_ready(DummyApp("owner-password", True), environ={})
    assert not private_gate_ready(DummyApp("owner-password", False), environ={})
    assert not private_gate_ready(DummyApp("", True), environ={})
    assert not private_gate_ready(DummyApp("owner-password", "true"), environ={})


def test_on_demand_screen_entry_point_remains_backward_compatible():
    import inspect
    from research_ui import render
    signature=inspect.signature(render)
    assert "external_private" in signature.parameters
    assert signature.parameters["external_private"].default is False
