"""Same immutable AI evidence, honest clocks and documented request budgets."""
from datetime import datetime,timedelta,timezone
import hashlib
import json

import pytest

import llm_research_panel as panel
from provider_response_diagnostics import ProviderResponseError

NOW=datetime(2034,11,20,0,tzinfo=timezone.utc)
VOTE={"home_win":.5,"draw":.3,"away_win":.2,"exact_home":2,"exact_away":1}


def row(lead=180):
    return {"provider":"API-Football","fixture_id":"900","forecast_id":"original",
            "model_key":"v8","input_digest":"a"*64,"competition":"PL",
            "home_team_id":"11","away_team_id":"22","home_team":"Home","away_team":"Away",
            "captured_at_utc":(NOW-timedelta(hours=2)).isoformat(),
            "kickoff_utc":(NOW+timedelta(minutes=lead)).isoformat(),
            "expected_goals":{"home":1.6,"away":1.1},"top10_scores":[[2,1,.12]],
            "predicted_score":[2,1],"outcome_probabilities":{"home":.5,"draw":.3,"away":.2}}


def research():
    games=[]
    for team in (11,22):
        for days in range(1,21):
            games.append({"id":team*100+days,"provider":"API-Football","status":"FINISHED",
                "utcDate":(NOW-timedelta(days=days)).isoformat(),
                "fetched_at_utc":(NOW-timedelta(minutes=3)).isoformat(),
                "homeTeam":{"id":team,"name":str(team)},"awayTeam":{"id":team+50,"name":"Opponent"},
                "competition":{"id":39,"code":"PL"},
                "score":{"duration":"REGULAR","fullTime":{"home":2,"away":1}}})
    return {"history":games,"input_digest":"qualified-research",
            "league_evidence_digest":"league-evidence","source_metadata":[{
                "fetched_at_utc":(NOW-timedelta(minutes=3)).isoformat(),"freshness_known":True}],
            "warnings":[],"external_requests":0}


class Response:
    def __init__(self,payload):self.payload=payload
    def raise_for_status(self):pass
    def json(self):return self.payload


def setup_capture(monkeypatch,record):
    import cached_ai_history
    import v21_capture_sources
    monkeypatch.setattr(panel,"read_latest",lambda source:{"one":record})
    monkeypatch.setattr(v21_capture_sources,"best_packet",lambda *a,**k:None)
    monkeypatch.setattr(cached_ai_history,"cached_history",lambda *a,**k:research())
    return {"LLM_RESEARCH_ENABLED":"true","OPENAI_API_KEY":"private-openai-key",
            "OPENAI_MODEL":"o3","XAI_API_KEY":"private-grok-key","XAI_MODEL":"grok-test"}


def test_capture_persists_one_reproducible_common_dossier_for_all_providers(monkeypatch,tmp_path):
    record=row()
    env=setup_capture(monkeypatch,record)
    prompts=[]
    def post(url,**kwargs):
        prompts.append(kwargs["json"]["messages"][0]["content"])
        return Response({"choices":[{"finish_reason":"stop","message":{"content":json.dumps(VOTE)}}]})
    report=panel.capture(now=NOW,clock=lambda:NOW,environ=env,root=tmp_path,
                         request=post,provider_root=tmp_path/"packets")
    saved=json.loads(next((tmp_path/"forecasts").glob("*.json")).read_text())
    assert len(prompts)==2 and prompts[0]==prompts[1]
    assert prompts[0]==panel._briefing_text(saved["shared_evidence"])
    assert saved["shared_prompt_sha256"]==hashlib.sha256(prompts[0].encode()).hexdigest()
    assert saved["shared_evidence_sha256"]==hashlib.sha256(json.dumps(
        saved["shared_evidence"],sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
    form=saved["shared_evidence"]["professional_match_evidence"]["features"]["goal_form"]["value"]
    assert [form["home"]["last"+str(n)]["matches"] for n in (5,10,20)]==[5,10,20]
    assert len(saved["research_history"])==40
    assert saved["research_as_of_utc"]==NOW.isoformat()
    assert saved["captured_at_utc"]==NOW.isoformat()
    assert saved["input_digest"]==record["input_digest"]
    assert saved["research_history_external_requests"]==0
    assert report["research_context_counts"]=={"with_cached_history":1,"with_provider_packet":0}
    assert saved["ai_numeric_weight"]==0
    assert all(secret not in json.dumps(saved) for secret in (env["OPENAI_API_KEY"],env["XAI_API_KEY"]))


@pytest.mark.parametrize("delay,captured",[(42,1),(50,0)])
def test_slow_response_stops_following_calls_and_keeps_actual_save_time(monkeypatch,tmp_path,delay,captured):
    env=setup_capture(monkeypatch,row(46))
    observed=[NOW]
    calls=[]
    def post(url,**kwargs):
        calls.append(url)
        observed[0]=NOW+timedelta(minutes=delay)
        return Response({"choices":[{"message":{"content":json.dumps(VOTE)}}]})
    report=panel.capture(now=NOW,clock=lambda:observed[0],environ=env,root=tmp_path,request=post)
    assert len(calls)==1
    assert report["captured"]==captured
    assert report["provider_error_details"]["grok"]["reason"]=="pre_match_window_expired"
    paths=list((tmp_path/"forecasts").glob("*.json"))
    assert len(paths)==captured
    if paths:
        saved=json.loads(paths[0].read_text())
        assert saved["captured_at_utc"]==observed[0].isoformat()
        assert saved["research_as_of_utc"]==NOW.isoformat()


def test_openai_uses_documented_completion_limit_without_extra_tokens():
    seen={}
    def post(url,**kwargs):
        seen.update(kwargs["json"])
        return Response({"choices":[{"message":{"content":json.dumps(VOTE)}}]})
    panel.ask("openai","placeholder","o3","Return JSON",request=post)
    assert seen["max_completion_tokens"]==800
    assert "max_tokens" not in seen
    seen.clear()
    panel.ask("grok","placeholder","grok-test","Return JSON",request=post)
    assert seen["max_tokens"]==800 and "max_completion_tokens" not in seen


@pytest.mark.parametrize("provider,payload",[
    ("openai",{"choices":[{"finish_reason":"length","message":{"content":json.dumps(VOTE)}}]}),
    ("cohere",{"finish_reason":"MAX_TOKENS","message":{"content":[{"type":"text","text":json.dumps(VOTE)}]}}),
])
def test_documented_incomplete_answers_keep_safe_error_reason(provider,payload):
    with pytest.raises(ProviderResponseError) as error:
        panel._extract(Response(payload),provider)
    assert panel.classify_provider_error(error.value)=={
        "type":"ProviderResponseError","reason":"provider_output_limit"}
