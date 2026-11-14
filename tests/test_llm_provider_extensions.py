"""Regression tests for ten optional LLM adapters; no network, keys or billing."""
import json
import pytest

from llm_research_panel import (
    PROVIDERS, _extract, _qwen_url, ask, classify_provider_error, consensus, settings
)


VOTE = {"home_win": .5, "draw": .3, "away_win": .2,
        "exact_home": 1, "exact_away": 0, "factors": []}


def response(payload):
    class Fake:
        def raise_for_status(self):
            pass
        def json(self):
            return payload
    return Fake()


def test_every_provider_requires_own_key_and_model():
    env={}
    for name,prefix in (("perplexity","PERPLEXITY"),("mistral","MISTRAL"),
                        ("qwen","QWEN"),("groqcloud","GROQ"),("cohere","COHERE")):
        env[prefix+"_API_KEY"]="placeholder"
        env[prefix+"_MODEL"]="test-model"
    got=settings(env)
    assert set(got)=={"perplexity","mistral","qwen","groqcloud","cohere"}
    assert len(PROVIDERS)==10


def test_agent_api_perplexity_and_cohere_response_shapes():
    agent={"output":[{"type":"message","content":[
        {"type":"output_text","text":json.dumps(VOTE)}]}]}
    assert _extract(response(agent),"perplexity")==VOTE
    cohere={"message":{"content":[{"type":"text","text":json.dumps(VOTE)}]}}
    assert _extract(response(cohere),"cohere")==VOTE


@pytest.mark.parametrize("name,endpoint", [
    ("perplexity","https://api.perplexity.ai/v1/agent"),
    ("mistral","https://api.mistral.ai/v1/chat/completions"),
    ("qwen","https://dashscope-intl.aliyuncs.com/compatible-mode/v1/chat/completions"),
    ("groqcloud","https://api.groq.com/openai/v1/chat/completions"),
    ("cohere","https://api.cohere.com/v2/chat"),
])
def test_official_endpoints_and_strict_nonredirects(name, endpoint):
    seen={}
    def post(url, **kwargs):
        seen.update(url=url, **kwargs)
        if name=="perplexity":
            return response({"output":[{"type":"message","content":[
                {"type":"output_text","text":json.dumps(VOTE)}]}]})
        if name=="cohere":
            return response({"message":{"content":[{"type":"text","text":json.dumps(VOTE)}]}})
        return response({"choices":[{"message":{"content":json.dumps(VOTE)}}]})
    result=ask(name,"placeholder","test-model","pre-match evidence",request=post)
    assert result["score"]==[1,0]
    assert seen["url"]==endpoint
    assert seen["allow_redirects"] is False
    assert seen["timeout"]<=40
    assert seen["headers"]["Authorization"]=="Bearer placeholder"


def test_qwen_only_alibaba_https_dedicated_or_legacy_hosts():
    assert _qwen_url(None).startswith("https://dashscope-intl")
    assert _qwen_url(
        "https://workspace.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1/chat/completions"
    ).startswith("https://workspace.")
    for hostile in ("http://dashscope-intl.aliyuncs.com/compatible-mode/v1/chat/completions",
                    "https://evil.com/compatible-mode/v1/chat/completions",
                    "https://dashscope-intl.aliyuncs.com.evil.com/compatible-mode/v1/chat/completions",
                    "https://dashscope-intl.aliyuncs.com/other"):
        with pytest.raises(ValueError):
            _qwen_url(hostile)


def test_ten_votes_do_not_count_as_verified_exact_score_accuracy():
    votes={name:{"home":.5,"draw":.3,"away":.2,"score":[1,0]} for name in PROVIDERS}
    result=consensus(votes)
    assert result["complete_five_provider_panel"] is True
    assert result["complete_ten_provider_panel"] is True
    assert "exact_score_probability" not in result

def test_safe_provider_errors_never_reveal_keys_or_provider_body():
    import requests
    response = requests.Response()
    response.status_code = 401
    response.url = "https://example.com/secret-should-not-be-printed?key=example"
    response._content = b"Private credential must not be logged"
    error = requests.HTTPError("Secret test token", response=response)
    result = classify_provider_error(error)
    assert result == {"type":"HTTPError", "http_status":401,
                      "reason":"invalid_or_expired_key"}
    assert "Secret" not in str(result)
    response.status_code = 429
    assert classify_provider_error(error)["reason"] == "quota_or_rate_limit"
    assert classify_provider_error(ValueError("sensitive response")) == {
        "type":"ValueError", "reason":"response_format_or_model_configuration"}

@pytest.mark.parametrize("name", ["openai", "grok", "mistral", "groqcloud", "cohere"])
def test_structured_json_requested_for_supported_providers(name):
    seen = {}
    def post(url, **kwargs):
        seen.update(kwargs)
        if name == "cohere":
            return response({"message": {"content": [{"type": "text", "text": json.dumps(VOTE)}]}})
        return response({"choices": [{"message": {"content": json.dumps(VOTE)}}]})
    ask(name, "placeholder", "test-model", "Return only JSON", request=post)
    assert seen["json"]["response_format"] == {"type": "json_object"}
    assert seen["allow_redirects"] is False


def test_groq_gpt_oss_limits_reasoning_to_leave_room_for_json():
    seen = {}
    def post(url, **kwargs):
        seen.update(kwargs)
        return response({"choices": [{"message": {"content": json.dumps(VOTE)}}]})
    ask("groqcloud", "placeholder", "openai/gpt-oss-20b", "Return JSON", request=post)
    assert seen["json"]["reasoning_effort"] == "low"
    assert seen["json"]["reasoning_format"] == "hidden"
    assert seen["json"]["max_tokens"] >= 800


def test_plain_and_fenced_json_responses_without_guessing_prose():
    fenced = response({"choices": [{"message": {"content": "```JSON\n" + json.dumps(VOTE) + "\n```"}}]})
    assert _extract(fenced, "openai") == VOTE
    import pytest
    empty = response({"choices": [{"message": {"content": None}}]})
    with pytest.raises(ValueError):
        _extract(empty, "groqcloud")
    segments = response({"choices": [{"message": {"content": [
        {"type": "text", "text": json.dumps(VOTE)}
    ]}}]})
    assert _extract(segments, "openai") == VOTE


def test_existing_immutable_panel_prevents_duplicate_paid_calls(tmp_path, monkeypatch):
    import llm_research_panel as panel
    from datetime import datetime, timezone
    kickoff = "2026-11-15T12:00:00Z"
    row = {"provider": "API-Football", "fixture_id": "321",
           "model_key": "v8", "forecast_id": "original",
           "input_digest": "a"*64, "captured_at_utc": "2026-11-14T00:00:00Z",
           "kickoff_utc": kickoff, "expected_goals": {"home": 1.2, "away": 1.0},
           "top10_scores": [[1,1,.12]]}
    monkeypatch.setattr(panel, "read_latest", lambda source: {("API-Football","321","v8"): row})
    root = tmp_path / "research"
    (root / "forecasts").mkdir(parents=True)
    (root / "forecasts" / "API-Football_321_prior.json").write_text(json.dumps({
        "research_only": True, "provider": "API-Football",
        "fixture_id": "321", "captured_at_utc": "2026-11-14T03:00:00Z",
        "kickoff_utc": kickoff
    }))
    def network(*args, **kwargs):
        raise AssertionError("A published fixture must not incur repeated AI calls")
    result = panel.capture(
        now=datetime(2026,11,15,1,tzinfo=timezone.utc),
        environ={"LLM_RESEARCH_ENABLED": "true", "OPENAI_API_KEY": "placeholder",
                 "OPENAI_MODEL": "gpt-4o-mini"},
        root=root, source=tmp_path, request=network)
    assert result["attempted"] == 0
    assert result["captured"] == 0

def test_latest_real_errors_survive_no_fixture_research_pass(tmp_path, monkeypatch):
    import llm_research_panel as panel
    root = tmp_path / "research"
    root.mkdir()
    (root / "status.json").write_text(json.dumps({
        "checked_at_utc": "2026-11-14T01:00:00Z",
        "last_attempt_at_utc": "2026-11-14T01:00:00Z",
        "last_attempt_errors": {"openai": {
            "type": "HTTPError", "http_status": 429,
            "reason": "quota_or_rate_limit"}}
    }))
    monkeypatch.setattr(panel, "read_latest", lambda source: {})
    report = panel.capture(
        environ={"LLM_RESEARCH_ENABLED": "true",
                 "OPENAI_API_KEY": "placeholder", "OPENAI_MODEL": "gpt-4o-mini"},
        root=root, source=tmp_path)
    assert report["attempted"] == 0
    assert report["last_attempt_errors"]["openai"]["http_status"] == 429
    assert report["last_attempt_at_utc"] == "2026-11-14T01:00:00Z"


def test_ai_generated_reports_allowed_in_safe_rebase():
    from reconcile_capture_conflicts import REPORT_EXACT
    for name in ("data/llm_research/status.json",
                 "data/llm_research/evaluation.json",
                 "data/independent_football/status.json",
                 "data/independent_football/evaluation.json"):
        assert name in REPORT_EXACT
