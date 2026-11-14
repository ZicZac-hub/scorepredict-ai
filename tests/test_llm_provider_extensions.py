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
