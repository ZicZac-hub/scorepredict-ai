"""Documented finish metadata cannot become a successful football AI vote."""
import json

import pytest

from provider_response_diagnostics import ProviderResponseError, check_completion


@pytest.mark.parametrize("provider,payload,reason", [
    ("openai", {"choices": [{"finish_reason": "length"}]}, "provider_output_limit"),
    ("grok", {"choices": [{"finish_reason": "content_filter"}]}, "provider_refusal"),
    ("openai", {"choices": [{"message": {"refusal": "Private text"}}]}, "provider_refusal"),
    ("mistral", {"choices": [{"finish_reason": "tool_calls"}]}, "provider_tool_output_unavailable"),
    ("claude", {"stop_reason": "max_tokens"}, "provider_output_limit"),
    ("claude", {"stop_reason": "model_context_window_exceeded"}, "provider_output_limit"),
    ("claude", {"stop_reason": "refusal"}, "provider_refusal"),
    ("claude", {"stop_reason": "tool_use"}, "provider_tool_output_unavailable"),
    ("claude", {"stop_reason": "pause_turn"}, "provider_turn_incomplete"),
    ("cohere", {"finish_reason": "MAX_TOKENS"}, "provider_output_limit"),
    ("cohere", {"finish_reason": "TOOL_CALL"}, "provider_tool_output_unavailable"),
    ("cohere", {"finish_reason": "ERROR"}, "provider_generation_failed"),
    ("cohere", {"finish_reason": "TIMEOUT"}, "provider_timeout"),
    ("gemini", {"candidates": [{"finishReason": "MAX_TOKENS"}]}, "provider_output_limit"),
    ("gemini", {"candidates": [{"finishReason": "CONTINUATION"}]}, "provider_output_limit"),
    ("gemini", {"candidates": [{"finishReason": "SAFETY"}]}, "provider_refusal"),
    ("gemini", {"promptFeedback": {"blockReason": "PROHIBITED_CONTENT"}}, "provider_refusal"),
    ("gemini", {"candidates": [{"finishReason": "UNEXPECTED_TOOL_CALL"}]}, "provider_tool_output_unavailable"),
    ("gemini", {"candidates": [{"finishReason": "MALFORMED_RESPONSE"}]}, "provider_generation_failed"),
])
def test_documented_noncompletion_is_classified_without_payload(provider, payload, reason):
    payload["private_prompt"] = "credential-shaped-sensitive-content"
    with pytest.raises(ProviderResponseError) as captured:
        check_completion(payload, provider)
    assert captured.value.reason == reason
    assert str(captured.value) == reason
    assert "private" not in str(captured.value)
    assert "credential" not in str(captured.value)


@pytest.mark.parametrize("provider,payload", [
    ("openai", {"choices": [{"finish_reason": "stop"}]}),
    ("claude", {"stop_reason": "end_turn"}),
    ("claude", {"stop_reason": "stop_sequence"}),
    ("cohere", {"finish_reason": "COMPLETE"}),
    ("gemini", {"candidates": [{"finishReason": "STOP"}]}),
    ("perplexity", {"output": []}),
    ("openai", {}),
    ("openai", {"choices": []}),
    ("gemini", {"candidates": [None], "promptFeedback": None}),
    ("grok", {"choices": [{"finish_reason": "unexpected private raw text"}]}),
])
def test_normal_and_legacy_metadata_remain_subject_to_existing_json_validation(provider, payload):
    original = json.dumps(payload, sort_keys=True)
    assert check_completion(payload, provider) is None
    assert json.dumps(payload, sort_keys=True) == original


def test_untrusted_diagnostic_cannot_be_used_as_an_exception_message():
    with pytest.raises(ValueError, match="Unsupported provider response diagnostic") as captured:
        ProviderResponseError("secret value from a provider body")
    assert "secret value" not in str(captured.value)
