"""Safe, fixed diagnostics for unusable successful AI responses.

No response text, model identifier, request URL, prompt or credential is kept.
This checks documented termination metadata; it never retries or increases a
provider's output/request allowance.

Provider references (checked 19 November 2026):
https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create
https://docs.cohere.com/v2/reference/chat
https://platform.claude.com/docs/en/build-with-claude/handling-stop-reasons
https://ai.google.dev/api/generate-content
"""

REASONS = frozenset(("provider_output_limit", "provider_refusal",
                     "provider_tool_output_unavailable", "provider_turn_incomplete",
                     "provider_generation_failed", "provider_timeout"))


class ProviderResponseError(ValueError):
    """Only an allowlisted category is exposed to existing error handlers."""

    def __init__(self, reason):
        if reason not in REASONS:
            raise ValueError("Unsupported provider response diagnostic")
        self.reason = reason
        super().__init__(reason)


def _first(value):
    return value[0] if isinstance(value, list) and value and isinstance(value[0], dict) else {}


def _reason(value):
    return value.casefold() if isinstance(value, str) else ""


def check_completion(payload, provider):
    """Reject a documented incomplete/refused generation before parsing votes.

    Missing finish metadata remains compatible with older documented payloads
    and injected test responses; local JSON/vote validation still applies.
    Unknown strings are never copied into an error or interpreted as a known
    status. Even parseable JSON at a reported token limit is not a complete
    accepted research response under this policy.
    """
    if not isinstance(payload, dict):
        return
    if provider == "claude":
        status = _reason(payload.get("stop_reason"))
        if status in ("max_tokens", "model_context_window_exceeded"):
            raise ProviderResponseError("provider_output_limit")
        if status == "refusal":
            raise ProviderResponseError("provider_refusal")
        if status == "tool_use":
            raise ProviderResponseError("provider_tool_output_unavailable")
        if status == "pause_turn":
            raise ProviderResponseError("provider_turn_incomplete")
    elif provider == "cohere":
        status = _reason(payload.get("finish_reason"))
        if status == "max_tokens":
            raise ProviderResponseError("provider_output_limit")
        if status == "tool_call":
            raise ProviderResponseError("provider_tool_output_unavailable")
        if status == "error":
            raise ProviderResponseError("provider_generation_failed")
        if status == "timeout":
            raise ProviderResponseError("provider_timeout")
    elif provider == "gemini":
        feedback = payload.get("promptFeedback")
        blocked = _reason(feedback.get("blockReason")) if isinstance(feedback, dict) else ""
        if blocked in ("safety", "blocklist", "prohibited_content", "image_safety", "other"):
            raise ProviderResponseError("provider_refusal")
        candidate = _first(payload.get("candidates"))
        status = _reason(candidate.get("finishReason"))
        if status in ("max_tokens", "continuation"):
            raise ProviderResponseError("provider_output_limit")
        if status in ("safety", "recitation", "blocklist", "prohibited_content", "spii",
                      "image_safety", "image_prohibited_content", "image_recitation",
                      "escalation", "pup_limited_disabled"):
            raise ProviderResponseError("provider_refusal")
        if status in ("malformed_function_call", "unexpected_tool_call", "too_many_tool_calls"):
            raise ProviderResponseError("provider_tool_output_unavailable")
        if status == "malformed_response":
            raise ProviderResponseError("provider_generation_failed")
    else:
        choice = _first(payload.get("choices"))
        status = _reason(choice.get("finish_reason"))
        if status == "length":
            raise ProviderResponseError("provider_output_limit")
        message = choice.get("message")
        if status == "content_filter" or (isinstance(message, dict) and message.get("refusal")):
            raise ProviderResponseError("provider_refusal")
        if status in ("tool_calls", "function_call"):
            raise ProviderResponseError("provider_tool_output_unavailable")
