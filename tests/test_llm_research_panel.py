"""Regression coverage for the opt-in pre-match five-AI comparison."""
import json
from datetime import datetime, timezone

import pytest

from llm_research_panel import (
    _extract, capture, consensus, eligible, settings, validate_vote
)


def test_panel_requires_both_model_and_key():
    env = {"OPENAI_API_KEY": "secret", "OPENAI_MODEL": "test-model",
           "ANTHROPIC_API_KEY": "another-key"}
    assert list(settings(env)) == ["openai"]


def test_strict_score_and_probability_validation():
    good = {"home_win": .5, "draw": .3, "away_win": .2,
            "exact_home": 1, "exact_away": 0, "factors": ["Home strength"]}
    assert validate_vote(good)["score"] == [1, 0]
    with pytest.raises(ValueError):
        validate_vote(dict(good, away_win=.8))
    with pytest.raises(ValueError):
        validate_vote(dict(good, exact_home=True))
    with pytest.raises(ValueError):
        validate_vote(dict(good, exact_home=12))


def test_json_fences_and_incompatible_model_formats():
    class Fake:
        def json(self):
            return {"choices": [{"message": {"content": "'''json"}}]}
    # A markdown-fenced answer must be parsed, without trusting explanatory prose.
    payload = {"home_win": .5, "draw": .3, "away_win": .2,
               "exact_home": 1, "exact_away": 0}
    class Response:
        def json(self):
            return {"choices": [{"message": {"content": "```json\n" +
                         json.dumps(payload) + "\n```"}}]}
    assert _extract(Response(), "openai") == payload


def test_consensus_is_outcome_only_not_an_exact_score_probability():
    votes = {name: {"home": .5, "draw": .3, "away": .2, "score": [1, 0]}
             for name in ("openai", "claude", "gemini", "grok", "deepseek")}
    result = consensus(votes)
    assert result["provider_count"] == 5
    assert result["complete_five_provider_panel"] is True
    assert result["suggested_score"] == [1, 0]
    assert result["outcome_probabilities"]["home"] == .5
    assert "exact_score_probability" not in result


def test_no_after_kickoff_predictions():
    row = {"model_key": "v8", "provider": "API-Football",
           "fixture_id": "123", "input_digest": "0"*64, "forecast_id": "a",
           "kickoff_utc": "2026-11-15T12:00:00Z",
           "captured_at_utc": "2026-11-14T01:00:00Z",
           "expected_goals": {"home": 1.4, "away": 1.1},
           "top10_scores": [[1, 1, .12]]}
    assert eligible(row, datetime(2026, 11, 15, 1, tzinfo=timezone.utc))
    assert not eligible(row, datetime(2026, 11, 15, 13, tzinfo=timezone.utc))


def test_missing_keys_make_no_paid_api_calls(tmp_path):
    result = capture(environ={"LLM_RESEARCH_ENABLED": "true"}, root=tmp_path)
    assert result["attempted"] == 0
    assert result["reason"] == "provider_keys_or_models_missing"
    assert json.loads((tmp_path/"status.json").read_text())["captured"] == 0
