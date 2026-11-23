"""Bounded bot tests: predeclared score variations and honest evidence gates."""
from datetime import datetime, timezone

import numpy as np
import pytest

from continuous_accuracy_bot import assemble, api_research_memo
from v35_continuous_double_trials import (
    METHODS, _five_hit, _paired_exact_sign_test, score_grid, summarize,
)


def sample_grid():
    return np.outer(
        np.array([.17, .29, .31, .16, .07]),
        np.array([.32, .34, .19, .10, .05]),
    )


def mock_item():
    base = sample_grid()
    return {"grid": base, "alternatives": {}}


def test_challenger_distributions_are_normalized_and_prespecified():
    item = mock_item()
    available = set(METHODS) - {"v25_anchored_75_25"}
    for method in available:
        m = np.asarray(score_grid(item, method), dtype=float)
        assert np.isfinite(m).all()
        assert (m >= 0).all()
        assert m.shape == (5, 5)
        assert float(m.sum()) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="unregistered_forecasting_method"):
        score_grid(item, "temperature_arbitrary_posthoc_optimizer")
    base = np.asarray(score_grid(item, "v8_original"))
    high = np.asarray(score_grid(item, "high_goal_tilt_0_15"))
    low = np.asarray(score_grid(item, "low_goal_tilt_0_15"))
    total = np.indices(base.shape).sum(axis=0)
    assert high[total>=4].sum() > base[total>=4].sum()
    assert low[total>=4].sum() < base[total>=4].sum()


def test_freeze_grade_needs_both_correct_scores_in_one_ticket():
    picks = [{"score1":"1-1","score2":"2-0"},
             {"score1":"2-0","score2":"1-1"}]
    assert _five_hit(picks, ["1-1","2-0"])
    assert not _five_hit(picks, ["1-1","1-1"])


def test_missing_verified_pairs_are_pending_not_zero_accuracy():
    r = summarize([], now=datetime(2035,1,1,tzinfo=timezone.utc))
    assert r["verified_same_pair_days"] == 0
    assert r["pending_same_pair_days"] == 0
    assert r["methods"]["v8_original"]["observed_hit_rate"] is None
    assert r["ready_for_separate_production_review"] == []
    assert r["research_only_methods_not_live_promoted"] is True


def test_uncorroborated_results_cannot_trigger_seventy_percent_gate():
    row = {
        "sydney_day":"2035-01-01", "status":"verified",
        "original_selected_fixture_keys":[["API-Football","123"],["API-Football","456"]],
        "actual_scores":["1-1","2-0"], "hits":{name: True for name in METHODS},
        "proper_scores":{name: [
            {"exact_score_log_loss":2., "multiclass_score_brier":.95,
             "over_3_5_brier":.1, "under_1_5_brier":.1} for _ in range(2)]
             for name in METHODS},
    }
    r = summarize([row], now=datetime(2035,1,1,tzinfo=timezone.utc))
    assert r["methods"]["v8_original"]["observed_hit_rate"] == 1
    assert all(not a["review_gate_passed_on_hit_rate_only"] for a in r["methods"].values())
    assert r["ready_for_separate_production_review"] == []
    assert _paired_exact_sign_test(5, 0) == pytest.approx(.0625)


def test_bot_report_keeps_correlated_pairs_separate_from_daily_results():
    now = datetime(2035,1,1,tzinfo=timezone.utc)
    out = assemble(
        {"observed":{"verified_sets":6,"hits":0,"pending_sets":1}},
        {"verified_matched_pairs":5,"pending_matched_pairs":1,
         "v8":{"five_double_hits":0},"v25":{"five_double_hits":1}},
        {"candidate_pair_rows_independently_verified":156,
         "candidate_pair_rows_awaiting_verified_finals":60,
         "fully_verified_slate_days":1},
        {"unique_verified_physical_fixtures":21,
         "frozen_ticket_failure_attribution":{"v8_optimal_original":{"both_leg_scores_absent":106}}},
        {"frozen_prospective_days":0,"verified_same_pair_days":0,
         "pending_same_pair_days":0,"methods":{},"ready_for_separate_production_review":[]},
        now=now)
    assert out["original_v8_daily_five_doubles"]["hit_rate"] == 0
    assert out["v8_v25_same_pair"]["v25_wins"] == 1
    assert out["v30_frozen_candidate_pair_rows"]["completed_independent_sydney_days"] == 1
    assert out["v30_frozen_candidate_pair_rows"]["correlated_verified"] == 156
    assert out["goal_verified"] is False
    assert out["production_model_changed_by_bot"] is False


def test_no_openai_request_is_made_when_disabled(tmp_path, monkeypatch):
    from continuous_accuracy_bot import run
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AI_RESEARCH_BOT_ENABLED", "false")
    r = run(root=tmp_path/"bot", use_ai=True)
    assert "AI_research_disabled_by_GitHub_variable" == r["ai_connection"]
    assert (tmp_path/"bot"/"status.json").is_file()


def test_llm_response_is_text_only(monkeypatch):
    import continuous_accuracy_bot as module
    class FakeResponse:
        def raise_for_status(self):
            pass
        def json(self):
            return {"output":[{"content":[{"type":"output_text","text":"Check more prospectively frozen games."}]}]}
    def fake_post(url, *,headers,json,timeout):
        assert url.endswith("/responses")
        assert "Bearer example" == headers["Authorization"]
        assert json["store"] is False
        assert json["max_output_tokens"] <= 450
        assert timeout <= 35
        return FakeResponse()
    monkeypatch.setattr(module.requests,"post",fake_post)
    report=assemble(None,None,None,None,None,now=datetime(2035,1,1,tzinfo=timezone.utc))
    note=api_research_memo(report,key="example")
    assert "frozen games" in note["memo"]
    assert note["authorizes_code_execution"] is False


def test_empty_github_model_variable_uses_allowlisted_default(tmp_path, monkeypatch):
    import continuous_accuracy_bot as module
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AI_RESEARCH_MODEL", "")
    monkeypatch.setenv("AI_RESEARCH_BOT_ENABLED", "")
    monkeypatch.setenv("OPENAI_API_KEY", "example-key")
    seen = []
    def fake_memo(report, *, key, model):
        seen.append((key, model))
        return {"memo":"Only new frozen verified matches can validate a challenger.",
                "source":"OpenAI_API_not_the_ChatGPT_website",
                "authorizes_code_execution":False}
    monkeypatch.setattr(module, "api_research_memo", fake_memo)
    result = module.run(root=tmp_path/"bot")
    assert seen == [("example-key", module.DEFAULT_AI_MODEL)]
    assert result["ai_connection"] == "daily_research_memo_saved"
    assert len(list((tmp_path/"bot").glob("openai_research_*.json"))) == 1
    module.run(root=tmp_path/"bot")
    assert len(seen) == 1


def test_http_429_is_logged_and_never_retried_same_utc_day(tmp_path, monkeypatch):
    import continuous_accuracy_bot as module
    import requests
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AI_RESEARCH_MODEL", "")
    monkeypatch.setenv("AI_RESEARCH_BOT_ENABLED", "")
    monkeypatch.setenv("OPENAI_API_KEY", "example-key")
    attempts = []
    def rate_limited(report, *, key, model):
        attempts.append(model)
        response = requests.Response()
        response.status_code = 429
        raise requests.HTTPError("Too Many Requests", response=response)
    monkeypatch.setattr(module, "api_research_memo", rate_limited)
    first = module.run(root=tmp_path/"bot")
    second = module.run(root=tmp_path/"bot")
    assert attempts == [module.DEFAULT_AI_MODEL]
    assert first["ai_connection"] == "api_http_429"
    assert second["ai_connection"] == "api_http_429"
    attempts_file = list((tmp_path/"bot").glob("openai_attempt_*.json"))
    assert len(attempts_file) == 1
    import json
    assert json.loads(attempts_file[0].read_text())["max_attempts_per_day"] == 1


def test_manual_key_rotation_retries_one_prior_401_once(tmp_path, monkeypatch):
    import json
    import requests
    import continuous_accuracy_bot as module
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AI_RESEARCH_BOT_ENABLED", "")
    monkeypatch.setenv("OPENAI_API_KEY", "rotated-test-key")
    now=datetime(2035,1,1,tzinfo=timezone.utc)
    attempts=[]
    def unauthorized(report, *, key, model):
        attempts.append("401")
        response=requests.Response()
        response.status_code=401
        raise requests.HTTPError("Unauthorized", response=response)
    monkeypatch.setattr(module,"api_research_memo",unauthorized)
    first=module.run(root=tmp_path/"bot",now=now)
    assert first["ai_connection"]=="api_http_401"
    original=list((tmp_path/"bot").glob("openai_attempt_*.json"))
    assert len(original)==1
    old=original[0].read_bytes()
    def successful(report, *, key, model):
        attempts.append("success")
        assert key=="rotated-test-key"
        return {"memo":"Evaluate future-only frozen pairs.",
                "authorizes_code_execution":False}
    monkeypatch.setattr(module,"api_research_memo",successful)
    second=module.run(root=tmp_path/"bot",now=now,manual_retry=True)
    assert second["ai_connection"]=="daily_research_memo_saved"
    assert attempts==["401","success"]
    assert original[0].read_bytes()==old
    retry=list((tmp_path/"bot").glob("openai_retry_*.json"))
    assert len(retry)==1
    assert json.loads(retry[0].read_text())["manual_key_rotation_retry"] is True
    assert len(list((tmp_path/"bot").glob("openai_research_*.json")))==1
    module.run(root=tmp_path/"bot",now=now,manual_retry=True)
    assert attempts==["401","success"]


def test_manual_retry_cannot_bypass_429_or_repeat_401(tmp_path, monkeypatch):
    import requests
    import json
    import continuous_accuracy_bot as module
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AI_RESEARCH_BOT_ENABLED","")
    monkeypatch.setenv("OPENAI_API_KEY","dummy")
    now=datetime(2035,1,1,tzinfo=timezone.utc)
    attempts=[]
    def reject(report, *, key, model):
        attempts.append("request")
        response=requests.Response()
        response.status_code=401
        raise requests.HTTPError("Unauthorized",response=response)
    monkeypatch.setattr(module,"api_research_memo",reject)
    first=module.run(root=tmp_path/"bot",now=now,manual_retry=True)
    assert first["ai_connection"]=="api_http_401"
    module.run(root=tmp_path/"bot",now=now,manual_retry=True)
    module.run(root=tmp_path/"bot",now=now,manual_retry=True)
    assert attempts==["request","request"]
    assert len(list((tmp_path/"bot").glob("openai_retry_*.json")))==1
    retry=json.loads(list((tmp_path/"bot").glob("openai_retry_*.json"))[0].read_text())
    assert retry["status"]=="api_http_401"

    later=datetime(2035,1,2,tzinfo=timezone.utc)
    def limited(report, *, key, model):
        attempts.append("429")
        response=requests.Response()
        response.status_code=429
        raise requests.HTTPError("Too many requests",response=response)
    monkeypatch.setattr(module,"api_research_memo",limited)
    module.run(root=tmp_path/"bot",now=later)
    module.run(root=tmp_path/"bot",now=later,manual_retry=True)
    assert attempts==["request","request","429"]
    assert not (tmp_path/"bot"/"openai_retry_2035-01-02.json").exists()


def test_workflow_uses_manual_owner_dispatch_only():
    from pathlib import Path
    source=Path(".github/workflows/continuous-five-double-bot.yml").read_text()
    assert "retry_openai_after_key_rotation:" in source
    assert "github.event_name == 'workflow_dispatch'" in source
    assert "python continuous_accuracy_bot.py --retry-after-key-rotation" in source
    assert "data/autobot/openai_retry_*.json" in source


def test_openai_http_error_diagnostic_is_allowlisted_and_secret_free():
    import requests
    from continuous_accuracy_bot import classify_openai_http_error
    response = requests.Response()
    response.status_code = 429
    response._content = b'{"error":{"code":"insufficient_quota","type":"insufficient_quota","message":"sensitive details","param":"private"}}'
    assert classify_openai_http_error(response) == {
        "http_status":429, "error_code":"insufficient_quota",
        "error_type":"insufficient_quota"}
    response._content = b'{"error":{"code":"unknown_secret","type":"custom","message":"token"}}'
    diagnostic = classify_openai_http_error(response)
    assert diagnostic["error_code"] == "unavailable"
    assert diagnostic["error_type"] == "unavailable"
    response._content = b'not-json'
    assert classify_openai_http_error(response)["error_code"] == "unavailable"


def test_openai_429_diagnostic_persisted_without_retry_or_raw_message(tmp_path, monkeypatch):
    import json
    import requests
    import continuous_accuracy_bot as module
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "example-secret")
    monkeypatch.setenv("AI_RESEARCH_BOT_ENABLED", "")
    now = datetime(2035, 1, 1, tzinfo=timezone.utc)
    calls = []
    def rejected(report, *, key, model):
        calls.append(1)
        response = requests.Response()
        response.status_code = 429
        response._content = b'{"error":{"code":"rate_limit_exceeded","type":"rate_limit_exceeded","message":"private-key-example-secret"}}'
        raise requests.HTTPError("secret", response=response)
    monkeypatch.setattr(module, "api_research_memo", rejected)
    first = module.run(root=tmp_path/"bot", now=now)
    second = module.run(root=tmp_path/"bot", now=now)
    assert calls == [1]
    assert first["ai_http_error_diagnostic"]["error_code"] == "rate_limit_exceeded"
    assert second["ai_http_error_diagnostic"] == first["ai_http_error_diagnostic"]
    receipt = (tmp_path/"bot"/"openai_attempt_2035-01-01.json").read_text()
    assert "private-key" not in receipt
    assert "example-secret" not in receipt
    assert json.loads(receipt)["http_error_diagnostic"]["http_status"] == 429
