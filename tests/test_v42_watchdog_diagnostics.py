"""Watchdog diagnostics must not alter the frozen V42 protocol or make real calls."""
from datetime import datetime, timedelta, timezone
import requests
import v42_research_bot as bot

def test_window_diagnostics_separates_old_and_future(tmp_path, monkeypatch):
    now = datetime(2026, 11, 25, 4, tzinfo=timezone.utc)
    pairs = tmp_path / "pairs"
    (pairs / "sets").mkdir(parents=True)
    root = tmp_path / "research"
    (root / "trials").mkdir(parents=True)
    leads = {"old": -1, "boundary": 5, "ready": 180,
             "early": 181, "later": 360, "frozen": 90, "invalid": 100}
    for name in leads:
        (pairs / "sets" / (name + ".json")).write_text("{}")
    (root / "trials" / "frozen.json").write_text("immutable")
    def pair(path, **kwargs):
        if path.stem == "invalid":
            raise ValueError("invalid")
        kickoff = now + timedelta(minutes=leads[path.stem])
        return {"sydney_date": path.stem}, [{"kickoff_utc": kickoff.isoformat()}]
    monkeypatch.setattr(bot, "_pair", pair)
    result = bot.window_diagnostics(now=now, root=root, pair_root=pairs)
    assert {key: result[key] for key in leads if key in result} == {"invalid": 1}
    assert result["already_frozen"] == 1
    assert result["too_early"] == 2
    assert result["too_late"] == 2
    assert result["in_window"] == 1
    assert result["next_window_opens_utc"] == (now + timedelta(minutes=1)).isoformat()
    assert (root / "trials" / "frozen.json").read_text() == "immutable"

def test_ai_http_status_is_safe_and_not_retried(tmp_path, monkeypatch):
    monkeypatch.setenv("V42_AI_RESEARCH_ENABLED", "true")
    monkeypatch.setenv("OPENAI_API_KEY", "synthetic-secret")
    calls = []
    def post(*args, **kwargs):
        calls.append(1)
        response = requests.Response()
        response.status_code = 429
        raise requests.HTTPError("synthetic-secret", response=response)
    monkeypatch.setattr(bot.requests, "post", post)
    report = {"evaluation": {k: {} for k in ("methods", "same_cutoff_v41_comparison",
              "lineup_coverage", "source_rejections", "result_rejections")}}
    assert bot.advisory(report, root=tmp_path) == "failed_http_429"
    assert bot.advisory(report, root=tmp_path) == "daily_attempt_already_reserved"
    assert calls == [1]
    assert all("synthetic-secret" not in p.read_text() for p in tmp_path.glob("*.json"))


def test_advisory_evidence_retains_http_failure_across_no_ai_refresh(tmp_path, monkeypatch):
    monkeypatch.setenv("V42_AI_RESEARCH_ENABLED", "true")
    monkeypatch.setenv("OPENAI_API_KEY", "synthetic-secret")
    calls = []
    def post(*args, **kwargs):
        calls.append(1)
        response = requests.Response()
        response.status_code = 429
        raise requests.HTTPError("synthetic-secret", response=response)
    monkeypatch.setattr(bot.requests, "post", post)
    report = {"evaluation": {k: {} for k in ("methods", "same_cutoff_v41_comparison",
              "lineup_coverage", "source_rejections", "result_rejections")}}
    assert bot.advisory(report, root=tmp_path) == "failed_http_429"
    assert bot.advisory_evidence(root=tmp_path)["outcome"] == "failed_http_429"
    assert bot.advisory(report, root=tmp_path) == "daily_attempt_already_reserved"
    assert bot.advisory_evidence(root=tmp_path)["outcome"] == "failed_http_429"
    assert calls == [1]
    assert all("synthetic-secret" not in p.read_text() for p in tmp_path.glob("*.json"))

def test_advisory_evidence_distinguishes_unattempted_reserved_and_saved(tmp_path):
    now = datetime(2026, 11, 25, 4, tzinfo=timezone.utc)
    day = "2026-11-25"
    assert bot.advisory_evidence(root=tmp_path, now=now)["outcome"] == "not_attempted"
    (tmp_path / ("ai_attempt_"+day+".json")).write_text('{"status":"reserved"}')
    assert bot.advisory_evidence(root=tmp_path, now=now)["outcome"] == "reserved_no_outcome"
    (tmp_path / ("ai_outcome_"+day+".json")).write_text('{"day":"2026-11-25","status":"saved"}')
    assert bot.advisory_evidence(root=tmp_path, now=now)["outcome"] == "inconsistent_outcome"
    (tmp_path / ("ai_memo_"+day+".json")).write_text('{"advisory_only":true,"memo":"test"}')
    result = bot.advisory_evidence(root=tmp_path, now=now)
    assert result["outcome"] == "saved"
    assert result["memo_present"] is True
