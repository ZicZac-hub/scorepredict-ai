import json
from datetime import datetime, timedelta, timezone
from threading import Barrier

import requests

from v43_research import provider_status, render_v43_research, run_v43_research


NOW = datetime(2031, 11, 20, 8, tzinfo=timezone.utc)
FIXTURE = {"provider": "API-Football", "id": 991,
           "utcDate": "2031-11-20T12:00:00Z",
           "homeTeam": {"id": 11, "name": "Home Eleven"},
           "awayTeam": {"id": 12, "name": "Away United"}}
STATISTICAL = {"model_version": "v43", "scores": [[2, 1, 0.12]],
               "outcome_probabilities": {"home": 0.52, "draw": 0.25, "away": 0.23}}


class Server:
    def __init__(self, *, signed_in=True, configured=True):
        self.session_state = {"ziczac_access_granted": signed_in}
        self.secrets = {
            "ZICZAC_ACCESS_PASSWORD": "owner-password",
            "OPENAI_API_KEY": "openai-private-key",
            "OPENAI_MODEL": "openai-test-model",
            "ANTHROPIC_API_KEY": "claude-private-key",
        }
        if configured:
            self.secrets["ANTHROPIC_MODEL"] = "claude-test-model"


class Response:
    def __init__(self, payload):
        self.payload = payload

    def json(self):
        return self.payload

    def raise_for_status(self):
        pass


def vote(*, factors=None, uncertainty=""):
    return json.dumps({"home_win": 0.52, "draw": 0.25, "away_win": 0.23,
                       "exact_home": 2, "exact_away": 1,
                       "factors": factors or [], "uncertainty": uncertainty})


def test_provider_readiness_requires_private_access_and_complete_settings():
    signed_in = provider_status(Server(configured=False), environ={})
    assert signed_in["private_access"] is True
    assert signed_in["providers"]["openai"] == "ready"
    assert signed_in["providers"]["claude"] == "not_configured"
    assert "private-key" not in str(signed_in)
    signed_out = provider_status(Server(signed_in=False), environ={})
    assert signed_out["private_access"] is False
    assert set(signed_out["providers"].values()) == {"access_required"}


def test_private_gate_and_kickoff_prevent_requests():
    calls = []

    def request(*args, **kwargs):
        calls.append(args)

    result = run_v43_research(Server(signed_in=False), FIXTURE, [], STATISTICAL,
                             now=NOW, environ={}, request=request)
    assert result["status"] == "private_access_required"
    result = run_v43_research(Server(), FIXTURE, [], STATISTICAL,
                             now=NOW.replace(hour=12), environ={}, request=request)
    assert result["status"] == "kickoff_too_close"
    server = Server()
    server.secrets = {"ZICZAC_ACCESS_PASSWORD": "owner-password"}
    result = run_v43_research(server, FIXTURE, [], STATISTICAL, now=NOW,
                             environ={}, request=request)
    assert result["status"] == "no_streamlit_api_credentials"
    assert calls == []


def test_openai_and_claude_consult_identical_cutoff_evidence_concurrently():
    ready = Barrier(2, timeout=5)
    calls = {}
    historical = {"provider": "API-Football", "id": 9911, "status": "FINISHED",
                  "utcDate": "2031-11-18T09:00:00Z", "homeTeam": FIXTURE["homeTeam"],
                  "awayTeam": {"id": 99, "name": "Earlier Opponent"},
                  "score": {"fullTime": {"home": 2, "away": 0}}}
    after_cutoff = {**historical, "id": 9912, "observed_at_utc": "2031-11-20T09:00:00Z",
                    "awayTeam": {"id": 97, "name": "Future Observation"}}

    def request(url, **kwargs):
        calls[url] = kwargs
        ready.wait()
        opinion = vote(factors=["Earlier Opponent finished before the cutoff"],
                       uncertainty="One result gives limited support")
        if url.startswith("https://api.anthropic.com/"):
            return Response({"stop_reason": "end_turn", "content": [{"type": "text", "text": opinion}]})
        return Response({"choices": [{"finish_reason": "stop", "message": {"content": opinion}}]})

    result = run_v43_research(Server(), FIXTURE, [historical, after_cutoff], STATISTICAL,
                             now=NOW, environ={}, request=request)
    assert result["status"] == "completed"
    assert result["attempted"] == ["openai", "claude"]
    assert set(result["votes"]) == {"openai", "claude"}
    assert result["consensus"]["suggested_score"] == [2, 1]
    assert result["research_only"] is True
    assert result["ai_numeric_weight"] == 0.0
    openai = calls["https://api.openai.com/v1/chat/completions"]
    claude = calls["https://api.anthropic.com/v1/messages"]
    assert openai["allow_redirects"] is claude["allow_redirects"] is False
    assert openai["headers"]["Authorization"] == "Bearer openai-private-key"
    assert claude["headers"]["x-api-key"] == "claude-private-key"
    assert openai["json"]["messages"] == claude["json"]["messages"]
    prompt = openai["json"]["messages"][0]["content"]
    assert "Earlier Opponent" in prompt and "Future Observation" not in prompt
    assert '"statistical_model_version": "v43"' in prompt
    assert "private-key" not in prompt


def test_one_provider_can_fail_without_disclosing_credentials_or_dropping_other_vote():
    def request(url, **kwargs):
        if url.startswith("https://api.anthropic.com/"):
            raise requests.Timeout("private provider error from claude-private-key")
        opinion = vote(factors=["Earlier match", "echoed openai-private-key"],
                       uncertainty="echoed claude-private-key")
        return Response({"choices": [{"message": {"content": opinion}}]})

    result = run_v43_research(Server(), FIXTURE, [], STATISTICAL, now=NOW,
                             environ={}, request=request)
    assert result["status"] == "completed"
    assert result["votes"]["openai"]["factors"] == ["Earlier match"]
    assert result["votes"]["openai"]["uncertainty"] == ""
    assert result["errors"]["claude"]["reason"] == "provider_timeout"
    assert "private-key" not in str(result)


def test_all_provider_failures_return_a_failed_research_status():
    def request(*args, **kwargs):
        raise requests.Timeout()

    result = run_v43_research(Server(), FIXTURE, [], STATISTICAL, now=NOW,
                             environ={}, request=request)
    assert result["status"] == "providers_failed"
    assert result["votes"] == {}
    assert set(result["errors"]) == {"openai", "claude"}


def test_research_ui_respects_per_match_consent_and_reuses_session_result(monkeypatch):
    from contextlib import nullcontext

    import v43_research

    class App(Server):
        consent = False
        clicked = True
        checkbox_keys = []

        def checkbox(self, label, *, key):
            self.checkbox_keys.append(key)
            return self.consent

        def button(self, label, **kwargs):
            return self.clicked

        def spinner(self, label):
            return nullcontext()

        def markdown(self, value):
            pass

        def caption(self, value):
            pass

    calls = []

    def run(st, fixture, *args, **kwargs):
        calls.append(fixture)
        return {"attempted": [], "votes": {}, "errors": {}, "status": "completed"}

    monkeypatch.setattr(v43_research, "run_v43_research", run)
    monkeypatch.setattr(v43_research, "frozen_for_fixture", lambda fixture: None)
    kickoff = (datetime.now(timezone.utc) + timedelta(hours=4)).isoformat()
    fixture = {**FIXTURE, "utcDate": kickoff}
    rescheduled_teams = {**fixture, "homeTeam": {"id": 19, "name": "Other Home"}}
    app = App()

    render_v43_research(app, fixture, [], STATISTICAL)
    assert calls == []
    app.consent = True
    render_v43_research(app, fixture, [], STATISTICAL)
    render_v43_research(app, fixture, [], STATISTICAL)
    assert calls == [fixture]

    app.consent = False
    render_v43_research(app, rescheduled_teams, [], STATISTICAL)
    assert calls == [fixture]
    assert len(set(app.checkbox_keys)) == 2
