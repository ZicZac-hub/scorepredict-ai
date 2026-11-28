"""Test Sportmonks access diagnoses without exhausting live provider quota."""

from datetime import datetime, timezone
import json

import pytest
import requests

from v45_provider_health import archived_health, probe_sportmonks


TOKEN = "private-sportmonks-token"
FIXTURE = {
    "id": 1234, "state_id": 5, "league_id": 8, "season_id": 1002,
    "participants": [{"id": 1, "meta": {"location": "home"}},
                     {"id": 2, "meta": {"location": "away"}}],
}
XG_ROWS = [
    {"type_id": 5304, "fixture_id": 1234, "participant_id": 1,
     "location": "home", "data": {"value": 1.9}},
    {"type_id": 5304, "fixture_id": 1234, "participant_id": 2,
     "location": "away", "data": {"value": 0.0}},
]


class Response:
    def __init__(self, code, payload):
        self.status_code = code
        self.payload = payload

    def json(self):
        assert self.status_code == 200
        if isinstance(self.payload, ValueError):
            raise self.payload
        return self.payload


def sample(responses):
    pending = iter(responses)
    calls = []

    def get(url, **options):
        assert url.startswith("https://api.sportmonks.com/v3/football/")
        assert TOKEN not in url and TOKEN not in str(options["params"])
        assert options["headers"]["Authorization"] == TOKEN
        assert options["allow_redirects"] is False
        calls.append((url.rsplit("/football", 1)[1], options["params"]))
        reply = next(pending)
        if isinstance(reply, requests.RequestException):
            raise reply
        return Response(*reply)

    report = probe_sportmonks(environ={"SPORTMONKS_API_TOKEN": TOKEN,
                                      "V45_SPORTMONKS_FIXTURE_ID": "1234"}, request=get)
    assert report["requests_this_run"] == len(calls)
    assert TOKEN not in json.dumps(report)
    return report, calls


@pytest.mark.parametrize(("fixture_id", "environ", "status"), [
    (None, {"SPORTMONKS_API_TOKEN": TOKEN}, "fixture_not_configured"),
    ("../leagues", {"SPORTMONKS_API_TOKEN": TOKEN}, "invalid_fixture_id"),
    (0, {"SPORTMONKS_API_TOKEN": TOKEN}, "invalid_fixture_id"),
    (1234, {}, "sportmonks_token_missing"),
])
def test_probe_requires_sample_id_and_token(fixture_id, environ, status):
    def unexpected_request(*args, **kwargs):
        pytest.fail("Probe issued a request without a sample and token")

    report = probe_sportmonks(fixture_id=fixture_id, environ=environ, request=unexpected_request)
    assert report["status"] == status
    assert report["requests_this_run"] == 0


@pytest.mark.parametrize("field", ["xgfixture", "expected"])
def test_fixture_pair_proves_access_and_omits_extra_feed_request(field):
    report, calls = sample([
        (200, {"data": [{"id": 8}]}),
        (200, {"data": FIXTURE}),
        (200, {"data": {**FIXTURE, field: XG_ROWS}}),
    ])
    assert report["fixture_access"]["league_id"] == 8
    assert report["fixture_xg"]["status"] == "available"
    assert report["xg_access"] == "observed"
    assert calls == [("/leagues", {"per_page": 1}), ("/fixtures/1234", None),
                     ("/fixtures/1234", {"include": "participants;xGFixture"})]


def test_empty_fixture_does_not_obscure_access_elsewhere():
    report, calls = sample([
        (200, {"data": [{"id": 8}]}), (200, {"data": FIXTURE}),
        (200, {"data": {**FIXTURE, "xgfixture": []}}), (200, {"data": XG_ROWS[:1]}),
    ])
    assert report["fixture_xg"]["status"] == "xg_missing"
    assert report["team_xg_feed"]["status"] == "available"
    assert report["xg_access"] == "observed"
    assert calls[-1][0] == "/expected/fixtures"


def test_competing_metrics_and_empty_feed_cannot_prove_xg_entitlement():
    wrong_type = [{**row, "type_id": 5305} for row in XG_ROWS]
    report, _ = sample([
        (200, {"data": [{"id": 8}]}), (200, {"data": FIXTURE}),
        (200, {"data": {**FIXTURE, "xgfixture": wrong_type}}), (200, {"data": []}),
    ])
    assert report["fixture_xg"]["status"] == "xg_pair_unverified"
    assert report["team_xg_feed"]["status"] == "xg_not_observed"
    assert report["xg_access"] == "unconfirmed"


def test_xg_forbidden_is_distinct_from_normal_fixture_access():
    steps = [(200, {"data": [{"id": 8}]}), (200, {"data": FIXTURE}),
             (403, {"error": TOKEN}), (403, {"error": TOKEN})]
    report, _ = sample(steps)
    assert report["fixture_access"]["status"] == "available"
    assert report["fixture_xg"] == {"status": "subscription_denied", "http_status": 403}
    assert report["xg_access"] == "restricted"


def test_fixture_not_found_does_not_fail_account_or_xg_feed_access():
    report, _ = sample([
        (200, {"data": [{"id": 8}]}), (404, {}), (200, {"data": XG_ROWS}),
    ])
    assert report["fixture_access"]["status"] == "resource_not_found"
    assert report["fixture_xg"]["status"] == "not_checked"
    assert report["xg_access"] == "observed"


@pytest.mark.parametrize(("reply", "expected"), [
    ((401, {}), "authentication_failed"),
    ((403, {}), "subscription_denied"),
    ((429, {}), "rate_limited"),
    ((503, {}), "provider_server_error"),
    ((302, {}), "unexpected_http_status"),
    ((200, ValueError(TOKEN)), "invalid_response"),
    ((200, {"message": f"No data found. Invalid token {TOKEN}", "data": []}),
     "provider_access_or_subscription"),
    (requests.ConnectionError(TOKEN), "network_error"),
])
def test_base_request_outcomes_do_not_fall_through_to_multiple_calls(reply, expected):
    report, _ = sample([reply])
    assert report["status"] == expected
    assert report["fixture_access"]["status"] == "not_checked"
    assert report["requests_this_run"] == 1


def test_rate_limit_on_fixture_stops_before_optional_feed():
    report, _ = sample([(200, {"data": [{"id": 8}]}), (429, {})])
    assert report["fixture_access"]["status"] == "rate_limited"
    assert report["team_xg_feed"]["status"] == "not_checked"
    assert report["requests_this_run"] == 2


def test_archived_run_and_last_provider_observation_remain_distinct(tmp_path):
    archive = tmp_path / "sportmonks_xg"
    archive.mkdir()
    at = datetime(2026, 11, 28, tzinfo=timezone.utc)
    last_run = {"at_utc": at.isoformat(), "reason": "daily_request_budget_exhausted",
                "requests_this_run": 0, "daily_request_limit": 4, "daily_requests_used": 4}
    observation = {"at_utc": at.isoformat(), "reason": "provider_access_or_subscription",
                   "requests_this_run": 1, "captured": 0, "raw_provider_data": TOKEN}
    (archive / "status.json").write_text(json.dumps(last_run))
    (archive / "last_provider_observation.json").write_text(json.dumps(observation))
    report = archived_health(tmp_path)
    assert report["latest_run"]["reason"] == last_run["reason"]
    assert report["last_provider_observation"]["reason"] == observation["reason"]
    assert report["last_provider_observation"]["requests_this_run"] == 1
    assert report["requests_this_run"] == 0
    assert TOKEN not in json.dumps(report)
