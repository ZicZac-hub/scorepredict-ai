from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json

import pytest

import v45_lineups as lineups


START = datetime(2032, 8, 17, 17, 30, tzinfo=timezone.utc)
KICKOFF = START+timedelta(minutes=30)
FORECAST = {"provider": "API-Football", "fixture_id": "11", "home_team_id": "21",
            "away_team_id": "22", "home_team": "Home Town", "away_team": "Away Town",
            "kickoff_utc": KICKOFF.isoformat(), "model_key": "v8_near",
            "forecast_id": "frozen-original", "input_digest": "test-only",
            "captured_at_utc": (START-timedelta(hours=5)).isoformat()}
ENV = {"V45_LINEUPS_ENABLED": "true", "SPORTMONKS_API_TOKEN": "sportmonks-test-secret",
       "API_FOOTBALL_KEY": "api-test-secret"}


def sportmonks_fixture():
    return {"id": 90, "starting_at_timestamp": int(KICKOFF.timestamp()), "state_id": 1,
            "participants": [
                {"id": 91, "meta": {"location": "home"}, "name": "Home Town"},
                {"id": 92, "meta": {"location": "away"}, "name": "Away Town"}],
            "metadata": [{"type_id": 572, "values": {"confirmed": True}}],
            "lineups": [
                {"fixture_id": 90, "team_id": 91+side, "player_id": 300+side*11+i,
                 "type_id": 11, "player_name": f"Example Player {side}-{i}"}
                for side in range(2) for i in range(11)]}


class Response:
    status_code = 200

    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


def collect(tmp_path, *, sportmonks=None, change_api=None, env=ENV, now=START, before_sm=None,
            embedded=True):
    fixture = sportmonks_fixture() if sportmonks is None else sportmonks
    clock = [now]
    calls = []

    def api_request(key, endpoint, params, *, with_metadata):
        clock[0] += timedelta(seconds=1)
        calls.append(endpoint)
        assert key == ENV["API_FOOTBALL_KEY"] and with_metadata
        xis = [{"team": {"id": 21+side}, "startXI": [
            {"player": {"id": 700+side*11+i, "name": f"example player {side} {i}"}}
            for i in reversed(range(11))]} for side in range(2)]
        if endpoint == "/fixtures":
            assert params == {"id": "11"}
            data = [{"fixture": {"id": 11, "date": KICKOFF.isoformat(), "status": {"short": "NS"}},
                     "teams": {"home": {"id": 21}, "away": {"id": 22}}}]
            if embedded:
                data[0]["lineups"] = xis
        else:
            assert params == {"fixture": "11"}
            data = xis
        envelope = {"data": data, "metadata": {"provider": "API-Football", "stale": False,
                    "fetched_at_utc": clock[0].isoformat()}}
        if change_api:
            change_api(endpoint, envelope)
        return envelope

    def sm_request(url, *, params, headers, timeout, allow_redirects):
        clock[0] += timedelta(seconds=1)
        calls.append(url)
        assert headers["Authorization"] == ENV["SPORTMONKS_API_TOKEN"] and not allow_redirects
        if "/date/" in url:
            return Response({"data": [fixture], "pagination": {"has_more": False}})
        assert url.endswith("/fixtures/90")
        assert params == {"include": "participants;lineups;metadata"}
        if before_sm:
            clock[0] = before_sm
        return Response({"data": fixture})

    report = lineups.collect(root=tmp_path, now=now, clock=lambda: clock[0], environ=env,
                             originals=[FORECAST], sportmonks_request=sm_request,
                             api_football_request=api_request)
    return report, calls, clock[0]


def test_collects_corroborated_xis_with_source_time_and_redacted_receipt(tmp_path, monkeypatch):
    private = {}

    def storage(url, receipt, names):
        private.update(url=url, receipt=receipt, names=names)
        return "saved"

    monkeypatch.setattr(lineups, "_private_archive", storage)
    report, calls, finished = collect(tmp_path, env={**ENV, "DATABASE_URL": "private-test-dsn"})
    assert report["reason"] == "verified"
    assert report["saved_receipt"] and report["private_archive"] == "saved"
    assert report["requests_this_run"] == len(calls) == 3
    assert len(private["names"]["Sportmonks"]["away"]) == 11
    record_path = next((tmp_path/"lineups"/"receipts").rglob("*.json"))
    record_text = record_path.read_text()
    assert "player" not in record_text and "secret" not in record_text and "private-test-dsn" not in record_text
    receipt = lineups.load(record_path, FORECAST, cutoff=finished)
    assert receipt["sources"]["Sportmonks"]["fixture_id"] == "90"
    assert receipt["sources"]["API-Football"]["fixture_id"] == "11"
    assert receipt["agreement_by_side"] == {"home": True, "away": True}
    assert lineups.status_for(FORECAST, root=tmp_path, cutoff=START)["status"] == "missing"
    assert lineups.status_for(FORECAST, root=tmp_path, cutoff=finished)["status"] == "verified"
    assert lineups.status_for({**FORECAST, "away_team_id": "12"}, root=tmp_path,
                              cutoff=finished)["status"] == "missing"
    record_path.write_text(record_text.replace('"status":"verified"', '"status":"confirmed"'))
    with pytest.raises(ValueError, match="invalid_lineup_receipt"):
        lineups.load(record_path, FORECAST, cutoff=finished)


@pytest.mark.parametrize("damage,expected", [("unconfirmed", "lineups_not_confirmed"),
                                               ("duplicate", "incomplete_or_duplicate_starting_xis")])
def test_requires_confirmation_and_distinct_starters(tmp_path, damage, expected):
    fixture = sportmonks_fixture()
    if damage == "unconfirmed":
        fixture["metadata"][0]["values"]["confirmed"] = False
    else:
        fixture["lineups"][10] = deepcopy(fixture["lineups"][0])
    report, _, finished = collect(tmp_path, sportmonks=fixture)
    assert report["providers"]["Sportmonks"] == expected
    evidence = lineups.status_for(FORECAST, root=tmp_path, cutoff=finished)
    assert evidence["status"] == "unconfirmed"
    assert set(evidence["sources"]) == {"API-Football"}


def test_disagreement_and_stale_or_rescheduled_api_replies(tmp_path):
    def disagree(endpoint, envelope):
        if endpoint.endswith("lineups"):
            envelope["data"][0]["startXI"][0]["player"]["name"] = "Different Starter"

    report, calls, finished = collect(tmp_path/"conflict", change_api=disagree, embedded=False)
    assert report["reason"] == "conflict"
    assert calls[:2] == ["/fixtures", "/fixtures/lineups"]
    assert lineups.status_for(FORECAST, root=tmp_path/"conflict", cutoff=finished)[
        "agreement_by_side"] == {"home": False, "away": True}

    def stale(endpoint, envelope):
        if endpoint.endswith("lineups"):
            envelope["metadata"]["stale"] = True

    report, _, _ = collect(tmp_path/"stale", change_api=stale, embedded=False)
    assert report["reason"] == "confirmed"
    assert report["providers"]["API-Football"] == "lineup_response_missing_or_stale"

    def rescheduled(endpoint, envelope):
        if endpoint == "/fixtures":
            envelope["data"][0]["fixture"]["date"] = (KICKOFF+timedelta(hours=1)).isoformat()

    report, calls, _ = collect(tmp_path/"rescheduled", change_api=rescheduled)
    assert report["reason"] == "confirmed"
    assert report["providers"]["API-Football"] == "fixture_identity_or_state_mismatch"
    assert "/fixtures/lineups" not in calls


def test_does_not_freeze_after_kickoff(tmp_path):
    report, calls, _ = collect(tmp_path, before_sm=KICKOFF)
    assert report["reason"] == "collection_finished_after_kickoff"
    assert report["requests_this_run"] == len(calls) == 3
    assert not list((tmp_path/"lineups"/"receipts").rglob("*.json"))


def test_reservations_bound_retries_and_opt_in(tmp_path):
    report, calls, _ = collect(tmp_path, env={**ENV, "V45_LINEUPS_ENABLED": "false"})
    assert report["reason"] == "disabled" and not calls
    env = {**ENV, "V45_LINEUP_DAILY_REQUESTS": "4"}
    assert collect(tmp_path, env=env)[0]["reason"] == "verified"
    report, calls, _ = collect(tmp_path, env=env, now=START+timedelta(minutes=1))
    assert report["reason"] == "eligible_fixtures_already_checked_in_phase" and not calls
    report, calls, finished = collect(tmp_path, env=env, now=START+timedelta(minutes=15))
    assert report["daily_requests_used"] == 4 and len(calls) == 1
    assert report["reason"] == "unconfirmed"
    assert lineups.status_for(FORECAST, root=tmp_path, cutoff=finished)["status"] == "verified"
    report, calls, _ = collect(tmp_path, env=env, now=START+timedelta(minutes=20))
    assert report["reason"] == "daily_request_budget_exhausted" and not calls
    status_file = tmp_path/"lineups"/"status.json"
    assert json.loads(status_file.read_text()) == report
