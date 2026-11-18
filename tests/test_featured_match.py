"""The displayed pre-match forecast stays frozen once the fixture starts."""
from datetime import datetime, timedelta, timezone

import pytest

import featured_match as featured

NOW = datetime(2032, 11, 20, 0, tzinfo=timezone.utc)


def row(fid=101, score=(1, 0), hours=1):
    return {"fixture": {"id": fid, "provider": "API-Football",
                        "utcDate": (NOW+timedelta(hours=hours)).isoformat(),
                        "homeTeam": {"name": "Home"}, "awayTeam": {"name": "Away"}},
            "kickoff": (NOW+timedelta(hours=hours)).isoformat(),
            "result": {"scores": [[*score, .15]], "home_win": .5,
                       "draw": .25, "away_win": .25}}


def test_live_feature_keeps_latest_displayed_prematch_score(monkeypatch):
    monkeypatch.setattr(featured, "fixture_state", lambda *args: ("active", None))
    state = {}
    first, _ = featured.choose_feature([row()], state, "test", now=NOW)
    assert first["result"]["scores"][0][:2] == [1, 0]
    refreshed, _ = featured.choose_feature([row(score=(2, 0))], state, "test",
                                          now=NOW+timedelta(minutes=10))
    assert refreshed["result"]["scores"][0][:2] == [2, 0]
    live, _ = featured.choose_feature([row(score=(9, 9))], state, "test",
                                     now=NOW+timedelta(hours=1, minutes=10))
    assert live["result"]["scores"][0][:2] == [2, 0]


def test_feature_snapshot_does_not_share_mutable_result_with_source(monkeypatch):
    monkeypatch.setattr(featured, "fixture_state", lambda *args: ("active", None))
    state, original = {}, row()
    featured.choose_feature([original], state, "test", now=NOW)
    original["result"]["scores"][0][:2] = [8, 8]
    assert state["featured_match"]["row"]["result"]["scores"][0][:2] == [1, 0]


def test_earlier_revised_kickoff_does_not_allow_postkickoff_replacement(monkeypatch):
    monkeypatch.setattr(featured, "fixture_state", lambda *args: ("active", None))
    state = {}
    featured.choose_feature([row(hours=2)], state, "test", now=NOW)
    changed = row(score=(9, 9), hours=1)
    result, _ = featured.choose_feature([changed], state, "test",
                                       now=NOW+timedelta(hours=1, minutes=10))
    assert result["result"]["scores"][0][:2] == [1, 0]


def test_verified_finish_advances_to_another_future_fixture(monkeypatch):
    state = {}
    monkeypatch.setattr(featured, "fixture_state", lambda *args: ("active", None))
    featured.choose_feature([row()], state, "test", now=NOW)
    state["featured_status"]["time"] = 0
    monkeypatch.setattr(featured, "fixture_state", lambda key, f:
                        ("finished", (1, 0)) if f["id"] == 101 else ("active", None))
    result, status = featured.choose_feature([row(102, hours=5)], state, "test",
                                             now=NOW+timedelta(hours=2))
    assert result["fixture"]["id"] == 102
    assert status["state"] == "active"


@pytest.mark.parametrize("status, expected", [("FINISHED", "finished"),
    ("POSTPONED", "cancelled"), ("CANCELLED", "cancelled"), ("TIMED", "unknown")])
def test_non_api_provider_status_is_respected(status, expected):
    fixture = {"provider": "football-data.org", "id": 1, "status": status}
    assert featured.fixture_state(None, fixture) == (expected, None)


def test_wrong_fixture_response_never_verifies_the_feature(monkeypatch):
    class Response:
        def raise_for_status(self):
            pass
        def json(self):
            return {"response": [{"fixture": {"id": 999,
                "status": {"short": "FT"}}, "goals": {"home": 2, "away": 0}}]}
    monkeypatch.setattr(featured.requests, "get", lambda *a, **k: Response())
    assert featured.fixture_state("test", row()["fixture"]) == ("unknown", None)


def test_fixture_only_date_is_supported_and_started_game_is_not_new_feature(monkeypatch):
    monkeypatch.setattr(featured, "fixture_state", lambda *args: ("active", None))
    candidate = row()
    candidate.pop("kickoff")
    picked, _ = featured.choose_feature([candidate], {}, "test", now=NOW)
    assert picked["fixture"]["id"] == candidate["fixture"]["id"]
    assert picked["kickoff"] == candidate["fixture"]["utcDate"]
    assert featured.choose_feature([candidate], {}, "test",
                                   now=NOW+timedelta(hours=1)) == (None, None)
