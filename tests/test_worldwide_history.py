"""Worldwide league populations must not collapse to the two selected teams."""
from datetime import datetime, timezone
from unittest.mock import patch

from worldwide_model import model_worldwide


class ObservedDatetime(datetime):
    @classmethod
    def now(cls, tz=None):
        return cls(2026, 1, 10, tzinfo=timezone.utc)


def raw_match(identifier, home=1, away=3, *, league=39, season=2025,
              day="2025-12-01T12:00:00Z", status="FT"):
    return {"fixture": {"id": identifier, "date": day, "status": {"short": status}},
            "teams": {"home": {"id": home, "name": f"Team {home}"},
                      "away": {"id": away, "name": f"Team {away}"}},
            "league": {"id": league, "season": season, "name": "Test league"},
            "goals": {"home": 2, "away": 1},
            "score": {"fulltime": {"home": 2, "away": 1}}}


def event():
    return {"league": "Premier League", "league_id": 39, "season": 2025,
            "fixture": {"id": 1000, "utcDate": "2026-01-20T12:00:00Z",
                        "provider": "API-Football", "competition": {"id": 39, "code": "PL"},
                        "homeTeam": {"id": 1, "name": "Team 1"},
                        "awayTeam": {"id": 2, "name": "Team 2"}}}


def population(season=2025):
    # Both target teams have ample history. The other clubs must also remain
    # in the population used to estimate the league's scoring environment.
    return ([raw_match(i, 1, 3, season=season) for i in range(1, 11)]
            + [raw_match(i, 4, 2, season=season) for i in range(11, 21)]
            + [raw_match(21, 5, 6, season=season)])


def run_model(request, match_event=None):
    with patch("worldwide_model.datetime", ObservedDatetime), \
         patch("worldwide_model._request", side_effect=request) as requests, \
         patch("worldwide_model.predict", return_value={"model_version": "test"}) as prediction:
        row, reason = model_worldwide("test-key", match_event or event())
    return row, reason, prediction, requests


def test_complete_league_retains_other_teams_and_excludes_wrong_or_future_rows():
    all_rows = population() + [
        raw_match(30, 5, 6, league=140),
        raw_match(31, 5, 6, season=2024),
        raw_match(32, 5, 6, day="2026-01-15T12:00:00Z"),  # future at capture
        raw_match(33, 5, 6, day="2026-01-20T12:00:00Z"),  # same kickoff
        raw_match(34, 5, 6, status="NS"),
    ]

    def request(_key, endpoint, params):
        return all_rows if endpoint == "/fixtures" and params == {"league": 39, "season": 2025} else []

    row, reason, prediction, requests = run_model(request)
    assert reason is None
    league_history = prediction.call_args.kwargs["league_history"]
    assert {m["id"] for m in league_history} == set(range(1, 22))
    assert any(m["homeTeam"]["id"] == 5 and m["awayTeam"]["id"] == 6 for m in league_history)
    assert all(m["competition"]["id"] == 39 and m["competition"]["code"] == "PL" for m in league_history)
    assert row["history_counts"] == {"home": 10, "away": 10, "league": 21}
    fixture_calls = [c.args[2] for c in requests.call_args_list if c.args[1] == "/fixtures"]
    assert fixture_calls == [{"league": 39, "season": 2025}]


def test_sparse_current_season_uses_one_previous_complete_season():
    def request(_key, endpoint, params):
        if endpoint != "/fixtures":
            return []
        if params == {"league": 39, "season": 2025}:
            return [raw_match(100, 1, 2)]
        if params == {"league": 39, "season": 2024}:
            return population(2024)
        return []

    row, reason, prediction, requests = run_model(request)
    assert reason is None
    assert len(prediction.call_args.kwargs["league_history"]) == 22
    assert row["history_counts"]["home"] == 11
    seasons = [c.args[2]["season"] for c in requests.call_args_list
               if c.args[1] == "/fixtures" and "league" in c.args[2]]
    assert seasons == [2025, 2024]


def test_team_fallback_never_becomes_a_league_population():
    def request(_key, endpoint, params):
        if endpoint != "/fixtures":
            return []
        if "league" in params:
            raise RuntimeError("league season unavailable")
        if params.get("team") == 1:
            return [raw_match(i, 1, 3) for i in range(1, 6)]
        if params.get("team") == 2:
            return [raw_match(i, 4, 2) for i in range(6, 11)]
        return []

    row, reason, prediction, _requests = run_model(request)
    assert reason is None
    assert prediction.call_args.kwargs["league_history"] == []
    assert row["history_counts"] == {"home": 5, "away": 5, "league": 0}
    assert any("generic priors" in warning for warning in row["research_warnings"])


def test_cup_results_do_not_suppress_target_competition_date_fallback():
    def request(_key, endpoint, params):
        if endpoint != "/fixtures" or "league" in params:
            return []
        tid = params.get("team")
        if tid not in (1, 2):
            return []
        if params.get("last"):
            return [raw_match(100 + tid * 10 + i, tid, 8, league=2) for i in range(10)]
        if "from" in params:
            return [raw_match(tid * 10 + i, tid, 8) for i in range(5)]
        return []

    row, reason, prediction, requests = run_model(request)
    assert reason is None
    assert row["history_counts"] == {"home": 5, "away": 5, "league": 0}
    assert len([c for c in requests.call_args_list if "from" in c.args[2]]) == 2
    assert any(m["competition"]["id"] == 2 for m in prediction.call_args.args[1])
    assert prediction.call_args.kwargs["league_history"] == []


def test_fixture_metadata_can_supply_league_and_season_without_event_duplicates():
    match_event = event()
    del match_event["league_id"]
    del match_event["season"]
    match_event["fixture"]["competition"]["season"] = 2025

    def request(_key, endpoint, params):
        return population() if endpoint == "/fixtures" and params == {"league": 39, "season": 2025} else []

    row, reason, prediction, _requests = run_model(request, match_event)
    assert reason is None
    assert len(prediction.call_args.kwargs["league_history"]) == 21
    assert row["fixture"]["competition"]["season"] == 2025


def test_real_engine_receives_complete_population():
    def request(_key, endpoint, params):
        return population() if endpoint == "/fixtures" and params == {"league": 39, "season": 2025} else []

    with patch("worldwide_model.datetime", ObservedDatetime), \
         patch("worldwide_model._request", side_effect=request):
        row, reason = model_worldwide("test-key", event())
    assert reason is None
    assert row["result"]["league_history_count"] == 21
    assert row["result"]["league_baseline_source"] == "competition_history"
    assert row["result"]["history_count"] == 21
    assert abs(float(row["result"]["matrix"].sum()) - 1.0) < 1e-10


def test_real_engine_refuses_cup_only_histories_for_league_fixture():
    def request(_key, endpoint, params):
        tid = params.get("team")
        if endpoint == "/fixtures" and tid in (1, 2):
            return [raw_match(100 + tid * 10 + i, tid, 8, league=2) for i in range(10)]
        return []

    with patch("worldwide_model.datetime", ObservedDatetime), \
         patch("worldwide_model._request", side_effect=request):
        row, reason = model_worldwide("test-key", event())
    assert row is None
    assert "home team 0, away team 0" in reason
