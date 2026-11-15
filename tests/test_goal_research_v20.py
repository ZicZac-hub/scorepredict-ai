"""V20 regression tests: real historical goal research, no score forcing."""
from datetime import datetime, timedelta, timezone

import numpy as np

from goal_research_v20 import goal_profile, goal_form_challenger


NOW = datetime(2031, 9, 1, 12, tzinfo=timezone.utc)
FIXTURE = {"id": 1001, "provider": "API-Football",
           "utcDate": (NOW+timedelta(days=1)).isoformat(),
           "competition": {"id": 71, "code": "BSA"},
           "homeTeam": {"id": 10, "name": "Home Strong"},
           "awayTeam": {"id": 20, "name": "Away Strong"}}


def played(days_ago, home, away, hg, ag, *, provider="API-Football", league=71):
    return {"status": "FINISHED", "provider": provider,
            "id": days_ago*100+home, "competition": {"id": league},
            "utcDate": (NOW-timedelta(days=days_ago)).isoformat(),
            "homeTeam": {"id": home, "name": str(home)},
            "awayTeam": {"id": away, "name": str(away)},
            "score": {"duration": "REGULAR", "fullTime": {"home": hg, "away": ag}}}


def history(high=False):
    matches = []
    for n in range(1, 11):
        matches.append(played(n, 10, 30+n, 3 if high else 0, 1))
        matches.append(played(n, 40+n, 20, 0, 2 if high else 1))
    return matches


def base():
    from goal_research_v20 import _poisson
    matrix=np.outer(_poisson(1.3, 11), _poisson(1.1, 11))
    return {"fixture": FIXTURE, "result": {"matrix": matrix, "scores": [[1, 1, .1]]}}


def test_profile_uses_real_home_away_history_not_future_or_other_provider():
    matches=history(high=True)
    matches += [played(1, 10, 999, 9, 0, provider="another"),
                played(1, 10, 999, 9, 0, league=999),
                {**played(1, 10, 999, 9, 0),
                 "utcDate": (NOW+timedelta(hours=2)).isoformat()}]
    profile=goal_profile(FIXTURE, matches, NOW)
    assert profile["home_last10"]["scored"] == 3
    assert profile["home_at_home_last10"]["matches"] == 10
    assert profile["away_away_last10"]["scored"] == 2
    assert profile["h2h_last6"] == []


def test_real_goal_evidence_can_shift_but_does_not_force_large_scores():
    high=goal_form_challenger(base(),history(True),NOW)
    low=goal_form_challenger(base(),history(False),NOW)
    assert high["available"] and low["available"]
    assert high["research_expected_goals"][0] > low["research_expected_goals"][0]
    assert high["research_expected_goals"][1] > low["research_expected_goals"][1]
    assert high["four_plus_goal_probability_research"] > low["four_plus_goal_probability_research"]
    assert len(high["scores"]) == 10
    assert np.isclose(high["matrix"].sum(),1.0)
    assert high["proven_accuracy_improvement"] is False


def test_no_research_score_without_sufficient_evidence_or_full_grid():
    missing=goal_form_challenger(base(),[],NOW)
    assert not missing["available"]
    row=base()
    row["result"].pop("matrix")
    archived=goal_form_challenger(row,history(True),NOW)
    assert not archived["available"]
    assert "complete original score matrix" in archived["reason"]


def test_future_results_never_enter_profile():
    future=played(1,10,30,9,1)
    future["utcDate"]=(NOW+timedelta(hours=2)).isoformat()
    p=goal_profile(FIXTURE,[future],NOW)
    assert p["home_last10"]["matches"] == 0
    assert p["away_last10"]["matches"] == 0


def test_h2h_from_home_perspective_no_double_count():
    matches=history(True)
    matches.append(played(15,20,10,2,3))
    p=goal_profile(FIXTURE,matches,NOW)
    assert p["h2h_last6"][0]["home_goals"]==3
    assert p["h2h_last6"][0]["away_goals"]==2
