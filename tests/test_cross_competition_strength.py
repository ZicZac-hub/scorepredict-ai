from datetime import datetime, timedelta, timezone
import numpy as np
import pytest
from cross_competition_strength import research

NOW = datetime(2031, 10, 1, tzinfo=timezone.utc)
FIXTURE = {"provider": "API-Football", "id": 9999, "utcDate": (NOW+timedelta(hours=1)).isoformat(),
           "homeTeam": {"id": 1}, "awayTeam": {"id": 2},
           "competition": {"id": 2, "code": "CL", "season": 2031}}
P = np.outer([.4,.3,.2,.1], [.4,.3,.2,.1])


def league(team, code, league_id, start, strong):
    rows = []
    for i in range(60):
        opponent = start + i % 9
        h, a = (team, opponent) if i % 2 == 0 else (opponent, team)
        hg, ag = ((3, 0) if h == team else (0, 3)) if strong else (1, 1)
        rows.append({"provider": "API-Football", "id": start*100+i,
            "utcDate": (NOW-timedelta(days=i+1)).isoformat(),
            "homeTeam": {"id": h}, "awayTeam": {"id": a},
            "competition": {"id": league_id, "code": code, "season": 2031},
            "status": "FINISHED", "score": {"duration": "REGULAR",
                "fullTime": {"home": hg, "away": ag}}})
    # Include unrelated matches for a league population.
    for i in range(60):
        rows.append({**rows[i], "id": start*100+60+i,
            "homeTeam": {"id": start+i%9}, "awayTeam": {"id": start+(i+1)%9},
            "score": {"duration": "REGULAR", "fullTime": {"home": 1, "away": 1}}})
    return rows


def cache():
    meta = {"freshness_known": True, "stale": False, "fetched_at_utc": NOW.isoformat()}
    return {("API-Football","39",2031): (league(1,"PL",39,100,True),dict(meta),None),
            ("API-Football","140",2031): (league(2,"PD",140,200,False),dict(meta),None)}


def test_cross_competition_normalized_and_unchanged_control():
    before = P.copy()
    result = research(FIXTURE,P,cache(),as_of=NOW)
    assert result["status"] == "research_candidate"
    q = np.array(result["challenger"])
    assert q.shape == P.shape and np.isclose(q.sum(),1) and np.all(q>=0)
    assert np.array_equal(P,before) and np.array_equal(result["control"],P)
    assert result["home"]["competition"] == "PL" and result["away"]["competition"] == "PD"
    assert result["home_tilt"] >= result["away_tilt"]
    assert result["production_promoted"] is False
    assert result["absolute_cross_league_strength_estimated"] is False


@pytest.mark.parametrize("mode", ["missing","stale","future_observation","ambiguous","after_kickoff"])
def test_unqualified_context_abstains(mode):
    source = cache()
    k = ("API-Football","140",2031)
    cutoff = NOW
    if mode == "missing":
        source.pop(k)
    elif mode == "stale":
        source[k][1]["stale"] = True
    elif mode == "future_observation":
        source[k][1]["fetched_at_utc"] = (NOW+timedelta(minutes=1)).isoformat()
    elif mode == "ambiguous":
        source[("API-Football","78",2031)] = (league(2,"BL1",78,300,False),dict(source[k][1]),None)
    else:
        cutoff = NOW+timedelta(hours=2)
    assert research(FIXTURE,P,source,as_of=cutoff)["status"] == "abstained"


def test_future_results_and_conflicting_duplicate_do_not_change_control():
    source = cache()
    first = research(FIXTURE,P,source,as_of=NOW)
    rows = source[("API-Football","39",2031)][0]
    rows.append({**rows[0], "id": 55555, "utcDate": (NOW+timedelta(minutes=5)).isoformat()})
    assert research(FIXTURE,P,source,as_of=NOW) == first
    with pytest.raises(ValueError,match="invalid_european_control"):
        research(FIXTURE,P*2,source,as_of=NOW)
