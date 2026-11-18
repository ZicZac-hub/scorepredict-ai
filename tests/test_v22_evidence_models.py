"""V22 market, xG, lineup integrity and abstention tests (no API calls)."""
import hashlib
import json
from datetime import datetime,timedelta,timezone

import numpy as np
import pytest
from v21_goal_model_comparison import _poisson
from v22_evidence_models import (
    parse_markets,xg_profile,lineup_status,calibrate_markets,combine_goal_evidence)
from v22_shadow import shadow

NOW=datetime(2032,1,20,11,tzinfo=timezone.utc)
FIXTURE={"id":999,"provider":"API-Football","utcDate":(NOW+timedelta(hours=3)).isoformat(),
         "homeTeam":{"id":11},"awayTeam":{"id":22}}
GRID=np.outer(_poisson(1.9,11),_poisson(1.1,11))


def market_response(hours=1):
    updated=(NOW-timedelta(hours=hours)).isoformat()
    return [{"update":updated,"bookmakers":[{"name":"Book A","bets":[
       {"name":"Match Winner","values":[{"value":k,"odd":v} for k,v in
           (("Home",2.0),("Draw",3.8),("Away",4.1))]},
       {"name":"Goals Over/Under","values":[{"value":k,"odd":v} for k,v in
           (("Over 2.5",1.65),("Under 2.5",2.2),("Over 3.5",2.5),("Under 3.5",1.54))]},
       {"name":"Both Teams Score","values":[{"value":"Yes","odd":"1.66"},
                                           {"value":"No","odd":"2.21"}]}]}]}]


def make_packet(xg_count=8,venue=True):
    rows=[]
    for team in (11,22):
        for n in range(xg_count):
            home,away=(team,30+n+team) if ((n%2==0) and venue) else (35+n+team,team)
            rows.append({"fixture_id":f"{team}-{n}",
                        "kickoff_utc":(NOW-timedelta(days=n+1)).isoformat(),
                        "home_team_id":str(home),"away_team_id":str(away),
                        "home_xg":2.2 if home==team else 1.1,
                        "away_xg":1.6 if away==team else .8})
    packet={"schema":1,"status":"captured_pre_kickoff","provider":"API-Football",
            "fixture_id":"999","home_team_id":"11","away_team_id":"22",
            "kickoff_utc":FIXTURE["utcDate"],"captured_at_utc":NOW.isoformat(),
            "verified_historical_fixture_xg":rows,"multi_market":parse_markets(
                market_response(),observed_at=NOW),
            "confirmed_starting_xi":[{"team_id":"11","formation":"4-3-3",
                    "starting_xi":[f"player {i}" for i in range(11)]},
                   {"team_id":"22","formation":"4-4-2",
                    "starting_xi":[f"opponent {i}" for i in range(11)]}],
            "reported_injuries":[]}
    packet["snapshot_sha256"]=hashlib.sha256(
        json.dumps(packet,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
    return packet


def test_timestamped_multimarket_normalizes_without_mixing_books():
    prices=parse_markets(market_response(),observed_at=NOW)
    assert set(prices)=={"1x2","over_2_5","over_3_5","btts_yes"}
    for market in prices.values():
        assert sum(market["probabilities"].values())==pytest.approx(1,abs=1e-6)
        assert market["n_bookmakers"]==1
    assert parse_markets(market_response(hours=30),observed_at=NOW)=={}


def test_real_pre_match_xg_requires_sample_and_venue_history():
    p=make_packet()
    profile=xg_profile(FIXTURE,p,as_of=NOW)
    assert profile["ready"] and profile["home_count"]==8
    assert profile["home_venue_count"]==4
    few=xg_profile(FIXTURE,make_packet(xg_count=2),as_of=NOW)
    assert few["ready"] is False and "insufficient" in few["reason"]
    bad=dict(p,home_team_id="incorrect")
    assert not xg_profile(FIXTURE,bad,as_of=NOW)["ready"]


def test_lineup_is_context_only_not_untrained_numeric_effect():
    status=lineup_status(FIXTURE,make_packet(),as_of=NOW)
    assert status["ready"] and not status["numeric_player_coefficient_fitted"]


@pytest.mark.parametrize("case",["same_team_twice","conflicting_team","repeated_player"])
def test_two_lineup_rows_do_not_prove_two_complete_confirmed_teams(case):
    packet=make_packet()
    first=packet["confirmed_starting_xi"][0]
    if case=="same_team_twice":
        packet["confirmed_starting_xi"]=[first,first]
    elif case=="conflicting_team":
        changed={**first,"formation":"3-5-2"}
        packet["confirmed_starting_xi"].append(changed)
    else:
        packet["confirmed_starting_xi"][1]["starting_xi"]=["Repeated player"]*11
    packet.pop("snapshot_sha256")
    packet["snapshot_sha256"]=hashlib.sha256(json.dumps(
        packet,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
    status=lineup_status(FIXTURE,packet,as_of=NOW)
    assert status["ready"] is False
    assert len({team["team_id"] for team in status["teams"]})<2


def test_calibration_keeps_grid_and_caps_market_change():
    markets=make_packet()["multi_market"]
    result=calibrate_markets(GRID,markets)
    assert result["ready"]
    assert result["matrix"].sum()==pytest.approx(1)
    assert result["total_variation"]<=.1500001
    assert np.all(result["matrix"]>=0)
    assert not calibrate_markets(GRID,{})["ready"]


def test_shadow_abstains_without_real_evidence_or_odds():
    base={"fixture":FIXTURE,"result":{"matrix":GRID}}
    assert not shadow(base,make_packet(xg_count=2),as_of=NOW)["ready"]
    ready=shadow(base,make_packet(),as_of=NOW)
    assert ready["ready"]
    assert ready["prediction_method"].startswith("v22")
    assert ready["numerical_lineup_adjustment"]==0
    assert len(ready["scores"])==10
    packet=make_packet()
    packet["multi_market"]={}
    packet["snapshot_sha256"]=hashlib.sha256(json.dumps(
        {k:v for k,v in packet.items() if k!="snapshot_sha256"},sort_keys=True,
        separators=(",",":"),allow_nan=False).encode()).hexdigest()
    assert shadow(base,packet,as_of=NOW)["ready"] is False
