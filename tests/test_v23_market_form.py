"""V23 real-time provenance, market-only fallback, deterministic score changes."""
import hashlib
import json
from datetime import datetime,timedelta,timezone

import numpy as np
from v21_goal_model_comparison import _poisson
from v22_evidence_models import parse_markets
from v23_market_form import shadow

NOW=datetime(2032,5,1,8,tzinfo=timezone.utc)
KICK=(NOW+timedelta(hours=3)).isoformat()
BASE=np.outer(_poisson(1.15,11),_poisson(1.2,11))
FIX={"id":11011,"provider":"API-Football","utcDate":KICK,
     "homeTeam":{"id":11},"awayTeam":{"id":22},
     "competition":{"id":39,"code":"PL"}}


def odds(*,updated=True,high=True):
    row={"bookmakers":[{"name":"Source A","bets":[
        {"name":"Match Winner","values":[
            {"value":"Home","odd":"1.58"},
            {"value":"Draw","odd":"4.1"},
            {"value":"Away","odd":"5.6"}]},
        {"name":"Goals Over/Under","values":[
            {"value":"Over 2.5","odd":"1.48" if high else "2.6"},
            {"value":"Under 2.5","odd":"2.6" if high else "1.5"}]}]}]}
    if updated:
        row["update"]=(NOW-timedelta(minutes=25)).isoformat()
    return [row]


def pack(*,high=True,market=True,old_market=False):
    quotes=parse_markets(odds(high=high),observed_at=NOW)
    if old_market:
        for record in quotes.values():
            record["last_source_update_utc"]=(NOW-timedelta(days=2)).isoformat()
    pkt={"status":"captured_pre_kickoff","schema":1,
         "provider":"API-Football","fixture_id":"11011",
         "home_team_id":"11","away_team_id":"22",
         "kickoff_utc":KICK,"captured_at_utc":NOW.isoformat(),
         "verified_historical_fixture_xg":[],"confirmed_starting_xi":[],
         "multi_market":quotes if market else {}}
    pkt["snapshot_sha256"]=hashlib.sha256(json.dumps(pkt,sort_keys=True,separators=(",",":"),
                                                allow_nan=False).encode()).hexdigest()
    return pkt


def row():
    return {"fixture":FIX,"result":{"matrix":BASE,"model_version":"v8"},
            "league_history_matches":120}


def test_provenance_of_quote_fetch_without_bookmaker_timestamp():
    assert parse_markets(odds(updated=False),observed_at=NOW)=={}
    confirmed=parse_markets(
        odds(updated=False),observed_at=NOW,
        verified_provider_fetched_at=(NOW-timedelta(minutes=10)).isoformat())
    assert set(confirmed)=={"1x2","over_2_5"}
    assert all(q["timestamp_basis"]=="api_response_fetched_at"
               for m in confirmed.values() for q in m["quotes"])
    assert parse_markets(odds(updated=False),observed_at=NOW,
        verified_provider_fetched_at=(NOW-timedelta(hours=20)).isoformat())=={}


def test_market_only_fallback_is_real_prediction_and_tracks_higher_goals():
    result=shadow(row(),pack(high=True),as_of=NOW)
    assert result["ready"]
    assert result["model"]=="v23_market_form_shadow_not_validated"
    assert "unchanged_original_goal_prior" in result["applied_signals"]
    assert "genuine_historical_shot_xg" not in result["applied_signals"]
    assert result["adjusted_over_2_5_probability"]>result["prior_over_2_5_probability"]
    assert len(result["scores"])==10
    assert abs(result["matrix"].sum()-1)<1e-9
    assert result["home_xg"]>0 and result["away_xg"]>0
    assert not result["proven_accuracy_gain"]


def test_market_can_lower_high_score_probability_no_3_2_forcing():
    low=shadow(row(),pack(high=False),as_of=NOW)
    high=shadow(row(),pack(high=True),as_of=NOW)
    assert low["ready"] and high["ready"]
    assert high["adjusted_over_2_5_probability"]>low["adjusted_over_2_5_probability"]


def test_abstains_without_real_market_quotes_or_freshness_or_identity():
    assert not shadow(row(),pack(market=False),as_of=NOW)["ready"]
    assert not shadow(row(),pack(old_market=True),as_of=NOW)["ready"]
    wrong=pack()
    wrong["home_team_id"]="99"
    assert not shadow(row(),wrong,as_of=NOW)["ready"]
    only={**row(),"league_history_matches":15}
    assert shadow(only,pack(),as_of=NOW)["reason"]=="insufficient_league_model_history"
