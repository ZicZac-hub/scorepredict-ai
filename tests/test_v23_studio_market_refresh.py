"""V23 mobile studio odds refresh: authentic evidence only, no forced bets."""
import hashlib
import json
from datetime import datetime, timedelta, timezone

from ai_five_double_ui import (
    select_v23_packet, v23_capture_needed, _source_fixture_key)
from v23_market_form import market_status, shadow
from v22_evidence_models import parse_markets
from v21_goal_model_comparison import _poisson
import numpy as np


NOW=datetime(2032,5,1,8,tzinfo=timezone.utc)
KICK=(NOW+timedelta(hours=3)).isoformat()
FIX={"id":444,"provider":"API-Football","utcDate":KICK,
     "homeTeam":{"id":11},"awayTeam":{"id":22}}
BASE=np.outer(_poisson(1.2,11),_poisson(1.1,11))


def packet(*,source_age_hours=0.5,captured_age_hours=0.15,
           has_totals=True, has_1x2=True, undated=False, tamper=False):
    update=NOW-timedelta(hours=source_age_hours)
    # Parse the provider quote at its genuine capture time; old quotes
    # may later expire at the *new* decision time without being fabricated.
    captured=NOW-timedelta(hours=captured_age_hours)
    parsed_quote_time=(NOW-timedelta(minutes=20)
                       if source_age_hours<0 else update)
    data={"bookmakers":[{"name":"Source A","bets":[
        {"name":"Match Winner","values":[
            {"value":"Home","odd":"2.1"},{"value":"Draw","odd":"3.2"},
            {"value":"Away","odd":"3.8"}]},
        {"name":"Goals Over/Under","values":[
            {"value":"Over 2.5","odd":"1.8"},
            {"value":"Under 2.5","odd":"2.05"}]}]}],
          "update":parsed_quote_time.isoformat()}
    odds=parse_markets([data],observed_at=captured)
    if source_age_hours<0:
        for entry in odds.values():
            entry["last_source_update_utc"]=update.isoformat()
    if not has_totals:
        odds.pop("over_2_5",None)
    if not has_1x2:
        odds.pop("1x2",None)
    if undated:
        for row in odds.values():
            row.pop("last_source_update_utc",None)
    pkt={"status":"captured_pre_kickoff","schema":1,
         "provider":"API-Football","fixture_id":"444",
         "home_team_id":"11","away_team_id":"22",
         "kickoff_utc":KICK,
         "captured_at_utc":captured.isoformat(),
         "multi_market":odds,"verified_historical_fixture_xg":[],
         "confirmed_starting_xi":[]}
    pkt["snapshot_sha256"]=hashlib.sha256(json.dumps(pkt,sort_keys=True,
          separators=(",",":"),allow_nan=False).encode()).hexdigest()
    if tamper:
        pkt["multi_market"]={}
    return pkt


def test_fresh_market_acceptance_and_stale_market_details():
    fresh=market_status(packet(),FIX,as_of=NOW)
    assert fresh["ready"]
    assert fresh["fresh_market_types"]==["1x2","over_2_5"]
    assert not fresh["missing_required_1x2"]
    assert not fresh["missing_required_goal_total"]
    stale=market_status(packet(source_age_hours=13,captured_age_hours=12.5),
                        FIX,as_of=NOW)
    assert not stale["ready"]
    assert stale["fresh_market_types"]==[]
    assert stale["stale_or_future_market_types"]==["1x2","over_2_5"]
    assert stale["reason"]=="timestamped_1x2_and_goal_total_markets_required"


def test_missing_total_undated_and_future_quotes_are_not_fabricated():
    missing=market_status(packet(has_totals=False),FIX,as_of=NOW)
    assert not missing["ready"]
    assert missing["fresh_market_types"]==["1x2"]
    assert missing["missing_required_goal_total"]
    undated=market_status(packet(undated=True),FIX,as_of=NOW)
    assert not undated["ready"]
    assert undated["undated_market_types"]==["1x2","over_2_5"]
    future=market_status(packet(source_age_hours=-1),FIX,as_of=NOW)
    assert not future["ready"]
    assert future["stale_or_future_market_types"]==["1x2","over_2_5"]
    assert not shadow({"fixture":FIX,"result":{"matrix":BASE},
                       "league_history_matches":100},
                      packet(has_totals=False),as_of=NOW)["ready"]


def test_forged_market_digest_is_rejected_before_refresh_choice():
    forged=packet(tamper=True)
    state=market_status(forged,FIX,as_of=NOW)
    assert not state["ready"]
    assert state["reason"]=="missing_or_mismatched_provider_snapshot"
    assert select_v23_packet(FIX,NOW,forged) is None


def test_prefers_fresh_private_market_over_newer_but_stale_github_market():
    repo=packet(source_age_hours=18,captured_age_hours=17.5)
    private=packet(source_age_hours=0.5,captured_age_hours=0.2)
    selected=select_v23_packet(FIX,NOW,repo,private)
    assert selected["snapshot_sha256"]==private["snapshot_sha256"]
    assert market_status(selected,FIX,as_of=NOW)["ready"]
    assert _source_fixture_key(FIX)==_source_fixture_key(dict(FIX))


def test_refresh_requires_explicit_consent_owner_login_and_api_key():
    stale=packet(source_age_hours=18,captured_age_hours=17.5)
    for consent,owner,key in ((False,True,True),(True,False,True),(True,True,False)):
        assert not v23_capture_needed(stale,FIX,NOW,source_consent=consent,
                                       private=owner,has_key=key)
    assert v23_capture_needed(stale,FIX,NOW,source_consent=True,
                              private=True,has_key=True)
    assert v23_capture_needed(None,FIX,NOW,source_consent=True,
                              private=True,has_key=True)
    assert not v23_capture_needed(packet(),FIX,NOW,source_consent=True,
                                  private=True,has_key=True)
