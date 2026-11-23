"""Never accept unmatched, stale, or incomplete live bookmaker quotes."""
from datetime import datetime,timezone,timedelta
from v25_live_market_check import inspect_pair
from v25_upcoming import automatic_quality

NOW=datetime(2031,11,18,12,tzinfo=timezone.utc)
KICK=(NOW+timedelta(hours=6)).isoformat()


def fixture(fid):
    return {"provider":"API-Football","id":fid,"utcDate":KICK,
            "homeTeam":{"id":fid*10,"name":"Home"},
            "awayTeam":{"id":fid*10+1,"name":"Away"}}


def report():
    return {"status":"ready","first_fixture":fixture(101),
            "second_fixture":fixture(202),"early_frozen_source":[False,False],
            "v8_optimal_five_estimated_mass":.12,
            "source_evidence_audit":[{"verified_pre_match_market":False}]*2}


def odds(fid,date=KICK,include_totals=True):
    bets=[{"name":"Match Winner","values":[
        {"value":"Home","odd":"1.80"},{"value":"Draw","odd":"3.50"},
        {"value":"Away","odd":"4.50"}]}]
    if include_totals:
        bets.append({"name":"Goals Over/Under","values":[
            {"value":"Over 2.5","odd":"1.91"},
            {"value":"Under 2.5","odd":"1.93"}]})
    return {"fixture":{"id":fid,"date":date},
            "bookmakers":[{"name":"Example Book","bets":bets}]}


def test_real_two_leg_fixture_matched_prices_can_pass_the_market_gate():
    calls=[]
    def fetch(key,endpoint,params,with_metadata=False):
        assert key=="provider-secret" and endpoint=="/odds" and with_metadata
        calls.append(params["fixture"])
        return {"metadata":{"fetched_at_utc":NOW.isoformat(),"stale":False},
                "data":[odds(params["fixture"])]}
    original=report()
    out=inspect_pair("provider-secret",original,now=NOW,request=fetch)
    assert calls==[101,202]
    assert out["both_verified"]
    assert out["provider_requests_attempted"]==2
    assert out["does_not_change_v25_scores"] is True
    assert "live_market_evidence" not in original
    actual={**original,"live_market_evidence":out}
    assert automatic_quality(actual)["passed"]


def test_mismatched_fixture_and_kickoff_cannot_fake_market_validation():
    def wrong_id(key,endpoint,params,with_metadata=False):
        return {"metadata":{"fetched_at_utc":NOW.isoformat(),"stale":False},
                "data":[odds(999)]}
    out=inspect_pair("key",report(),now=NOW,request=wrong_id)
    assert not out["both_verified"]
    assert all(leg["reason"]=="no_id_and_kickoff_matched_market"
               for leg in out["legs"])
    def wrong_date(key,endpoint,params,with_metadata=False):
        return {"metadata":{"fetched_at_utc":NOW.isoformat(),"stale":False},
                "data":[odds(params["fixture"],
                             date=(NOW+timedelta(days=4)).isoformat())]}
    assert not inspect_pair("key",report(),now=NOW,request=wrong_date)["both_verified"]


def test_missing_totals_or_stale_api_source_is_not_accepted():
    def missing_totals(key,endpoint,params,with_metadata=False):
        return {"metadata":{"fetched_at_utc":NOW.isoformat(),"stale":False},
                "data":[odds(params["fixture"],include_totals=False)]}
    r=report()
    out=inspect_pair("key",r,now=NOW,request=missing_totals)
    assert not out["both_verified"]
    assert "no_source_matched_fresh_market_on_both_legs" in automatic_quality(
        {**r,"live_market_evidence":out})["reasons"]
    def stale(key,endpoint,params,with_metadata=False):
        return {"metadata":{"fetched_at_utc":(NOW-timedelta(hours=18)).isoformat(),
                            "stale":True},
                "data":[odds(params["fixture"])]}
    assert not inspect_pair("key",r,now=NOW,request=stale)["both_verified"]


def test_report_fixture_id_must_match_spot_checked_ids():
    original=report()
    spoof={"both_verified":True,"legs":[
        {"fixture_id":"999","verified_pre_match_market":True},
        {"fixture_id":"202","verified_pre_match_market":True}]}
    assert not automatic_quality({**original,"live_market_evidence":spoof})["passed"]


def test_no_api_key_or_started_game_makes_no_provider_calls():
    def forbidden(*args,**kwargs):
        raise AssertionError("Must not call paid provider here")
    assert not inspect_pair("",report(),now=NOW,request=forbidden)["both_verified"]
    too_late=NOW+timedelta(hours=6)
    out=inspect_pair("key",report(),now=too_late,request=forbidden)
    assert out["provider_requests_attempted"]==0
    assert not out["both_verified"]
