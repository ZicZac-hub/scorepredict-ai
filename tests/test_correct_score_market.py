"""Real single-leg prices remain distinct from modeled probability and double EV."""
from datetime import datetime,timedelta,timezone
from copy import deepcopy

from correct_score_market import parse_correct_score_quotes
from v21_provider_evidence import capture,_shots_statistics,valid_snapshot

NOW=datetime(2032,7,3,3,tzinfo=timezone.utc)
FIXTURE={"id":123,"provider":"API-Football",
         "utcDate":(NOW+timedelta(hours=8)).isoformat(),
         "homeTeam":{"id":11},"awayTeam":{"id":22}}


def response():
    return [{"fixture":{"id":123,"date":FIXTURE["utcDate"]},
             "update":(NOW-timedelta(minutes=10)).isoformat(),
             "bookmakers":[{"name":"Book","bets":[
                 {"name":"Correct Score","values":[
                     {"value":"1:0","odd":"7.5"},
                     {"value":"2-1","odd":"11.0"},
                     {"value":"Other","odd":"30"},
                     {"value":"3:2","odd":"inf"}]}]}]}]


def test_preserves_partial_raw_prices_without_implied_probability_normalization():
    result=parse_correct_score_quotes(response(),FIXTURE,observed_at=NOW)
    assert result["ready"] and not result["executable_double_quote"]
    assert [q["score"] for q in result["quotes"]]==[[1,0],[2,1]]
    assert [q["decimal_odds"] for q in result["quotes"]]==[7.5,11.0]
    assert all(q["fair_probability"] is None for q in result["quotes"])
    assert all(q["timestamp_basis"]=="bookmaker_quote_updated_at" for q in result["quotes"])
    assert result["partial_score_market_is_not_normalized"]


def test_rejects_wrong_fixture_time_teams_and_non_regulation_market():
    for change in ("fixture","kickoff","teams","market","future","stale"):
        raw=deepcopy(response())
        item=raw[0]
        if change=="fixture": item["fixture"]["id"]=999
        elif change=="kickoff": item["fixture"]["date"]=(NOW+timedelta(days=1)).isoformat()
        elif change=="teams": item["teams"]={"home":{"id":99},"away":{"id":22}}
        elif change=="market": item["bookmakers"][0]["bets"][0]["name"]="Correct Score - First Half"
        elif change=="future": item["update"]=(NOW+timedelta(seconds=1)).isoformat()
        elif change=="stale": item["update"]=(NOW-timedelta(hours=13)).isoformat()
        assert not parse_correct_score_quotes(raw,FIXTURE,observed_at=NOW)["ready"],change
    assert not parse_correct_score_quotes(response(),FIXTURE,observed_at=NOW+timedelta(hours=8))["ready"]


def test_missing_quote_time_needs_verified_api_fetch_and_keeps_its_timestamp_basis():
    raw=response(); raw[0].pop("update")
    assert not parse_correct_score_quotes(raw,FIXTURE,observed_at=NOW)["ready"]
    result=parse_correct_score_quotes(raw,FIXTURE,observed_at=NOW,
                     verified_provider_fetched_at=(NOW-timedelta(minutes=2)).isoformat())
    assert result["ready"]
    assert all(q["timestamp_basis"]=="api_response_fetched_at" for q in result["quotes"])


def test_same_existing_statistics_and_odds_calls_retain_shots_and_prices():
    old={"id":122,"provider":"API-Football","status":"FINISHED",
         "utcDate":(NOW-timedelta(days=3)).isoformat(),
         "homeTeam":{"id":11},"awayTeam":{"id":44}}
    calls=[]
    def request(key,endpoint,params):
        calls.append(endpoint)
        if endpoint=="/odds": return response()
        if endpoint=="/fixtures/statistics":
            return [{"team":{"id":11},"statistics":[
                {"type":"Total Shots","value":14},{"type":"Shots on Goal","value":5}]},
                {"team":{"id":44},"statistics":[
                {"type":"Total Shots","value":"9"},{"type":"Shots on Goal","value":"3"}]}]
        return []
    packet=capture(FIXTURE,[old],"stub",now=NOW,clock=lambda:NOW,request=request)
    assert calls==["/fixtures/lineups","/injuries","/odds","/fixtures/statistics"]
    assert packet["verified_historical_fixture_xg"]==[]
    shots=packet["verified_historical_fixture_shots"][0]
    assert (shots["home_shots"],shots["away_shots"],shots["home_shots_on_target"],shots["away_shots_on_target"])==(14,9,5,3)
    assert len(packet["correct_score_market_quotes"])==2
    assert packet["available_sources"]["historical_shots"]
    assert valid_snapshot(packet,FIXTURE,now=NOW)


def test_inconsistent_shots_do_not_become_invented_zero_target_counts():
    raw=[{"team":{"id":11},"statistics":[
        {"type":"Total Shots","value":4},{"type":"Shots on Goal","value":5}]},
        {"team":{"id":22},"statistics":[{"type":"Total Shots","value":0}]}]
    shots=_shots_statistics(raw,11,22)
    assert "shots_on_target" not in shots["11"]
    assert "shots_on_target" not in shots["22"]
    raw[1]["statistics"][0]["value"]=True
    assert _shots_statistics(raw,11,22) is None
