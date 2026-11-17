"""Owner-requested, source-verified scouting fields are never treated as xG."""
from datetime import datetime, timedelta, timezone

from v21_provider_evidence import (capture, valid_snapshot,
                                   _historical_match_stats, _provider_prediction)

NOW = datetime(2032, 11, 18, 5, tzinfo=timezone.utc)
FIXTURE = {"provider":"API-Football","id":4321,
           "utcDate":(NOW+timedelta(hours=20)).isoformat(),
           "homeTeam":{"id":31},"awayTeam":{"id":32}}
PAST = {"provider":"API-Football","id":4001,"status":"FINISHED",
        "utcDate":(NOW-timedelta(days=2)).isoformat(),
        "homeTeam":{"id":31},"awayTeam":{"id":80}}


def test_collect_source_stats_and_independent_opinion_without_fake_xg():
    calls=[]
    def request(key,endpoint,params):
        calls.append(endpoint)
        if endpoint=="/fixtures/statistics":
            return [
                {"team":{"id":31},"statistics":[
                    {"type":"Shots on Goal","value":8},
                    {"type":"Corner Kicks","value":7},
                    {"type":"Ball Possession","value":"56%"}]},
                {"team":{"id":80},"statistics":[
                    {"type":"Shots on Goal","value":2},
                    {"type":"Corner Kicks","value":1},
                    {"type":"Ball Possession","value":"44%"}]},
            ]
        if endpoint=="/predictions":
            return [{"fixture":{"id":4321},
                     "teams":{"home":{"id":31},"away":{"id":32}},
                     "predictions":{"percent":{"home":"50%","draw":"27%",
                                               "away":"23%"},
                                    "advice":"Provider predicts home or draw"}}]
        return []
    result=capture(FIXTURE,[PAST],"test",now=NOW,request=request,
                   clock=lambda:NOW,max_history_per_team=2,
                   include_provider_prediction=True)
    assert result["status"]=="captured_pre_kickoff"
    assert result["verified_historical_fixture_xg"]==[]
    assert result["verified_historical_match_statistics"][0]["home"]=={
        "shots_on_target":8,"corners":7,"possession_pct":56}
    assert result["verified_historical_match_statistics"][0]["not_genuine_xg"]
    assert result["provider_pre_match_opinion"]["probabilities_1x2"]=={
        "home":0.5,"draw":0.27,"away":0.23}
    assert result["provider_pre_match_opinion"]["not_production_model_weight"]
    assert result["diagnostics"]["xg_unavailable_reason"]==(
        "provider_returned_stats_without_genuine_xg")
    assert result["available_sources"]["historical_match_statistics"] is True
    assert result["available_sources"]["historical_xg"] is False
    assert calls.count("/fixtures/statistics")==1
    assert calls.count("/predictions")==1
    assert valid_snapshot(result,FIXTURE,now=NOW)


def test_provider_prediction_rejects_mismatched_teams():
    wrong=[{"fixture":{"id":4321},
            "teams":{"home":{"id":31},"away":{"id":999}},
            "predictions":{"percent":{"home":"50%","draw":"27%","away":"23%"}}}]
    assert _provider_prediction(wrong,4321,31,32) is None


def test_history_statistics_require_both_teams():
    single=[{"team":{"id":31},
             "statistics":[{"type":"Total Shots","value":14}]}]
    assert _historical_match_stats(single,31,80) is None


def test_missing_shot_stat_is_not_zero():
    data=[{"team":{"id":31},"statistics":[
              {"type":"Total Shots","value":0},
              {"type":"Corner Kicks","value":None}]},
          {"team":{"id":80},"statistics":[
              {"type":"Total Shots","value":11},
              {"type":"Corner Kicks","value":3}]}]
    found=_historical_match_stats(data,31,80)
    assert found["31"]["total_shots"]==0
    assert "corners" not in found["31"]


def test_48_hour_selection_prioritises_near_kickoff_for_lineups(monkeypatch,tmp_path):
    import v10_prospective
    import v21_capture_sources as mod
    near = {"provider":"API-Football","fixture_id":"901","model_key":"v8",
            "input_digest":"a","kickoff_utc":(NOW+timedelta(minutes=30)).isoformat()}
    early = {"provider":"API-Football","fixture_id":"902","model_key":"v8",
             "input_digest":"b","kickoff_utc":(NOW+timedelta(hours=35)).isoformat()}
    monkeypatch.setattr(v10_prospective,"read_latest",
                        lambda root:{("API-Football","902","v8"):early,
                                     ("API-Football","901","v8"):near})
    monkeypatch.setattr(mod,"original_model",lambda root,leg:dict(leg))
    picked=mod.current_pair(tmp_path,NOW)
    assert [r["fixture_id"] for r in picked]==["901","902"]


def test_stats_with_pass_accuracy_and_fouls_are_not_xg():
    raw=[{"team":{"id":31},"statistics":[
              {"type":"Passes %","value":"83%"},
              {"type":"Fouls","value":9},
              {"type":"Total passes","value":550}]},
         {"team":{"id":80},"statistics":[
              {"type":"Passes %","value":"70%"},
              {"type":"Fouls","value":16},
              {"type":"Total passes","value":300}]}]
    parsed=_historical_match_stats(raw,31,80)
    assert parsed["31"]["pass_accuracy_pct"]==83
    assert parsed["80"]["fouls"]==16
    assert parsed["31"]["total_passes"]==550
    assert "expected_goals" not in parsed["31"]
