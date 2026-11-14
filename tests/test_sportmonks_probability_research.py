"""No-network regressions for independently frozen Sportmonks predictions."""
import json
from datetime import datetime,timezone

from sportmonks_probability_research import (
    _name, capture, match_fixture, parse_predictions, evaluate
)


def _row():
    return {
        "fixture_id":"100","provider":"API-Football","model_key":"v8",
        "forecast_id":"frozen-a","input_digest":"a"*64,
        "home_team_id":"10","away_team_id":"20",
        "home_team":"Manchester City","away_team":"Chelsea",
        "kickoff_utc":"2026-11-16T13:00:00+00:00",
        "captured_at_utc":"2026-11-15T00:00:00+00:00",
        "predicted_score":[2,1]
    }


def _fixture(home="Manchester City",away="Chelsea",id=500):
    return {"id":id,"starting_at_timestamp":int(datetime(2026,11,16,13,
                                                         tzinfo=timezone.utc).timestamp()),
            "participants":[{"name":home,"meta":{"location":"home"}},
                            {"name":away,"meta":{"location":"away"}}]}


def test_strict_cross_provider_mapping():
    row=_row()
    assert match_fixture(row,[_fixture()])==500
    assert match_fixture(row,[_fixture(away="Liverpool")]) is None
    assert match_fixture(row,[_fixture(),_fixture(id=501)]) is None
    assert _name("Atlético Madrid")=="atletico madrid"


def test_correct_score_percentage_conversion_and_outcome():
    payload={"data":[
        {"fixture_id":500,"type":{"developer_name":"CORRECT_SCORE_PROBABILITY"},
         "predictions":{"1-1":12.4,"2-1":11.3,"0-0":9.3,"other":66.0}},
        {"fixture_id":500,"type":{"developer_name":"FULLTIME_RESULT_PROBABILITY"},
         "predictions":{"home":50,"draw":30,"away":20}}]}
    result=parse_predictions(payload,500)
    assert result["score_probabilities"]["1-1"]==.124
    assert result["top_scores"][0]=="1-1"
    assert result["outcome_probabilities"]["away"]==.2
    assert parse_predictions(payload,600)=={}


def test_not_enabled_or_without_token_never_calls_network(tmp_path):
    def forbidden(*args,**kwargs):
        raise AssertionError("Unexpected API usage")
    result=capture(root=tmp_path,source=tmp_path,request=forbidden,
                   environ={"SPORTMONKS_PREDICTIONS_ENABLED":"false",
                            "THIRD_FOOTBALL_API_KEY":"placeholder"})
    assert result["reason"]=="disabled"
    assert result["attempted"]==0
    result=capture(root=tmp_path,source=tmp_path,request=forbidden,
                   environ={"SPORTMONKS_PREDICTIONS_ENABLED":"true"})
    assert result["reason"]=="token_missing"


def test_pending_results_are_not_counted_as_losses(tmp_path):
    record=_row()
    record.update({"captured_at_utc":"2026-11-15T01:00:00+00:00",
                   "score_probabilities":{"1-1":.125},
                   "top_scores":["1-1"],"source":"Sportmonks Predictions add-on"})
    (tmp_path/"forecasts").mkdir()
    (tmp_path/"forecasts"/"API-Football_100.json").write_text(json.dumps(record))
    empty=tmp_path/"results.json"
    empty.write_text("{}")
    report=evaluate(root=tmp_path,results_path=empty)
    assert report["verified_matched_fixtures"]==0
    assert report["sportmonks_exact_rate"] is None
    assert report["original_exact_rate"] is None
