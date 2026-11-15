"""V21 forensic regression: no fabricated providers, high-goal metrics or future leaks."""
from datetime import datetime,timedelta,timezone
import json
import numpy as np
import pytest

from v21_provider_evidence import capture,valid_snapshot,_market
from v21_goal_model_comparison import variants,inspect_prediction,compare_frozen,walk_forward

NOW=datetime(2032,11,15,3,tzinfo=timezone.utc)
KICK=(NOW+timedelta(hours=5)).isoformat()
FIXTURE={"id":9001,"provider":"API-Football","utcDate":KICK,
         "homeTeam":{"id":11,"name":"Home"},"awayTeam":{"id":22,"name":"Away"}}


def test_missing_access_or_after_kickoff_makes_zero_requests():
    calls=[]
    def req(*a,**kw):
        calls.append(1); raise AssertionError("Must not make a request")
    assert capture(FIXTURE,[],"",now=NOW,request=req)["status"]=="unavailable"
    assert capture(FIXTURE,[],"key",now=NOW+timedelta(hours=6),request=req)["status"]=="unavailable"
    assert not calls


def test_source_capture_real_xg_lineup_injury_and_de_vig_market():
    old={"id":8991,"provider":"API-Football","status":"FINISHED",
         "utcDate":(NOW-timedelta(days=3)).isoformat(),
         "homeTeam":{"id":11},"awayTeam":{"id":44}}
    calls=[]
    def req(key,endpoint,params):
        calls.append((endpoint,params))
        if endpoint=="/fixtures/lineups":
            return [{"team":{"id":11},"formation":"4-3-3",
                     "startXI":[{"player":{"name":"Player "+str(i)}} for i in range(11)]}]
        if endpoint=="/injuries":
            return [{"team":{"id":22},"player":{"name":"Injured","reason":"injury"}}]
        if endpoint=="/odds":
            return [{"update":(NOW-timedelta(hours=1)).isoformat(),
                     "bookmakers":[{"name":"Pinnacle","bets":[
                         {"name":"Match Winner","values":[
                             {"value":"Home","odd":"2.20"},
                             {"value":"Draw","odd":"3.45"},
                             {"value":"Away","odd":"3.90"}]}]}]}]
        if endpoint=="/fixtures/statistics":
            assert str(params["fixture"])=="8991"
            return [{"team":{"id":11},"statistics":[{"type":"expected_goals","value":"2.19"}]},
                    {"team":{"id":44},"statistics":[{"type":"Expected Goals","value":1.07}]}]
        raise AssertionError(endpoint)
    result=capture(FIXTURE,[old],"sample-key",now=NOW,request=req)
    assert result["status"]=="captured_pre_kickoff"
    assert result["confirmed_starting_xi"][0]["formation"]=="4-3-3"
    assert result["verified_historical_fixture_xg"][0]["home_xg"]==pytest.approx(2.19)
    assert result["reported_injuries"][0]["player"]=="Injured"
    assert abs(sum(result["market"]["fair_1x2"].values())-1)<1e-5
    assert valid_snapshot(result,FIXTURE,now=NOW)
    assert result["diagnostics"]["queries"]==4
    assert "sample-key" not in json.dumps(result)
    changed=dict(result,fixture_id="123")
    assert not valid_snapshot(changed,FIXTURE)


def test_stale_odds_not_falsely_validated():
    raw=[{"update":(NOW-timedelta(days=2)).isoformat(),
          "bookmakers":[{"name":"Pinnacle","bets":[{"name":"Match Winner",
              "values":[{"value":"Home","odd":"2"},{"value":"Away","odd":"3"},
                        {"value":"Draw","odd":"3.3"}]}]}]}]
    assert _market(raw,NOW) is None


def test_model_variants_are_normalized_and_include_3_2_without_forcing_it():
    from v21_goal_model_comparison import _poisson
    base=np.outer(_poisson(2.1,11),_poisson(1.4,11))
    models=variants(base)
    assert set(models)=={"baseline_v8","poisson_marginal",
                         "negative_binomial_r3","bivariate_poisson_c012"}
    for method,grid in models.items():
        assert grid.sum()==pytest.approx(1,abs=1e-8)
        assert float(grid[3,2])>0
        metrics=inspect_prediction(grid,3,2)
        assert metrics["actual_high_score"]==1
        assert 0 < metrics["high_score_probs"]["3-2"] < 1
    assert models["negative_binomial_r3"][3,2] != pytest.approx(base[3,2])


def test_comparison_is_retrospective_and_no_proven_improvement():
    grid=variants(np.outer(np.ones(11)/11,np.ones(11)/11))["poisson_marginal"]
    result=compare_frozen([{"matrix":grid,"home_goals":3,"away_goals":2}])
    assert not result["model_promotion"]
    assert result["evidence"]["baseline_v8"]["verified"]==1
    assert result["evidence"]["negative_binomial_r3"]["verified"]==1


def test_walkforward_excludes_same_day_finished_not_available():
    games=[]
    for days in range(75,0,-1):
        at=NOW-timedelta(days=days)
        games.append({"provider":"API-Football","id":days,
                      "utcDate":at.isoformat(),"status":"FINISHED",
                      "homeTeam":{"id":11},"awayTeam":{"id":22},
                      "score":{"fullTime":{"home":1,"away":2}}})
    observed=[]
    def forecast(fixture,prior,as_of):
        assert fixture["status"]=="SCHEDULED"
        assert fixture["score"]["fullTime"]["home"] is None
        assert all(datetime.fromisoformat(x["utcDate"])+timedelta(hours=3)<=as_of for x in prior)
        observed.append(len(prior))
        return np.outer(np.ones(11)/11,np.ones(11)/11)
    report=walk_forward(games,forecast,min_prior=60)
    assert report["matched_test_fixtures"]>0
    assert min(observed)>=60
    assert not report["comparison"]["model_promotion"]
