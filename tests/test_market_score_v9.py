"""Offline regression checks for correctly timestamped market-informed v9 scores."""
from datetime import datetime,timedelta,timezone
import numpy as np
import pytest
import market_score_v9 as v9


def _fixture(now):
    return {"id":2026,"utcDate":(now+timedelta(days=1)).isoformat(),
            "provider":"API-Football","homeTeam":{"id":1},"awayTeam":{"id":2}}


def _response(now, fixture_id=2026, update_offset=0):
    return [{"fixture":{"id":fixture_id},
             "update":(now+timedelta(hours=update_offset)).isoformat(),
             "bookmakers":[{"id":1,"name":"Pinnacle",
                            "bets":[{"id":1,"name":"Match Winner",
                                     "values":[{"value":"Home","odd":"1.80"},
                                               {"value":"Draw","odd":"3.50"},
                                               {"value":"Away","odd":"4.60"}]},
                                    {"id":5,"name":"Goals Over/Under",
                                     "values":[{"value":"Under 2.5","odd":"1.90"},
                                               {"value":"Over 2.5","odd":"1.95"}]}]}]}]


def _baseline(*args,**kwargs):
    matrix=np.ones((8,8),dtype=float)
    matrix[1,1]=14
    matrix[2,0]=4
    matrix/=matrix.sum()
    return {"matrix":matrix,"scores":[(1,1,float(matrix[1,1]))],
            "home_xg":1.4,"away_xg":1.1,"model_version":"v8",
            "component_weights":{"preserved_v6":.55,"opponent_dixon_coles_v7":.45}}


def test_market_source_verified_snapshot_and_margin_removed():
    now=datetime.now(timezone.utc)
    item=v9.select_market_response(_response(now),_fixture(now),as_of=now)
    assert item["bookmaker"]=="Pinnacle"
    assert sum(item["winner"])==pytest.approx(1)
    assert sum(item["totals"])==pytest.approx(1)
    assert item["winner"][0]>item["winner"][2]


def test_reject_wrong_match_future_timestamp_and_stale_odds():
    now=datetime.now(timezone.utc)
    fixture=_fixture(now)
    assert v9.select_market_response(_response(now,3000),fixture,now) is None
    assert v9.select_market_response(_response(now,2026,2),fixture,now) is None
    assert v9.select_market_response(_response(now-timedelta(days=4)),fixture,now) is None
    assert v9.select_market_response(_response(now),fixture,now+timedelta(days=2)) is None


def test_v9_market_projection_changes_probabilities_not_a_manufactured_score(monkeypatch):
    now=datetime.now(timezone.utc)
    fixture=_fixture(now)
    monkeypatch.setattr(v9,"research_predict",_baseline)
    market=v9.select_market_response(_response(now),fixture,now)
    new=v9.predict(fixture,[],market=market,as_of=now)
    old=v9.predict(fixture,[],as_of=now)
    assert new["market_used"] and not old["market_used"]
    assert new["model_version"]==v9.VERSION
    assert new["llm_provider_calls"]==0
    assert new["matrix"].sum()==pytest.approx(1)
    assert new["home_win"]>old["home_win"]
    assert np.all(new["matrix"]>=0)
    assert new["scores"][0][2]<=1


def test_no_odds_is_identical_to_original_research_distribution(monkeypatch):
    now=datetime.now(timezone.utc)
    monkeypatch.setattr(v9,"research_predict",_baseline)
    a=v9.predict(_fixture(now),[],as_of=now)
    assert np.array_equal(a["matrix"],_baseline()["matrix"])
    assert not a["market_used"]


def test_market_cannot_be_paired_with_wrong_fixture_or_future_timestamp(monkeypatch):
    now=datetime.now(timezone.utc)
    monkeypatch.setattr(v9,"research_predict",_baseline)
    fixture=_fixture(now)
    market=v9.select_market_response(_response(now),fixture,now)
    with pytest.raises(ValueError):
        v9.predict({**fixture,"id":999},[],market=market,as_of=now)
    with pytest.raises(ValueError):
        v9.predict(fixture,[],market={**market,"snapshot_utc":(now+timedelta(hours=2)).isoformat()},as_of=now)
