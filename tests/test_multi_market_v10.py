"""No-network v10 multi-bookmaker and frozen comparison tests."""
from datetime import datetime,timedelta,timezone
import numpy as np
import pytest

import multi_market_v10 as multi
import v10_prospective as shadow
import v10_evaluate as grader
from forecast_ledger import append_file_record, read_file_records

from test_v10_prospective import fixture,when,candidate,evidence,verified


def packet(now,fixture_id=9011,changes=None,dateshift=0):
    changes=changes or []
    bookmakers=[]
    for i in range(3):
        bookmakers.append({"name":("Pinnacle","Bet365","Bookmaker Three")[i],
            "bets":[{"name":"Match Winner","values":[
                {"value":"Home","odd":str(2.0+(changes[i] if i<len(changes) else i*.05))},
                {"value":"Draw","odd":"3.50"},
                {"value":"Away","odd":"4.0"}]},
                {"name":"Goals Over/Under","values":[
                    {"value":"Under 2.5","odd":"1.85"},
                    {"value":"Over 2.5","odd":"2.06"}]}]})
    return [{"fixture":{"id":fixture_id},
             "update":(now+timedelta(hours=dateshift)).isoformat(),
             "bookmakers":bookmakers}]


def test_requires_multiple_real_source_snapshots():
    now=when();game=fixture(9011,now,"Test A","Test B",36)
    assert multi.market_consensus(packet(now),game,as_of=now)
    missing=packet(now)
    missing[0]["bookmakers"]=missing[0]["bookmakers"][:2]
    assert multi.market_consensus(missing,game,as_of=now) is None
    assert multi.market_consensus(packet(now,fixture_id=999),game,as_of=now) is None
    assert multi.market_consensus(packet(now,dateshift=2),game,as_of=now) is None


def test_v10_median_distribution_validated_before_kickoff(monkeypatch):
    now=when();game=fixture(9011,now,"Test A","Test B",36)
    market=multi.market_consensus(packet(now,changes=[0,.10,.17]),game,as_of=now)
    assert market["bookmaker_count"]==3
    assert sum(market["winner"])==pytest.approx(1.0)
    assert sum(market["totals"])==pytest.approx(1.0)
    monkeypatch.setattr(multi,"research_predict",lambda *a,**k:candidate())
    result=multi.predict(game,[],market=market,as_of=now)
    assert result["matrix"].sum()==pytest.approx(1)
    assert result["model_version"]==multi.VERSION
    assert result["bookmakers_count"]==3
    assert result["llm_provider_calls"]==0
    with pytest.raises(ValueError):
        multi.predict(game,[],market={**market,"snapshot_utc":(now+timedelta(days=2)).isoformat()},as_of=now)


def test_same_odds_request_can_archive_three_independent_models(monkeypatch,tmp_path):
    now=when();game=fixture(9011,now,"Test A","Test B",36)
    monkeypatch.setattr(shadow,"research_predict",lambda *a,**k:candidate())
    monkeypatch.setattr(multi,"research_predict",lambda *a,**k:candidate())
    def run_market(*args,**kwargs):
        values=dict(candidate())
        values.update(model_version="market-informed-v9-experimental",market_used=True)
        return values
    monkeypatch.setattr(shadow,"market_predict",run_market)
    calls=[]
    def api(key,endpoint,params,with_metadata=False):
        calls.append(endpoint)
        return {"data":packet(now),"metadata":{
            "fetched_at_utc":now.isoformat(),"retrieved_at_utc":now.isoformat(),
            "freshness_known":True,"stale":False}}
    result=shadow.capture(game,evidence(now),now,api_key="test-key",
                          request=api,market_calls_remaining=1,root=tmp_path)
    assert result["v8_recorded"] and result["v9_recorded"] and result["v10_recorded"]
    assert calls==["/odds"]
    records=read_file_records(root=tmp_path,kind="forecasts")
    assert {r["model_key"] for r in records}=={"v8","v9","v10"}
    assert next(x for x in records if x["model_key"]=="v10")["market_snapshot"]["bookmaker_count"]==3


def test_a_frozen_double_compares_v10_on_the_same_pair(tmp_path):
    now=when()
    games=[fixture(9261,now,"A","B",36),fixture(9262,now,"C","D",37)]
    v8=candidate()
    for game in games:
        append_file_record(shadow._record(game,evidence(now),v8,"v8",now),root=tmp_path)
        v10={**v8,"model_version":multi.VERSION}
        source={"bookmaker":"Median of 3 bookmakers","bookmaker_count":3,
                "bookmaker_names":["A","B","C"],"fixture_id":game["id"],
                "snapshot_utc":now.isoformat(),
                "winner":(.44,.29,.27),"totals":(.48,.52),
                "settlement":"regulation_90_minutes"}
        append_file_record(shadow._record(game,evidence(now),v10,"v10",now,market=source),root=tmp_path)
    result=grader.freeze_doubles(root=tmp_path,now=now)
    assert len(result["days_saved"])==1
    path=tmp_path/"doubles"/(result["days_saved"][0]+".json")
    import json
    saved=json.loads(path.read_text())
    assert "v10" in saved["same_pair_challengers"]
    assert len(saved["same_pair_challengers"]["v10"])==5
    rows=read_file_records(root=tmp_path,kind="forecasts")
    originals={str(x["fixture_id"]):x for x in rows if x["model_key"]=="v8"}
    final={key:verified(row,1,0) for key,row in originals.items()}
    report=grader.evaluate(root=tmp_path,results=final)
    assert report["models"]["v10"]["verified_fixtures"]==2
    assert report["five_doubles"]["same_pair_challengers"]["v10"]["same_pair_verified_days"]==1
    assert report["paired_v8_v10"]["verified_same_fixture_count"]==2
