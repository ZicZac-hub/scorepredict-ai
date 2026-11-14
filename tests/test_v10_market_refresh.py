"""Regression tests for live near-kickoff market refresh and fair model grading."""
from datetime import datetime, timedelta, timezone
import json

from forecast_ledger import append_file_record, read_file_records
import v10_prospective as shadow
import v10_evaluate as grader
from test_v10_prospective import candidate, fixture, evidence, verified, when


def _real_odds(now,match_id):
    return {"metadata":{"freshness_known":True,"stale":False,
             "fetched_at_utc":now.isoformat(),"retrieved_at_utc":now.isoformat()},
            "data":[{"fixture":{"id":match_id},"update":now.isoformat(),
             "bookmakers":[{"name":"Pinnacle","bets":[{"name":"Match Winner",
              "values":[{"value":"Home","odd":"2.05"},{"value":"Draw","odd":"3.50"},
                        {"value":"Away","odd":"3.95"}]}]}]}]}


def test_existing_v8_gets_later_market_forecast_without_overwrite(monkeypatch,tmp_path):
    now=when()
    game=fixture(98301,now,"A","B",12)
    previous=now-timedelta(days=2)
    old_packet=evidence(previous)
    original=shadow._record(game,old_packet,candidate(),"v8",previous)
    assert append_file_record(original,root=tmp_path)
    original_file=read_file_records(root=tmp_path)[0]
    new_packet=evidence(now)
    new_packet["input_digest"]="new-history-before-kickoff"
    monkeypatch.setattr(shadow,"research_predict",lambda *args,**kwargs:candidate())
    def market_model(*args,**kwargs):
        new=dict(candidate())
        new["model_version"]="market-informed-v9-experimental"
        new["market_used"]=True
        return new
    monkeypatch.setattr(shadow,"market_predict",market_model)
    calls=[]
    def request(key,endpoint,params,with_metadata=False):
        assert key=="secret" and endpoint=="/odds" and with_metadata
        calls.append(params["fixture"])
        return _real_odds(now,game["id"])
    report=shadow.capture_market_followup(game,new_packet,now,
        api_key="secret",request=request,root=tmp_path,market_calls_remaining=1)
    assert report["v8_control_recorded"] is True
    assert report["v9_recorded"] is True
    assert report["market_calls"]==1
    assert calls==[98301]
    data=read_file_records(root=tmp_path)
    assert {r["model_key"] for r in data}=={"v8","v8_control","v9"}
    assert next(r for r in data if r["model_key"]=="v8")==original_file
    control=next(r for r in data if r["model_key"]=="v8_control")
    market=next(r for r in data if r["model_key"]=="v9")
    assert control["input_digest"]==market["input_digest"]
    assert control["input_digest"]!=original["input_digest"]
    assert "secret" not in json.dumps(data)
    # A duplicate discovery must not query or rewrite a frozen source.
    repeated=shadow.capture_market_followup(game,new_packet,now,
        api_key="secret",request=request,root=tmp_path,market_calls_remaining=1)
    assert repeated["reason"]=="v9_already_frozen"
    assert calls==[98301]
    final={str(game["id"]):verified(market,1,0)}
    scored=grader.evaluate(root=tmp_path,results=final)
    assert scored["paired_v8_v9"]["verified_same_fixture_count"]==1
    assert scored["comparison_diagnostics"]["matched_market_time_controls"]>=1
    assert scored["comparison_diagnostics"]["excluded_mismatched_evidence_pairs"]==0


def test_challenger_not_fairly_compared_to_older_history(tmp_path):
    now=when()
    game=fixture(98302,now,"C","D",8)
    old=now-timedelta(days=2)
    historical=shadow._record(game,evidence(old),candidate(),"v8",old)
    assert append_file_record(historical,root=tmp_path)
    fresh=evidence(now)
    fresh["input_digest"]="new-match-evidence-same-game"
    updated={**candidate(),"model_version":"market-informed-v9-experimental"}
    market={"bookmaker":"Pinnacle","snapshot_utc":now.isoformat(),
            "winner":(.4,.3,.3),"totals":None,"settlement":"regulation_90_minutes"}
    assert append_file_record(shadow._record(game,fresh,updated,"v9",now,market=market),
                              root=tmp_path)
    last={str(game["id"]):verified(historical,1,0)}
    report=grader.evaluate(root=tmp_path,results=last)
    assert report["models"]["v8"]["verified_fixtures"]==1
    assert report["models"]["v9"]["verified_fixtures"]==1
    assert report["paired_v8_v9"]["verified_same_fixture_count"]==0
    assert report["comparison_diagnostics"]["excluded_mismatched_evidence_pairs"]==1
