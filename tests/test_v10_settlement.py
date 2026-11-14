"""Offline integration tests for independent verified v10 result settlement."""
from datetime import datetime,timedelta,timezone
from pathlib import Path
import json

import v10_settlement as settlement
import v10_evaluate as evaluator
import v10_prospective as shadow
from forecast_ledger import append_file_record
from test_v10_prospective import fixture,candidate,evidence


class Response:
    def __init__(self,payload):
        self.payload=payload
    def raise_for_status(self):
        return None
    def json(self):
        return self.payload


def fulltime_result(row,gh=1,ga=0,*,status="FT",other_team=None):
    return {"fixture":{"id":int(row["fixture_id"]),
                       "date":row["kickoff_utc"],
                       "status":{"short":status}},
            "league":{"id":39,"season":2026,"name":"Premier League"},
            "teams":{"home":{"id":int(row["home_team_id"]),"name":row["home_team"]},
                     "away":{"id":int(row["away_team_id"])+int(bool(other_team)),
                             "name":row["away_team"]}},
            "score":{"fulltime":{"home":gh,"away":ga}},
            "goals":{"home":gh,"away":ga}}


def one_frozen(root,fid=97310):
    now=datetime.now(timezone.utc).replace(microsecond=0)
    yesterday=now-timedelta(days=2)
    game=fixture(fid,yesterday,"Verified Home","Verified Away",delta_hours=25)
    row=shadow._record(game,evidence(yesterday),candidate(),"v8",yesterday)
    assert append_file_record(row,root=root,kind="forecasts")
    return row


def test_verify_research_only_fixture_and_update_prospective_performance(tmp_path,monkeypatch):
    row=one_frozen(tmp_path)
    calls=[]
    def fetch(url,params,headers,timeout):
        calls.append((url,params["id"]))
        return Response({"response":[fulltime_result(row)],"errors":[]})
    monkeypatch.setattr(evaluator,"RESULTS",tmp_path/"legacy_results.json")
    report=settlement.collect("sample-key",root=tmp_path,
                              legacy_results={},request=fetch,limit=3)
    assert report["newly_verified"]==1
    assert report["checked"]==1
    recorded=json.loads((tmp_path/"verified_results.json").read_text())
    assert recorded[str(row["fixture_id"])]["home_goals"]==1
    stats=evaluator.evaluate(root=tmp_path)
    assert stats["models"]["v8"]["verified_fixtures"]==1
    assert stats["models"]["v8"]["pending_fixtures"]==0
    assert stats["models"]["v9"]["verified_fixtures"]==0
    # A second collection reuses the immutable verified source evidence.
    again=settlement.collect("sample-key",root=tmp_path,
                             legacy_results={},request=fetch,limit=3)
    assert again["newly_verified"]==0
    assert again["already_verified"]==1
    assert len(calls)==1


def test_wrong_team_identity_cannot_be_counted_as_real_result(tmp_path,monkeypatch):
    row=one_frozen(tmp_path,97311)
    def fetch(url,params,headers,timeout):
        return Response({"response":[fulltime_result(row,other_team=True)],"errors":[]})
    result=settlement.collect("sample-key",root=tmp_path,
                              legacy_results={},request=fetch)
    assert result["newly_verified"]==0
    assert result["rejected_or_unavailable"]==1
    assert json.loads((tmp_path/"verified_results.json").read_text())=={}


def test_unfinished_fixture_does_not_count_as_loss(tmp_path,monkeypatch):
    row=one_frozen(tmp_path,97312)
    def fetch(url,params,headers,timeout):
        return Response({"response":[fulltime_result(row,status="NS")],"errors":[]})
    status=settlement.collect("sample-key",root=tmp_path,
                              legacy_results={},request=fetch)
    assert status["newly_verified"]==0
    monkeypatch.setattr(evaluator,"RESULTS",tmp_path/"legacy_results.json")
    report=evaluator.evaluate(root=tmp_path)
    assert report["models"]["v8"]["verified_fixtures"]==0
    assert report["models"]["v8"]["pending_fixtures"]==1
    assert report["models"]["v8"]["exact_score_rate"] is None


def test_research_result_provider_quota_limit_is_honoured(tmp_path):
    r1=one_frozen(tmp_path,97320)
    r2=one_frozen(tmp_path,97321)
    seen=[]
    def fetch(url,params,headers,timeout):
        seen.append(params["id"])
        return Response({"response":[],"errors":[]})
    report=settlement.collect("test",root=tmp_path,
                              legacy_results={},request=fetch,limit=1)
    assert report["checked"]==1 and len(seen)==1
    assert report["still_pending"]>=1
