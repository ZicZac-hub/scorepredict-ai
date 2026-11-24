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


def test_recent_kickoff_is_not_queried_before_fulltime(tmp_path):
    now=datetime.now(timezone.utc).replace(microsecond=0)
    captured=now-timedelta(hours=3)
    game=fixture(97330,captured,"Recent Home","Recent Away",
                 delta_hours=2)
    row=shadow._record(game,evidence(captured),candidate(),"v8",captured)
    assert append_file_record(row,root=tmp_path,kind="forecasts")
    def no_request(*args,**kwargs):
        raise AssertionError("Cannot verify a match only 60 minutes after kickoff")
    report=settlement.collect("test",root=tmp_path,legacy_results={},
                              now=now,request=no_request)
    assert report["checked"]==0
    assert report["still_pending"]==1
    assert report["waiting_for_full_time"]==1
    assert report["newly_verified"]==0


def test_unfinished_result_retry_uses_60_minute_cooldown(tmp_path):
    row=one_frozen(tmp_path,97331)
    calls=[]
    def unfinished(url,params,headers,timeout):
        calls.append(params["id"])
        return Response({"response":[fulltime_result(row,status="NS")],
                         "errors":[]})
    now=datetime.now(timezone.utc)
    first=settlement.collect("test",root=tmp_path,legacy_results={},
                             now=now,request=unfinished)
    assert first["checked"]==1
    assert first["still_pending"]==1
    second=settlement.collect("test",root=tmp_path,legacy_results={},
                              now=now+timedelta(minutes=15),request=unfinished)
    assert second["checked"]==0
    assert second["retry_cooldown"]==1
    assert len(calls)==1
    third=settlement.collect("test",root=tmp_path,legacy_results={},
                             now=now+timedelta(minutes=61),request=unfinished)
    assert third["checked"]==1
    assert len(calls)==2
    assert third["still_pending"]==1
    # A newly finished result is still checked by exact original team IDs.
    def completed(url,params,headers,timeout):
        return Response({"response":[fulltime_result(row)],"errors":[]})
    fourth=settlement.collect("test",root=tmp_path,legacy_results={},
                              now=now+timedelta(minutes=122),request=completed)
    assert fourth["newly_verified"]==1
    assert fourth["still_pending"]==0


def test_next_five_double_legs_receive_lookup_priority_over_unrelated_singles(tmp_path):
    old_a=one_frozen(tmp_path,97341)
    old_b=one_frozen(tmp_path,97342)
    now=datetime.now(timezone.utc).replace(microsecond=0)
    capture=now-timedelta(days=1)
    newer_fixture=fixture(97343,capture,"Unrelated Home","Unrelated Away",
                          delta_hours=22)
    unrelated=shadow._record(newer_fixture,evidence(capture),candidate(),"v8",capture)
    assert append_file_record(unrelated,root=tmp_path,kind="forecasts")
    pair_legs=[]
    for row in (old_a,old_b):
        pair_legs.append({k:row[k] for k in
            ("provider","fixture_id","kickoff_utc","home_team_id","away_team_id")})
    double_folder=tmp_path/"doubles"
    double_folder.mkdir()
    (double_folder/"2026-11-15.json").write_text(json.dumps({
        "mode":"prospective","status":"frozen_pre_kickoff",
        "captured_at_utc":old_a["captured_at_utc"],
        "first":pair_legs[0],"second":pair_legs[1]
    }))
    reference={str(r["fixture_id"]):r for r in (old_a,old_b,unrelated)}
    ordered=settlement._verification_order(reference,tmp_path,now)
    assert [x[0] for x in ordered[:2]]==[
        str(old_a["fixture_id"]),str(old_b["fixture_id"])]
    assert ordered[-1][0]==str(unrelated["fixture_id"])


def test_inconsistent_frozen_pair_cannot_prioritize_wrong_provider_identity(tmp_path):
    a=one_frozen(tmp_path,97351)
    b=one_frozen(tmp_path,97352)
    legs=[{k:r[k] for k in
          ("provider","fixture_id","kickoff_utc","home_team_id","away_team_id")}
          for r in (a,b)]
    legs[1]["away_team_id"]="wrong"
    out=tmp_path/"doubles"
    out.mkdir()
    (out/"2026-11-15.json").write_text(json.dumps({
        "mode":"prospective","status":"frozen_pre_kickoff",
        "captured_at_utc":a["captured_at_utc"],
        "first":legs[0],"second":legs[1]
    }))
    now=datetime.now(timezone.utc)
    refs={str(r["fixture_id"]):r for r in (a,b)}
    ordered=settlement._verification_order(refs,tmp_path,now)
    assert len(ordered)==2
    # A corrupted pair cannot add any new verification reference IDs.
    assert set(x[0] for x in ordered)==set(refs)


def test_systemic_provider_refusals_stop_repeat_requests_without_losing_results(tmp_path):
    cases=[
        (429,{}, "provider_quota","ProviderQuotaError"),
        (403,{}, "provider_credentials_or_plan","ProviderAccessError"),
        (200,{"errors":{"requests":"You have reached the limit of calls for this day"}},
         "provider_quota","ProviderQuotaError"),
        (200,{"errors":{"token":"Invalid API key: sensitive-value"}},
         "provider_credentials_or_plan","ProviderAccessError"),
    ]
    for index,(http,payload,reason,error_type) in enumerate(cases):
        root=tmp_path/str(index)
        root.mkdir()
        one_frozen(root,98001)
        one_frozen(root,98002)
        calls=[]
        def fetch(url,params,headers,timeout):
            calls.append(params["id"])
            reply=Response(payload)
            reply.status_code=http
            return reply
        report=settlement.collect("sensitive-value",root=root,legacy_results={},
                                  request=fetch,limit=20)
        assert len(calls)==report["checked"]==1
        assert report["provider_blocker"]["reason"]==reason
        assert report["source_errors"][0]["error_type"]==error_type
        assert report["deferred_provider_block"]==1
        assert report["still_pending"]==2 and report["newly_verified"]==0
        assert json.loads((root/"verified_results.json").read_text())=={}
        assert "sensitive-value" not in (root/"settlement_status.json").read_text()


def test_fixture_specific_refusal_does_not_block_other_results(tmp_path):
    first=one_frozen(tmp_path,98011)
    second=one_frozen(tmp_path,98012)
    rows={str(r["fixture_id"]):r for r in (first,second)}
    calls=[]
    def fetch(url,params,headers,timeout):
        calls.append(params["id"])
        if len(calls)==1:
            return Response({"errors":{"id":"Fixture not available"}})
        return Response({"response":[fulltime_result(rows[params["id"]])]})
    report=settlement.collect("test",root=tmp_path,legacy_results={},request=fetch)
    assert len(calls)==2
    assert report["provider_blocker"] is None
    assert report["source_errors"][0]["error_type"]=="ProviderRejectedError"
    assert report["newly_verified"]==1 and report["still_pending"]==1
