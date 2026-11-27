from datetime import datetime,timedelta,timezone
import pytest
import v44_opportunity_audit as audit

NOW=datetime(2025,3,1,12,tzinfo=timezone.utc)


def opportunity(i,issued=False,closed=True):
    end=NOW-timedelta(days=10-i) if closed else NOW+timedelta(hours=1)
    return {"day":str(i),"window_opens_utc":(end-timedelta(hours=2)).isoformat(),
            "window_closes_utc":end.isoformat(),"issued":issued,
            "fixture_ids":[["p",str(2*i)],["p",str(2*i+1)]],
            "events":[[str(2*i)],[str(2*i+1)]]}


def settled(i,hit=True):
    return {"day":str(i),"metrics":{n:{"hit":hit,"estimated_coverage":.2,
                "score_log_loss":2.} for n in audit.trial.METHODS}}


def test_missed_windows_count_without_invented_probability_loss(monkeypatch):
    monkeypatch.setattr(audit,"COHORT_SIZE",3)
    report=audit.summarize([opportunity(0,True),opportunity(1),
                           opportunity(2,True)],[settled(0),settled(2,False)],NOW)
    assert report["fixed_review_complete"]
    assert report["closed_without_trial"]==1
    assert report["closed_window_issuance_rate"]==2/3
    m=report["methods"]["ratings_blend"]
    assert m["verified_issued_hit_rate"]==.5
    assert m["fixed_cohort_success_rate"]==1/3
    assert m["mean_per_leg_log_loss"]==2.
    assert m["target_met"] is False


def test_pending_first_cohort_never_replaced_by_later_success(monkeypatch):
    monkeypatch.setattr(audit,"COHORT_SIZE",2)
    report=audit.summarize([opportunity(2,True),opportunity(1,True),opportunity(0,True)],
                           [settled(1),settled(2)],NOW)
    assert report["issued_pending"]==1
    assert report["verified_issued"]==1
    assert not report["fixed_review_complete"]
    assert report["methods"]["ratings_blend"]["fixed_cohort_success_rate"] is None


def test_open_window_and_repeated_fixture_prevent_claim(monkeypatch):
    monkeypatch.setattr(audit,"COHORT_SIZE",2)
    first=opportunity(0,True)
    second=opportunity(1,False,False)
    second["fixture_ids"]=first["fixture_ids"]
    report=audit.summarize([first,second],[settled(0)],NOW)
    assert report["open_or_future_windows"]==1
    assert not report["integrity_ok"]
    assert not report["fixed_review_complete"]


def test_empty_unregistered_report_is_read_only(tmp_path):
    report=audit.evaluate(root=tmp_path,report={"counts":{},"rows":[]},now=NOW)
    assert report["status"]=="not_registered"
    assert report["summary"]["methods"]["ratings_blend"]["verified_issued_hit_rate"] is None
    assert list(tmp_path.iterdir())==[]


def test_invalid_probability_rejected():
    r=settled(0)
    r["metrics"]["ratings_only"]["estimated_coverage"]=float("nan")
    with pytest.raises(ValueError,match="invalid_model_probability"):
        audit.summarize([opportunity(0,True)],[r],NOW)
