from datetime import datetime,timedelta,timezone
from v44_source_settlement_audit import time_reason,timing,evaluate

NOW=datetime(2035,1,1,tzinfo=timezone.utc)

def row(e,c,k):
    return {"model_key":"v8","provider":"API-Football","fixture_id":"1","home_team_id":"2","away_team_id":"3","forecast_id":"f",
            "evidence_as_of_utc":(NOW+timedelta(hours=e)).isoformat(),
            "captured_at_utc":(NOW+timedelta(hours=c)).isoformat(),
            "kickoff_utc":(NOW+timedelta(hours=k)).isoformat()}

def test_time_reasons_keep_expired_separate_from_corruption():
    cases=[(row(-5,-4,-1),"fixture_already_started"),(row(1,0,2),"evidence_after_capture"),
           (row(-1,1,1),"capture_at_or_after_kickoff"),(row(0,1,2),"capture_after_cutoff")]
    for r,expected in cases:
        assert time_reason(r,NOW)==expected
    assert time_reason(row(-2,-1,1),NOW)=="time_valid"
    result=timing([r for r,_ in cases],NOW)
    assert result["invalid_source_time_total"]==4
    assert result["breakdown"]=={expected:1 for _,expected in cases}

def test_empty_archive_is_read_only(tmp_path):
    report=evaluate(tmp_path/"original",tmp_path/"prematch",NOW)
    assert report["source_timing"]["invalid_source_time_total"]==0
    assert report["v37"]["unique_fixtures"]==0
    assert report["external_requests"]==0
    assert not report["automatic_promotion"]
    assert not list(tmp_path.iterdir())
