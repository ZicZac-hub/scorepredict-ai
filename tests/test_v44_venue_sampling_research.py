from datetime import datetime,timedelta,timezone
import v44_venue_sampling_research as research


def _fixture():
    return {"id":"target","utcDate":"2025-02-01T12:00:00Z",
            "homeTeam":{"id":"h"},"awayTeam":{"id":"a"}}


def _match(i,h,a,days):
    t=datetime(2025,2,1,12,tzinfo=timezone.utc)-timedelta(days=days)
    return {"provider":"API-Football","status":"FINISHED","id":str(i),
            "utcDate":t.isoformat(),"homeTeam":{"id":h},"awayTeam":{"id":a}}


def test_venue_balance_with_no_extra_requests():
    # Six recent games in the wrong venue, followed by three in the right venue.
    history=[_match(i,"x","h",i+1) for i in range(6)]
    history += [_match(i+10,"a","y",i+1) for i in range(6)]
    history += [_match(i+20,"h","x",i+8) for i in range(3)]
    history += [_match(i+30,"y","a",i+8) for i in range(3)]
    report=research.plan(_fixture(),history,"2025-02-01T10:00:00Z")
    assert not report["recent"]["potentially_ready"]
    assert report["venue_aware"]["potentially_ready"]
    assert report["recent"]["requests"]==report["venue_aware"]["requests"]==12


def test_future_and_conflicting_rows_excluded():
    history=[_match(1,"h","x",1),_match(1,"h","y",1),_match(2,"h","x",0)]
    report=research.plan(_fixture(),history,"2025-02-01T10:00:00Z")
    assert report["eligible_history_fixtures"]==0
    assert report["venue_aware"]["requests"]==0


def test_small_budget_never_silently_increased():
    history=[_match(i,"h","x",i+1) for i in range(8)]
    history += [_match(i+10,"y","a",i+1) for i in range(8)]
    report=research.plan(_fixture(),history,"2025-02-01T10:00:00Z",limit=1)
    assert report["venue_aware"]["requests"]==2
    assert not report["venue_aware"]["potentially_ready"]
    assert report["additional_requests"]==0
