from datetime import datetime,timedelta,timezone
import v44_venue_collector as collector

NOW=datetime(2025,2,1,10,tzinfo=timezone.utc)
FIXTURE={"provider":"API-Football","id":999,
         "utcDate":"2025-02-01T12:00:00Z","homeTeam":{"id":1},"awayTeam":{"id":2}}


def history():
    def match(i,h,a,days):
        return {"provider":"API-Football","status":"FINISHED","id":i,
                "utcDate":(NOW-timedelta(days=days)).isoformat(),
                "homeTeam":{"id":h},"awayTeam":{"id":a}}
    return ([match(i+10,3,1,i+1) for i in range(6)]
            +[match(i+20,2,4,i+1) for i in range(6)]
            +[match(i+30,1,3,i+8) for i in range(3)]
            +[match(i+40,4,2,i+8) for i in range(3)])


def test_adapter_executes_planned_statistics_requests():
    calls=[]
    def request(key,endpoint,args):
        calls.append((endpoint,dict(args)))
        return []
    report=collector.capture_venue_aware(FIXTURE,history(),"fake",enabled=True,
                  reserved_adapter_calls=16,now=NOW,clock=lambda:NOW,request=request)
    assert report["status"]=="captured"
    assert report["sampling"]["executable"]["potentially_ready"]
    assert not report["sampling"]["fallback_to_recent"]
    stats=[str(args["fixture"]) for endpoint,args in calls if endpoint=="/fixtures/statistics"]
    assert set(stats)==set(report["sampling"]["executable"]["fixture_ids"])
    assert len(stats)==12 and len(calls)==16
    assert report["provider_snapshot"]["verified_historical_fixture_xg"]==[]


def test_opt_in_and_reservation_before_requests():
    def request(*args,**kwargs):
        raise AssertionError("must not call provider")
    for enabled,reserved,reason in [(False,16,"research_collection_disabled"),
                                    (True,15,"insufficient_reserved_adapter_calls")]:
        report=collector.capture_venue_aware(FIXTURE,history(),"fake",enabled=enabled,
                      reserved_adapter_calls=reserved,now=NOW,clock=lambda:NOW,request=request)
        assert report["reason"]==reason


def test_cutoff_prevents_late_capture():
    report=collector.capture_venue_aware(FIXTURE,history(),"fake",enabled=True,
                   reserved_adapter_calls=16,now=NOW+timedelta(hours=2))
    assert report["reason"]=="outside_pre_match_window"
