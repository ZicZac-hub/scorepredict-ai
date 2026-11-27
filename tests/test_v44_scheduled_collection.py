from datetime import datetime,timedelta,timezone
import json

import api_football
import v21_capture_sources_v2 as scheduled
import v44_venue_collector as venue

NOW=datetime(2025,2,1,10,tzinfo=timezone.utc)


def _setup(monkeypatch):
    comp=next(iter(api_football.LEAGUES))
    leg={"provider":"API-Football","fixture_id":"101","forecast_id":"original",
         "input_digest":"original-input","home_team_id":"11","away_team_id":"22",
         "home_team":"Home","away_team":"Away","competition":comp,
         "kickoff_utc":(NOW+timedelta(hours=1)).isoformat()}
    monkeypatch.setattr(scheduled,"current_pair",lambda *a,**k:[leg])
    monkeypatch.setattr(scheduled,"targeted_pair_models",lambda **k:[])
    monkeypatch.setattr(scheduled,"best_packet",lambda *a,**k:None)
    monkeypatch.setattr(scheduled,"source_availability",lambda *a,**k:{"refresh_reasons":[]})
    monkeypatch.setattr(scheduled,"load_evidence",lambda *a,**k:{"history":[]})
    monkeypatch.setattr(scheduled,"valid_snapshot",lambda *a,**k:True)
    monkeypatch.setenv("V22_DEEP_XG_ENABLED","true")
    monkeypatch.setenv("V21_DAILY_CAPTURE_ATTEMPTS","6")
    return {"snapshot_sha256":"a"*64,"available_sources":{"historical_xg":True},
            "diagnostics":{"queries":16}}


def test_scheduled_opt_in_uses_existing_reserved_attempt(tmp_path,monkeypatch):
    packet=_setup(monkeypatch)
    monkeypatch.setenv("V44_VENUE_COLLECTION_ENABLED","true")
    calls=[]
    def adapter(*a,**k):
        calls.append(k)
        return {"status":"captured","provider_snapshot":packet,"sampling":{"policy":"venue"}}
    monkeypatch.setattr(venue,"capture_venue_aware",adapter)
    def legacy(*a,**k):
        raise AssertionError("must not also run legacy capture")
    monkeypatch.setattr(scheduled,"capture",legacy)
    report=scheduled.capture_pair(root=tmp_path,source_root=tmp_path,pair_root=tmp_path,
                 archive_root=tmp_path,now=NOW,clock=lambda:NOW,api_key="fake",enabled=True)
    assert len(calls)==1 and calls[0]["reserved_adapter_calls"]==16
    assert report["attempted_legs"]==1 and report["daily_attempts_used"]==1
    assert report["new_packets"]==["101"]
    assert report["v44_sampling"]["101"]["status"]=="captured"
    assert report["calls"]==16


def test_disabled_keeps_legacy_path(tmp_path,monkeypatch):
    packet=_setup(monkeypatch)
    monkeypatch.delenv("V44_VENUE_COLLECTION_ENABLED",raising=False)
    monkeypatch.setattr(scheduled,"capture",lambda *a,**k:packet)
    report=scheduled.capture_pair(root=tmp_path,source_root=tmp_path,pair_root=tmp_path,
                 archive_root=tmp_path,now=NOW,clock=lambda:NOW,api_key="fake",enabled=True)
    assert report["new_packets"]==["101"]
    assert "v44_sampling" not in report


def test_failed_adapter_no_second_attempt(tmp_path,monkeypatch):
    _setup(monkeypatch)
    monkeypatch.setenv("V44_VENUE_COLLECTION_ENABLED","true")
    monkeypatch.setattr(venue,"capture_venue_aware",lambda *a,**k:
                       {"status":"abstained","reason":"unqualified_provider_snapshot","adapter_calls":8})
    def legacy(*a,**k):
        raise AssertionError("unexpected second capture")
    monkeypatch.setattr(scheduled,"capture",legacy)
    report=scheduled.capture_pair(root=tmp_path,source_root=tmp_path,pair_root=tmp_path,
                 archive_root=tmp_path,now=NOW,clock=lambda:NOW,api_key="fake",enabled=True)
    assert report["calls"]==8 and report["daily_attempts_used"]==1
    assert report["new_packets"]==[]
