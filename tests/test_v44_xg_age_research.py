import json
from datetime import datetime, timedelta, timezone

import v44_xg_age_research as research


def test_historical_age_does_not_relabel_packet(tmp_path,monkeypatch):
    cutoff=datetime(2025,1,1,12,tzinfo=timezone.utc)
    model={"fixture_id":"1","forecast_id":"source"}
    path=tmp_path/"source.json"
    captured=(cutoff-timedelta(hours=8)).isoformat()
    path.write_text(json.dumps({"provider_snapshot":{"captured_at_utc":captured,
                        "snapshot_sha256":"original","source_limits":{"max_historical_fixtures_per_team":6}}}))
    monkeypatch.setattr(research.sources,"source_packet_path",lambda *a:path)
    monkeypatch.setattr(research.sources,"qualified_packet",lambda *a,**k:True)
    monkeypatch.setattr(research,"_profile",lambda *a:{"ready":True})
    strict, extended, counts=research._find([model],tmp_path,cutoff)
    assert strict is None
    assert extended[1]["captured_at_utc"]==captured
    assert extended[1]["source_forecast_id"]=="source"
    assert counts["packet_files"]==1


def test_future_packet_and_small_capture_are_diagnosed(tmp_path,monkeypatch):
    cutoff=datetime(2025,1,1,12,tzinfo=timezone.utc)
    model={"fixture_id":"1","forecast_id":"source"}
    path=tmp_path/"source.json"
    monkeypatch.setattr(research.sources,"source_packet_path",lambda *a:path)
    monkeypatch.setattr(research.sources,"qualified_packet",lambda *a,**k:True)
    monkeypatch.setattr(research,"_profile",lambda *a:{"ready":False,"reason":"insufficient_genuine_pre_match_xg"})
    for hours,expected in [(1,"outside_historical_age_window"),(-1,"capture_cap_below_six")]:
        path.write_text(json.dumps({"provider_snapshot":{
            "captured_at_utc":(cutoff+timedelta(hours=hours)).isoformat(),
            "snapshot_sha256":"original","source_limits":{"max_historical_fixtures_per_team":1}}}))
        strict,extended,counts=research._find([model],tmp_path,cutoff)
        assert strict is None and extended is None
        assert counts[expected]==1


def test_non_xg_rows_never_qualify():
    cutoff=datetime(2025,1,1,12,tzinfo=timezone.utc)
    packet={"captured_at_utc":cutoff.isoformat(),"verified_historical_fixture_xg":[
        {"fixture_id":"old","type":"shot_proxy"}]}
    assert not research._profile(packet,{"fixture_id":"1"},cutoff)["ready"]
