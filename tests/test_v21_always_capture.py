"""Advanced scheduled collection never silently disappears, spends without a cap,
or overwrites original source evidence."""
import json
from datetime import datetime, timedelta, timezone

import v10_prospective
import v21_capture_sources as mod

NOW = datetime(2031, 6, 1, 10, tzinfo=timezone.utc)


def leg(minutes=70):
    return {
        "provider": "API-Football", "fixture_id": "900001",
        "forecast_id": "v8-unique", "input_digest": "frozen-digest",
        "home_team_id": "50", "away_team_id": "51",
        "kickoff_utc": (NOW+timedelta(minutes=minutes)).isoformat(),
        "competition": "PL", "model_key": "v8",
    }


def test_original_selection_widens_to_48_hours_without_past_kickoff(monkeypatch,tmp_path):
    source=leg()
    monkeypatch.setattr(v10_prospective, "read_latest", lambda root: {"a":source})
    monkeypatch.setattr(mod, "original_model", lambda root, saved: dict(saved))
    assert mod.current_pair(tmp_path, NOW) == [source]
    assert mod.current_pair(tmp_path, NOW-timedelta(minutes=21)) == [source]
    assert mod.current_pair(tmp_path, NOW-timedelta(hours=47)) == []
    assert mod.current_pair(tmp_path, NOW+timedelta(minutes=64)) == []


def test_no_key_persists_explicit_reason_without_requests(tmp_path):
    result=mod.capture_pair(root=tmp_path,source_root=tmp_path,
                            now=NOW,api_key="",enabled=True)
    assert result["enabled"] is True
    assert result["reason"] == "API_FOOTBALL_KEY_missing"
    assert json.loads((tmp_path/"status.json").read_text())["reason"] == result["reason"]


def test_no_eligible_leg_persists_reason_even_with_key(monkeypatch,tmp_path):
    monkeypatch.setattr(mod,"current_pair",lambda root,now: [])
    result=mod.capture_pair(root=tmp_path,source_root=tmp_path,
                            now=NOW,api_key="key",enabled=True)
    assert result["reason"] == "no_original_frozen_leg_in_8_min_to_48_hour_window"
    assert (tmp_path/"status.json").exists()


def test_daily_budget_prevents_paid_capture(monkeypatch,tmp_path):
    monkeypatch.setattr(mod, "current_pair", lambda root,now: [leg()])
    monkeypatch.setenv("V21_DAILY_CAPTURE_ATTEMPTS","0")
    def must_not_call(*args,**kwargs):
        raise AssertionError("No API calls permitted with zero budget")
    monkeypatch.setattr(mod,"load_evidence",must_not_call)
    result=mod.capture_pair(root=tmp_path,source_root=tmp_path,
                            now=NOW,api_key="key",enabled=True)
    assert result["daily_attempts_limit"]==0
    assert result["attempted_legs"]==0
    assert result["skipped"] == ["900001:daily_attempt_budget_exhausted"]
    assert result["reason"]=="daily_attempt_budget_exhausted"


def test_early_packet_preserved_with_one_near_kickoff_refresh(monkeypatch,tmp_path):
    saved=leg()
    monkeypatch.setattr(mod,"current_pair",lambda root,now:[saved])
    monkeypatch.setattr(mod,"load_evidence",lambda *args,**kwargs:
                        {"history":[],"input_digest":"history-source"})
    monkeypatch.setattr(mod,"valid_snapshot",lambda *args,**kwargs:True)
    def source(*args,**kwargs):
        return {"snapshot_sha256":"a"*64, "captured_at_utc":kwargs["now"].isoformat(),
                "available_sources":{"historical_xg":True},
                "verified_historical_fixture_xg":[],
                "confirmed_starting_xi":[], "multi_market":{},
                "diagnostics":{"queries":3}}
    monkeypatch.setattr(mod,"capture",source)
    monkeypatch.setenv("V22_DEEP_XG_ENABLED","true")
    monkeypatch.setenv("V21_DAILY_CAPTURE_ATTEMPTS","6")
    first=mod.capture_pair(root=tmp_path,source_root=tmp_path,now=NOW,
                           api_key="key",enabled=True)
    assert first["new_packets"]==["900001"]
    base=mod.source_packet_path(tmp_path,saved)
    original=base.read_text()
    # 70 minutes at initial capture, 25 minutes on second capture.
    second=mod.capture_pair(root=tmp_path,source_root=tmp_path,
                            now=NOW+timedelta(minutes=45),
                            api_key="key",enabled=True)
    assert second["attempted_legs"]==1
    assert len(list((tmp_path/"versions").glob("*.json")))==1
    assert base.read_text()==original
    # A third scheduled pass does not silently requery every 15 minutes.
    third=mod.capture_pair(root=tmp_path,source_root=tmp_path,
                           now=NOW+timedelta(minutes=50),
                           api_key="key",enabled=True)
    assert third["attempted_legs"]==0
    assert third["skipped"]==["900001:refresh_already_attempted"]
    usage=json.loads((tmp_path/"usage"/"2031-06-01.json").read_text())
    assert usage["attempts"]==2
