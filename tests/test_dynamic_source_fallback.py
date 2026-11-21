"""Dynamic fallback evidence cannot masquerade as a fresh captured source."""
from datetime import datetime, timedelta, timezone
import hashlib
import json

import pytest
import requests

import api_football as api
from v21_provider_evidence import capture

NOW = datetime(2035, 6, 1, 10, tzinfo=timezone.utc)
KEY = "nonsecret-regression-key"
DYNAMIC = ("/fixtures/lineups", "/injuries", "/predictions", "/odds")


def prepare(monkeypatch, tmp_path):
    monkeypatch.setattr(api, "CACHE", tmp_path)
    monkeypatch.setattr(api.time, "time", lambda: NOW.timestamp())
    monkeypatch.setattr(api, "_QUOTA_BLOCK_UNTIL", {})
    def unavailable(*args, **kwargs):
        raise requests.Timeout("regression timeout")
    monkeypatch.setattr(api.requests, "get", unavailable)


def cache(endpoint, params, rows, *, seconds_old):
    signature = json.dumps([endpoint, params, hashlib.sha256(KEY.encode()).hexdigest()], sort_keys=True)
    path = api.CACHE/(hashlib.sha256(signature.encode()).hexdigest()+".json")
    fetched = NOW.timestamp()-seconds_old
    path.write_text(json.dumps({"time": fetched, "data": rows}))
    return path, fetched


@pytest.mark.parametrize("endpoint", DYNAMIC)
def test_dynamic_stale_rows_are_unavailable_without_metadata(monkeypatch, tmp_path, endpoint):
    prepare(monkeypatch, tmp_path)
    params = {"fixture": "101"}
    original = [{"original": "expired-source"}]
    path, _ = cache(endpoint, params, original, seconds_old=api.cache_ttl(endpoint, params)+1)
    archived = path.read_bytes()
    with pytest.raises(RuntimeError, match="Timeout"):
        api._request(KEY, endpoint, params)
    assert path.read_bytes() == archived


@pytest.mark.parametrize("endpoint", DYNAMIC)
def test_metadata_aware_fallback_keeps_actual_old_observation(monkeypatch, tmp_path, endpoint):
    prepare(monkeypatch, tmp_path)
    params = {"fixture": "101"}
    rows = [{"original": "expired-source"}]
    age = api.cache_ttl(endpoint, params)+1
    _, fetched = cache(endpoint, params, rows, seconds_old=age)
    result = api._request(KEY, endpoint, params, with_metadata=True)
    assert result["data"] == rows
    assert result["metadata"]["stale"] is True
    assert result["metadata"]["cache_status"] == "stale_fallback"
    assert result["metadata"]["fetched_at_utc"] == datetime.fromtimestamp(fetched, timezone.utc).isoformat()
    assert result["metadata"]["age_seconds"] == age


@pytest.mark.parametrize("endpoint", DYNAMIC)
def test_fresh_dynamic_cache_does_not_make_an_extra_request(monkeypatch, tmp_path, endpoint):
    prepare(monkeypatch, tmp_path)
    def must_not_request(*args, **kwargs):
        raise AssertionError("Fresh evidence should reuse the existing cache")
    monkeypatch.setattr(api.requests, "get", must_not_request)
    rows = [{"source": "fresh-cache"}]
    params = {"fixture": "101"}
    cache(endpoint, params, rows, seconds_old=1)
    assert api._request(KEY, endpoint, params) == rows


def test_historical_statistics_fallback_remains_compatible(monkeypatch, tmp_path):
    prepare(monkeypatch, tmp_path)
    endpoint, params = "/fixtures/statistics", {"fixture": "99"}
    rows = [{"team": {"id": 1}, "statistics": []}]
    cache(endpoint, params, rows, seconds_old=api.cache_ttl(endpoint, params)+1)
    assert api._request(KEY, endpoint, params) == rows


@pytest.mark.parametrize("status,error", [(403, api.ProviderAccessError), (429, api.ProviderQuotaError)])
def test_stale_dynamic_cache_does_not_hide_provider_access_or_quota_errors(monkeypatch, tmp_path, status, error):
    prepare(monkeypatch, tmp_path)
    params = {"fixture": "101"}
    cache("/injuries", params, [{"old": "source"}],
          seconds_old=api.cache_ttl("/injuries", params)+1)
    class Rejected:
        status_code = status
    monkeypatch.setattr(api.requests, "get", lambda *args, **kwargs: Rejected())
    with pytest.raises(error):
        api._request(KEY, "/injuries", params)


def test_capture_does_not_relabel_stale_dynamic_evidence_as_fresh(monkeypatch, tmp_path):
    prepare(monkeypatch, tmp_path)
    params = {"fixture": "101"}
    lineups = [{"team": {"id": team}, "startXI": [
        {"player": {"name": f"Player {team}-{i}"}} for i in range(11)]} for team in (1, 2)]
    injuries = [{"team": {"id": 1}, "player": {"name": "Expired injury", "reason": "Unfit"}}]
    for endpoint, rows in (("/fixtures/lineups", lineups), ("/injuries", injuries), ("/odds", [])):
        cache(endpoint, params, rows, seconds_old=api.cache_ttl(endpoint, params)+1)
    fixture = {"provider": "API-Football", "id": 101,
               "utcDate": (NOW+timedelta(hours=2)).isoformat(),
               "homeTeam": {"id": 1}, "awayTeam": {"id": 2}}
    result = capture(fixture, [], KEY, now=NOW, clock=lambda: NOW, max_history_per_team=0)
    assert result["confirmed_starting_xi"] == []
    assert result["reported_injuries"] == []
    assert result["available_sources"]["lineups"] is False
    assert result["diagnostics"]["unavailable"] == {
        "/fixtures/lineups": "RuntimeError", "/injuries": "RuntimeError",
        "/odds": "stale_or_undated_api_response"}
