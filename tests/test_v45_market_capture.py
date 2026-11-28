"""Near-kickoff market observations share the existing odds allowance."""
from datetime import timedelta

import api_football as api
import pytest
from test_professional_double_shadow import freeze
from test_v24_bulk_market_capture import market_item
from v10_prospective import utc
from v42_research_trial import _pair
import v45_market_capture as markets


def odds_response(model, fetched, **metadata):
    fixture = market_item(int(model["fixture_id"]), model["kickoff_utc"])
    fixture["update"] = (fetched - timedelta(minutes=10)).isoformat()
    return {"metadata": {"provider": "API-Football", "stale": False,
                         "fetched_at_utc": fetched.isoformat(), **metadata},
            "data": [fixture]}


def setup_capture(tmp_path):
    originals, pair_root, path, record = freeze(tmp_path)
    _, models = _pair(path, source_root=originals, decision=utc(record["predicted_at_utc"]))
    kickoff = min(utc(model["kickoff_utc"]) for model in models)
    config = dict(source_root=originals, pair_root=pair_root, api_key="fake-key", environ={},
                  root=tmp_path / "receipts", broad_root=tmp_path / "broad")
    return config, models, record["record_sha256"], kickoff


def test_early_and_late_capture_are_fixed_pair_time_bound_and_write_once(tmp_path):
    config, models, pair_sha, kickoff = setup_capture(tmp_path)
    early, late = kickoff - timedelta(minutes=28), kickoff - timedelta(minutes=13)
    by_fixture = {int(model["fixture_id"]): model for model in models}
    calls = []

    def collect(now):
        def fake(key, endpoint, params, *, with_metadata, max_cache_age_seconds):
            assert endpoint == "/odds" and with_metadata and max_cache_age_seconds == 300
            calls.append((now, params["fixture"]))
            return odds_response(by_fixture[params["fixture"]], now)
        return markets.capture(**config, now=now, clock=lambda: now, request=fake)

    for now, phase in [(early, "early"), (late, "late")]:
        saved = collect(now)
        assert saved["phase"] == phase and saved["requests"] == 2
        assert len(saved["receipts"]) == 2
        assert collect(now)["reason"] == "already_attempted"
    assert len(calls) == 4
    assert all(by_fixture[fid] in models for _, fid in calls)
    assert len(list((config["root"] / "versions").glob("*.json"))) == 4
    assert len(list(config["broad_root"].glob("requests/*/v45_target_*.json"))) == 4

    ref = dict(provider_root=config["root"], pair_sha256=pair_sha, earliest_kickoff=kickoff)
    for model in models:
        old = markets.best_market(model, cutoff=early, **ref)
        latest = markets.best_market(model, cutoff=late, **ref)
        assert utc(old["reference"]["captured_at_utc"]) == early
        assert utc(latest["reference"]["captured_at_utc"]) == late
        assert set(latest["value"]) == {"1x2", "over_2_5"}
        assert markets.load_market_reference(model, old["reference"], cutoff=late, **ref) == old
        with pytest.raises(ValueError, match="unverifiable_targeted_market_reference"):
            markets.load_market_reference(model, latest["reference"], cutoff=early, **ref)
        assert markets.best_market(model, cutoff=late, **{**ref, "pair_sha256": "other"}) is None


def test_broad_cap_keeps_targeted_reservations_inside_existing_cap(tmp_path):
    config, models, _, kickoff = setup_capture(tmp_path)
    at = kickoff - timedelta(minutes=28)
    broad_root = config["broad_root"]
    day = broad_root / "requests" / at.date().isoformat()
    day.mkdir(parents=True)
    for n in range(8):
        (day / f"existing_broad_{n}.json").write_text("{}")
    assert markets.broad_daily_limit(broad_root=broad_root, now=at, environ={}) == 8
    assert markets.broad_daily_limit(broad_root=broad_root, now=at, environ={}, enabled=False) == 12
    by_fixture = {int(model["fixture_id"]): model for model in models}
    calls = []

    def fake(key, endpoint, params, **kwargs):
        calls.append(params["fixture"])
        return odds_response(by_fixture[params["fixture"]], at)

    saved = markets.capture(**config, now=at, clock=lambda: at, request=fake)
    assert saved["shared_requests_used"] == 10 and len(calls) == 2
    assert markets.broad_daily_limit(broad_root=broad_root, now=at, environ={}) == 10
    (day / "additional_old_broad.json").write_text("{}")
    late = kickoff - timedelta(minutes=13)
    stopped = markets.capture(**config, now=late, clock=lambda: late, request=fake)
    assert stopped["reason"] == "daily_request_budget_exhausted" and len(calls) == 2


@pytest.mark.parametrize("problem", ["stale", "stale_quote", "missing_quote_time", "kickoff",
                                     "late_reply", "missing_goal_total"])
def test_markets_reject_stale_identity_unmatched_or_late_quotes(tmp_path, problem):
    config, models, _, kickoff = setup_capture(tmp_path)
    at = kickoff - timedelta(minutes=13)
    current = at
    by_fixture = {int(model["fixture_id"]): model for model in models}

    def fake(key, endpoint, params, **kwargs):
        nonlocal current
        row = odds_response(by_fixture[params["fixture"]], at)
        if problem == "stale":
            row["metadata"]["stale"] = True
        elif problem == "stale_quote":
            row["data"][0]["update"] = (at - timedelta(hours=3, minutes=5)).isoformat()
        elif problem == "missing_quote_time":
            row["data"][0].pop("update")
        elif problem == "kickoff":
            row["data"][0]["fixture"]["date"] = at.isoformat()
        elif problem == "late_reply":
            current = kickoff - timedelta(minutes=4)
        else:
            row["data"][0]["bookmakers"][0]["bets"] = row["data"][0]["bookmakers"][0]["bets"][:1]
        return row

    result = markets.capture(**config, now=at, clock=lambda: current, request=fake)
    assert not result["receipts"] and result["requests"] >= 1
    assert sum(result["rejections"].values()) >= 1


def test_opt_in_shorter_cache_does_not_reuse_earlier_phase(monkeypatch, tmp_path):
    monkeypatch.setattr(api, "CACHE", tmp_path)
    current = [2_000_000_000]
    monkeypatch.setattr(api.time, "time", lambda: current[0])
    requests = []

    class Response:
        status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            return {"response": [{"written_at": current[0]}]}

    def fake(*args, **kwargs):
        requests.append(current[0])
        return Response()

    monkeypatch.setattr(api.requests, "get", fake)
    params = {"fixture": 7}
    initial = api._request("fake-key", "/odds", params, with_metadata=True)
    current[0] += 15 * 60
    archived = api._request("fake-key", "/odds", params, with_metadata=True)
    assert archived["data"] == initial["data"] and len(requests) == 1
    newer = api._request("fake-key", "/odds", params, with_metadata=True, max_cache_age_seconds=300)
    assert newer["data"] != initial["data"] and len(requests) == 2
