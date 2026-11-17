"""No-network integration checks for the next Sportmonks daily collection.

These are recorded simulated API payloads, not a claim of live xG entitlement.
The mock response mirrors Sportmonks' documented include=xGFixture shape:
actual entries appear under `expected`, with type 5304 for xG and 5305
for post-shot xGoT. The fixture discovery response is paginated separately.
"""
import json
from datetime import datetime, timedelta, timezone

from sportmonks_xg_research import capture
from sportmonks_xg_validation import validate_packet, verify_packet_collection
from test_sportmonks_xg_research import original, next_match, historical, NOW

NEW_DAY = datetime(2032, 8, 17, 0, 2, tzinfo=timezone.utc)
TOKEN = "TEST-TOKEN-NEVER-STORE"


class FakeResponse:
    status_code = 200

    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


def sportmonks_historical(fixture_id, kickoff, home_id, away_id,
                          *, genuine_xg=True):
    """Use provider-documented `expected` rather than synthetic `xgfixture`."""
    record = historical(fixture_id, kickoff, home_id, away_id)
    record["expected"] = record.pop("xgfixture")
    record["expected"].extend([
        {"fixture_id": fixture_id, "type_id": 5305,
         "participant_id": home_id, "location": "home",
         "data": {"value": 1.73}},
        {"fixture_id": fixture_id, "type_id": 5305,
         "participant_id": away_id, "location": "away",
         "data": {"value": 1.12}},
    ])
    if not genuine_xg:
        record["expected"] = [
            row for row in record["expected"] if row["type_id"] == 5305]
    return record


def fake_api(calls, *, genuine_xg=True):
    def mock_request(url, **kwargs):
        params = kwargs["params"]
        headers = kwargs["headers"]
        assert headers["Authorization"] == TOKEN
        assert kwargs["allow_redirects"] is False
        calls.append((url, dict(params)))
        if "/fixtures/date/" in url:
            assert "/fixtures/date/2032-08-17" in url
            assert "participants" in params["include"]
            return FakeResponse({"data": [next_match()],
                                 "pagination": {"has_more": False}})
        assert params["include"] == "participants;state;xGFixture"
        if url.endswith("/31"):
            row = sportmonks_historical(900, NEW_DAY-timedelta(days=5),
                                        31, 80, genuine_xg=genuine_xg)
        elif url.endswith("/32"):
            row = sportmonks_historical(901, NEW_DAY-timedelta(days=6),
                                        70, 32, genuine_xg=genuine_xg)
        else:
            raise AssertionError("Unexpected Sportmonks URL")
        return FakeResponse({"data": [row]})
    return mock_request


def options():
    return {"SPORTMONKS_XG_ENABLED": "true", "SPORTMONKS_API_TOKEN": TOKEN}


def test_next_utc_day_resets_quota_and_validates_realistic_xg_packet(tmp_path):
    # Simulate the user's fully consumed PREVIOUS UTC day (4/4).
    old = tmp_path / "requests" / "2032-08-16"
    old.mkdir(parents=True)
    for i in range(4):
        (old / f"{i:03d}_sportmonks.json").write_text("{}")
    calls = []
    forecast = original()
    first = capture(now=NEW_DAY, root=tmp_path, originals=[forecast],
                    environ=options(), request=fake_api(calls),
                    clock=lambda: NEW_DAY)
    assert first["reason"] == "genuine_sportmonks_xg_captured"
    assert first["requests_this_run"] == 3
    assert first["daily_requests_used"] == 3
    assert first["captured"] == 1
    assert first["matched_future_fixtures"] == 1
    assert first["matched_historical_xg_fixtures"] == 2
    assert first["last_provider_observation"]["captured"] == 1
    assert len(calls) == 3
    assert len(list(old.glob("*.json"))) == 4
    assert len(list((tmp_path/"requests"/"2032-08-17").glob("*.json"))) == 3

    files = list((tmp_path/"forecasts").glob("*.json"))
    assert len(files) == 1
    packet = json.loads(files[0].read_text())
    assert len(packet["historical_xg"]) == 2
    assert packet["historical_xg_home_count"] == 1
    assert packet["historical_xg_away_count"] == 1
    assert all(row["type_id"] == 5304 for row in packet["historical_xg"])
    assert validate_packet(packet, [forecast]) == {
        "valid": True, "reason": "verified_historical_source_packet",
        "home_history": 1, "away_history": 1,
        "both_teams_have_xg": True,
    }
    audit = verify_packet_collection([packet], [forecast])
    assert audit["source_validated"] == 1
    assert audit["with_both_teams"] == 1
    assert audit["with_six_per_team"] == 0
    assert audit["no_proven_incremental_score_accuracy"] is True
    assert TOKEN not in files[0].read_text()
    assert TOKEN not in json.dumps(first)
    assert TOKEN not in (tmp_path/"status.json").read_text()

    def must_not_call(*args, **kwargs):
        raise AssertionError("Frozen xG evidence must not be refetched")
    later = NEW_DAY+timedelta(minutes=15)
    second = capture(now=later, root=tmp_path, originals=[forecast],
                     environ=options(), request=must_not_call,
                     clock=lambda: later)
    assert second["reason"] == "eligible_forecasts_already_checked"
    assert second["requests_this_run"] == 0
    assert second["last_provider_observation"] == first["last_provider_observation"]


def test_xgot_only_cannot_be_counted_as_genuine_xg(tmp_path):
    calls = []
    result = capture(now=NEW_DAY, root=tmp_path, originals=[original()],
                     environ=options(),
                     request=fake_api(calls, genuine_xg=False),
                     clock=lambda: NEW_DAY)
    assert result["reason"] == "genuine_xg_unavailable_subscription_or_coverage"
    assert result["captured"] == 0
    assert result["matched_historical_xg_fixtures"] == 0
    assert result["subscription_confirmed"] is False
    assert len(calls) == 3
    assert not list((tmp_path/"forecasts").glob("*.json"))
    assert TOKEN not in json.dumps(result)


def test_no_api_calls_before_quota_resets(tmp_path):
    used = tmp_path/"requests"/NEW_DAY.date().isoformat()
    used.mkdir(parents=True)
    for i in range(4):
        (used/f"{i:03d}_sportmonks.json").write_text("{}")
    def must_not_call(*args, **kwargs):
        raise AssertionError("Quota guard should prevent all calls")
    result = capture(now=NEW_DAY, root=tmp_path, originals=[original()],
                     environ=options(), request=must_not_call,
                     clock=lambda: NEW_DAY)
    assert result["reason"] == "daily_request_budget_exhausted"
    assert result["requests_this_run"] == 0
    assert result["daily_requests_used"] == 4
    assert result["captured"] == 0
