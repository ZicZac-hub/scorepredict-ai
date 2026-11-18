"""Regressions for verified odds, source timing and separate V25 evidence packets."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import hashlib
import json

import numpy as np

from v21_capture_sources import best_packet, source_packet_path
from v23_market_form import _qualified_markets
from v25_evidence_audit import _fresh_market, trace_leg
from v25_upcoming import qualified


NOW = datetime(2031, 11, 18, 10, tzinfo=timezone.utc)
KICK = NOW + timedelta(hours=5)


def stamp(hours_ago):
    return (NOW - timedelta(hours=hours_ago)).isoformat()


def markets():
    return {
        "1x2": {"last_source_update_utc": stamp(1)},
        "over_2_5": {"last_source_update_utc": stamp(1)},
        "over_3_5": {"last_source_update_utc": stamp(18)},
    }


def source_and_fixture():
    source = {"provider": "API-Football", "fixture_id": "101",
              "forecast_id": "frozen101", "input_digest": "digest101",
              "home_team_id": "201", "away_team_id": "202",
              "kickoff_utc": KICK.isoformat()}
    fixture = {"provider": "API-Football", "id": "101",
               "utcDate": KICK.isoformat(),
               "homeTeam": {"id": "201", "name": "Home"},
               "awayTeam": {"id": "202", "name": "Away"}}
    return source, fixture


def record(market_data, *, captured=None, xg=None):
    source, _ = source_and_fixture()
    packet = {"status": "captured_pre_kickoff", "provider": "API-Football",
              "fixture_id": source["fixture_id"],
              "home_team_id": source["home_team_id"],
              "away_team_id": source["away_team_id"],
              "kickoff_utc": source["kickoff_utc"],
              "captured_at_utc": captured or NOW.isoformat(),
              "verified_historical_fixture_xg": xg or [],
              "confirmed_starting_xi": [], "reported_injuries": [],
              "multi_market": market_data}
    raw = json.dumps(packet, sort_keys=True, separators=(",",":"),
                     allow_nan=False)
    packet["snapshot_sha256"] = hashlib.sha256(raw.encode()).hexdigest()
    return {"source_forecast": source, "provider_snapshot": packet}


def test_v25_requires_goal_totals_not_just_winner_market():
    with_totals = record(markets())["provider_snapshot"]
    assert _fresh_market(with_totals, NOW, KICK) is True
    without_totals = record({"1x2": markets()["1x2"]})["provider_snapshot"]
    assert _fresh_market(without_totals, NOW, KICK) is False
    only_stale_totals = record({
        "1x2": markets()["1x2"],
        "over_2_5": markets()["over_3_5"],
    })["provider_snapshot"]
    assert _fresh_market(only_stale_totals, NOW, KICK) is False


def test_market_update_after_snapshot_is_not_time_travelled():
    bad = record(markets(), captured=stamp(2))["provider_snapshot"]
    assert _fresh_market(bad, NOW, KICK) is False
    _, fixture = source_and_fixture()
    checked, diagnostic = _qualified_markets(bad, fixture, NOW)
    assert "1x2" not in checked
    assert diagnostic["missing_required_1x2"] is True


def test_v25_audit_reads_market_version_without_overwriting_history():
    source, fixture = source_and_fixture()
    xg = [
        {"fixture_id": "501", "home_team_id": "201", "away_team_id": "301",
         "type": "api_football_reported_expected_goals"},
        {"fixture_id": "502", "home_team_id": "202", "away_team_id": "302",
         "type": "api_football_reported_expected_goals"},
    ]
    researched = record({"1x2": markets()["1x2"]}, xg=xg)
    current_market = record(markets())
    before = deepcopy(researched)
    requests = []

    def load(_root, _source, *, now, priority):
        assert _source == source and now == NOW
        requests.append(priority)
        return researched if priority == "research" else current_market

    actual = trace_leg({"row": {"fixture": fixture,
                                "source_forecast": source}}, NOW,
                       packet_loader=load)
    assert requests == ["research", "market"]
    assert actual["matching_provider_packet"] is True
    assert actual["verified_historical_xg_both_teams"] is True
    assert actual["verified_pre_match_market"] is True
    assert actual["market_snapshot_at_utc"] == NOW.isoformat()
    assert actual["independent_context_applied_to_v25_model"] is False
    assert researched == before


def test_best_packet_keeps_good_goal_total_despite_stale_optional_total(tmp_path):
    source, _ = source_and_fixture()
    valid = record(markets())
    # Both packets have exact authenticated identity and were captured before
    # kickoff. Only one contains a genuinely fresh total.
    obsolete = record({"1x2": {"last_source_update_utc": stamp(13)},
                       "over_2_5": {"last_source_update_utc": stamp(13)}},
                      captured=stamp(1))
    canonical = source_packet_path(tmp_path, source)
    canonical.write_text(json.dumps(obsolete), encoding="utf-8")
    versions = tmp_path / "versions"
    versions.mkdir()
    (versions / (canonical.stem + "_current.json")).write_text(
        json.dumps(valid), encoding="utf-8")
    chosen = best_packet(tmp_path, source, now=NOW, priority="market")
    assert chosen["provider_snapshot"]["snapshot_sha256"] == (
        valid["provider_snapshot"]["snapshot_sha256"])


def test_v25_rejects_unstamped_live_grid_instead_of_inventing_capture_time():
    _, fixture = source_and_fixture()
    grid = np.ones((8, 8), dtype=float) / 64
    row = {"fixture": fixture, "kickoff": fixture["utcDate"],
           "result": {"matrix": grid.tolist()}}
    _, why = qualified(row, NOW)
    assert why == "invalid_or_future_source_timestamp"
    row["generated_at_utc"] = NOW.isoformat()
    assert qualified(row, NOW)[0] is not None
