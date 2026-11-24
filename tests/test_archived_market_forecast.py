"""Read-only market adjustments qualify each regular-time forecast separately."""

from datetime import datetime, timedelta, timezone
import json

import numpy as np
import pytest

from archived_market_forecast import VERSION, adjust_forecast, archived_market
from v24_bulk_market_capture import _hash


NOW = datetime(2028, 1, 15, 9, tzinfo=timezone.utc)
KICKOFF = NOW + timedelta(hours=3)
CAPTURED = NOW - timedelta(minutes=10)
UPDATED = CAPTURED - timedelta(minutes=10)
FIXTURE = {"provider": "API-Football", "id": 9001, "utcDate": KICKOFF.isoformat(),
           "homeTeam": {"id": 2}, "awayTeam": {"id": 5},
           "competition": {"code": "PL", "id": 39}}


def _markets():
    return {
        "1x2": {"probabilities": {"home": .60, "draw": .24, "away": .16},
                "last_source_update_utc": UPDATED.isoformat()},
        "over_2_5": {"probabilities": {"yes": .60, "no": .40},
                     "last_source_update_utc": UPDATED.isoformat()},
    }


def _write(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record), encoding="utf-8")
    return path


def _catalog(root, **overrides):
    row = {"schema": 1, "provider": "API-Football", "fixture_id": "9001",
           "source_fixture_kickoff_utc": KICKOFF.isoformat(), "source_league_id": 39,
           "captured_at_utc": CAPTURED.isoformat(), "provider_fetched_at_utc": UPDATED.isoformat(),
           "market_types": _markets(), **overrides}
    digest = _hash(row)
    row["snapshot_sha256"] = digest
    return _write(root / "fixtures" / "9001" / f"{digest}.json", row)


def _sources(tmp_path):
    return {"catalog_root": tmp_path / "catalog", "source_root": tmp_path / "sources",
            "forecast_root": tmp_path / "forecasts"}


def _forecast():
    grid = np.full((5, 5), 1 / 25)
    return {"model_version": "original-v8", "matrix": grid, "scores": [(0, 0, .04)],
            "home_xg": 2., "away_xg": 2., "components": {"v6": {"predicted_score": (0, 0)}}}


def test_catalog_uses_the_tried_markets_and_recomputes_consistent_probabilities(tmp_path, monkeypatch):
    import requests

    def no_network(*args, **kwargs):
        raise AssertionError("Archived adjustment must make zero provider requests")

    monkeypatch.setattr(requests.sessions.Session, "request", no_network)
    sources = _sources(tmp_path)
    markets = _markets()
    markets["over_3_5"] = {"probabilities": {"yes": .45, "no": .55},
                           "last_source_update_utc": UPDATED.isoformat()}
    receipt = _catalog(sources["catalog_root"], market_types=markets)
    baseline = _forecast()
    display = adjust_forecast(FIXTURE, baseline, now=NOW, **sources)

    assert display["model_version"] == VERSION
    assert display["market_used"] is True
    assert display["market_types"] == ["1x2", "over_2_5"]
    assert display["market_details"]["snapshot_sha256"] == receipt.stem
    assert display["home_win"] > .40
    assert sum(display[key] for key in ("home_win", "draw", "away_win")) == pytest.approx(1)
    home, away, score_probability = display["scores"][0]
    assert (home, away) == np.unravel_index(display["matrix"].argmax(), (5, 5))
    assert score_probability == display["matrix"][home, away]
    assert 0 < display["market_total_variation"] <= .150000001
    assert np.all(baseline["matrix"] == .04)
    assert baseline["scores"] == [(0, 0, .04)]
    assert adjust_forecast({**FIXTURE, "provider": "Sportmonks"}, baseline, now=NOW, **sources) is baseline
    assert adjust_forecast(FIXTURE, baseline, now=KICKOFF, **sources) is baseline


@pytest.mark.parametrize("reason", ["kickoff", "league", "future", "stale", "tampered",
                                    "quote_after_capture", "missing_total", "invalid_1x2"])
def test_incomplete_or_inconsistent_catalog_odds_keep_v8(tmp_path, reason):
    sources = _sources(tmp_path)
    markets = _markets()
    row = {"market_types": markets}
    if reason == "kickoff":
        row["source_fixture_kickoff_utc"] = (KICKOFF + timedelta(seconds=15)).isoformat()
    elif reason == "league":
        row["source_league_id"] = 140
    elif reason == "future":
        row["captured_at_utc"] = (NOW + timedelta(minutes=1)).isoformat()
    elif reason == "stale":
        markets["1x2"]["last_source_update_utc"] = (NOW - timedelta(hours=13)).isoformat()
    elif reason == "quote_after_capture":
        markets["1x2"]["last_source_update_utc"] = (CAPTURED + timedelta(minutes=1)).isoformat()
    elif reason == "missing_total":
        markets["over_3_5"] = markets.pop("over_2_5")
    elif reason == "invalid_1x2":
        markets["1x2"]["probabilities"]["away"] = .17

    path = _catalog(sources["catalog_root"], **row)
    if reason == "tampered":
        tampered = json.loads(path.read_text())
        tampered["market_types"]["over_2_5"]["probabilities"] = {"yes": .70, "no": .30}
        _write(path, tampered)
    baseline = _forecast()
    assert adjust_forecast(FIXTURE, baseline, now=NOW, **sources) is baseline


def test_v21_uses_actual_frozen_forecast_and_matching_source_snapshot(tmp_path):
    sources = _sources(tmp_path)
    identity = {"provider": "API-Football", "fixture_id": "9001", "home_team_id": "2",
                "away_team_id": "5", "kickoff_utc": KICKOFF.isoformat()}
    reference = {**identity, "forecast_id": "original", "input_digest": "inputs"}
    prior = {**reference, "model_key": "v8", "captured_at_utc": UPDATED.isoformat()}
    packet = {**identity, "status": "captured_pre_kickoff", "schema": 1,
              "captured_at_utc": CAPTURED.isoformat(), "multi_market": _markets()}
    packet["snapshot_sha256"] = _hash(packet)
    path = sources["source_root"] / "API-Football_9001_original.json"
    _write(path, {"schema": 1, "source_forecast": reference, "provider_snapshot": packet})

    assert archived_market(FIXTURE, now=NOW, **sources) is None
    original_path = sources["forecast_root"] / "forecasts" / "API-Football" / "9001" / "original.json"
    _write(original_path, prior)
    assert archived_market(FIXTURE, now=NOW, **sources)["receipt"]["source"] == "V21 fixture source"
    assert archived_market({**FIXTURE, "awayTeam": {"id": 6}}, now=NOW, **sources) is None
    _write(original_path, {**prior, "input_digest": "different"})
    assert archived_market(FIXTURE, now=NOW, **sources) is None
