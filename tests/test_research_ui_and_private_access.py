"""Offline regression tests for new private access and Deep Research evidence."""
from datetime import datetime, timedelta, timezone

from private_access import matches
from research_ui import _games, _stats


def _fixture(start):
    return {"utcDate": start.isoformat(),
            "homeTeam": {"id": 1, "name": "Home"},
            "awayTeam": {"id": 2, "name": "Away"}}


def _game(start, h=1, a=2, goals=(3, 0), status="FINISHED"):
    return {"utcDate": start.isoformat(), "status": status,
            "homeTeam": {"id": h, "name": str(h)},
            "awayTeam": {"id": a, "name": str(a)},
            "score": {"fullTime": {"home": goals[0], "away": goals[1]}}}


def test_private_access_comparison():
    assert matches("correct-abc", "correct-abc")
    assert not matches("incorrect", "correct-abc")
    assert not matches("", "")


def test_history_is_sorted_and_excludes_future_and_unfinished():
    now = datetime.now(timezone.utc)
    fixture = _fixture(now + timedelta(days=2))
    old = _game(now - timedelta(days=5), goals=(0, 2))
    new = _game(now - timedelta(days=1), goals=(3, 1))
    future = _game(now + timedelta(days=3), goals=(7, 0))
    pending = _game(now - timedelta(days=2), status="TIMED")
    rows = _games(fixture, [old, pending, future, new], team_id=1)
    assert [r["Result"] for r in rows] == ["W", "L"]
    assert _stats(rows)["Wins"] == 1
    assert _stats(rows)["Losses"] == 1


def test_venue_and_h2h_selections_use_matching_team_ids():
    now = datetime.now(timezone.utc)
    fixture = _fixture(now + timedelta(days=2))
    matches_ = [_game(now - timedelta(days=3), goals=(2, 0)),
                _game(now - timedelta(days=2), h=3, a=1, goals=(2, 1)),
                _game(now - timedelta(days=1), h=1, a=4, goals=(4, 1))]
    home = _games(fixture, matches_, team_id=1, venue="home")
    away = _games(fixture, matches_, team_id=1, venue="away")
    h2h = _games(fixture, matches_, h2h=True)
    assert len(home) == 2
    assert len(away) == 1
    assert len(h2h) == 1
    assert h2h[0]["Final score"] == "2-0"
