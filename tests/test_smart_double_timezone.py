"""Regression tests for Sydney-local five-double fixture pairing."""
from smart_double_ai import _sydney_day, rank_pairs


def _row(id_, kickoff):
    return {
        "fixture": {"provider": "test", "id": id_,
                    "homeTeam": {"name": f"Home{id_}"},
                    "awayTeam": {"name": f"Away{id_}"}},
        "kickoff": kickoff,
        "result": {"scores": [(1, 0, .25), (1, 1, .2), (0, 1, .15)]},
    }


def test_sydney_pairing_across_utc_midnight():
    # Both fixtures occur on 15 Nov in Sydney despite different UTC dates.
    rows = [_row(1, "2026-11-14T12:00:00Z"),
            _row(2, "2026-11-15T02:00:00Z")]
    ranked = rank_pairs(rows)
    assert len(ranked) == 1
    assert len(ranked[0]["doubles"]) == 5


def test_do_not_pair_different_sydney_days_even_if_same_utc_day():
    rows = [_row(1, "2026-11-14T01:00:00Z"),
            _row(2, "2026-11-14T14:00:00Z")]
    assert rank_pairs(rows) == []


def test_sydney_day_handles_daylight_saving():
    assert str(_sydney_day("2026-07-01T14:30:00Z")) == "2026-07-02"
    assert str(_sydney_day("2026-11-14T13:30:00Z")) == "2026-11-15"
