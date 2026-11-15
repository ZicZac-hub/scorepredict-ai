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
    rows = [_row(1, "2026-11-14T13:00:00Z"),
            _row(2, "2026-11-15T02:00:00Z")]
    assert str(_sydney_day(rows[0]["kickoff"])) == "2026-11-15"
    assert _sydney_day(rows[0]["kickoff"]) == _sydney_day(rows[1]["kickoff"])
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


def test_live_pairing_excludes_cached_started_and_imminent_fixtures():
    rows = [_row(1, "2026-11-15T01:00:00Z"),
            _row(2, "2026-11-15T02:00:00Z")]
    assert len(rank_pairs(rows, now="2026-11-15T00:00:00Z")) == 1
    assert rank_pairs(rows, now="2026-11-15T01:00:00Z") == []
    assert rank_pairs(rows, now="2026-11-15T00:55:00Z") == []
    assert rank_pairs(rows, now="invalid") == []
    # Explicit historical analysis can still reproduce dated rankings.
    assert len(rank_pairs(rows)) == 1


def test_live_pairing_requires_four_known_distinct_teams():
    first = _row(1, "2026-11-15T01:00:00Z")
    second = _row(2, "2026-11-15T02:00:00Z")
    second['fixture']['awayTeam']['name'] = '  HOME1  '
    assert rank_pairs([first, second]) == []
    second['fixture']['awayTeam']['name'] = ''
    assert rank_pairs([first, second]) == []
