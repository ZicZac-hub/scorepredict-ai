from pathlib import Path

import prospective_double_comparison as comparison
from persist_prematch import load_json, write_json


def test_legacy_strategy_freezes_only_before_both_kickoffs_and_verifies_both_legs(
        tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    first = {
        "provider": "API-Football", "fixture_id": 1, "competition": "PL",
        "home_team_id": 11, "away_team_id": 12, "home_team": "One", "away_team": "Two",
        "captured_at_utc": "2026-11-22T09:30:00Z", "kickoff_utc": "2026-11-22T12:00:00Z",
        "double_top10_scores": [[1, 0, .21], [1, 1, .14], [0, 0, .11], [0, 1, .1], [2, 0, .09]],
    }
    second = dict(first, fixture_id=2, home_team_id=21, away_team_id=22,
                  home_team="Three", away_team="Four", captured_at_utc="2026-11-22T09:40:00Z",
                  kickoff_utc="2026-11-22T14:00:00Z")
    write_json(Path("prematch_snapshots/prematch_test.json"), {"records": [first, second]})

    assert comparison.freeze(now="2026-11-22T09:35:00Z") == 0
    assert comparison.freeze(now="2026-11-22T12:00:00Z") == 0
    assert comparison.freeze(now="2026-11-22T16:00:00Z") == 0
    assert not list(comparison.PAIR_DIR.glob("*.json"))

    assert comparison.freeze(now="2026-11-22T10:00:00Z") == 1
    destination = comparison.PAIR_DIR / "PL_1_2.json"
    archived = destination.read_bytes()
    packet = load_json(destination, {})
    assert packet["captured_at_utc"] == "2026-11-22T10:00:00+00:00"
    assert packet["first_source"]["captured_at_utc"] == first["captured_at_utc"]
    assert packet["second_source"]["captured_at_utc"] == second["captured_at_utc"]
    assert comparison.freeze(now="2026-11-22T11:00:00Z") == 0
    assert destination.read_bytes() == archived

    finals = {}
    for leg in (first, second):
        final = dict(leg, status="FT", verified_at_utc="2026-11-22T16:00:00Z",
                     evidence_version=2, score_basis="regulation", home_goals=1, away_goals=0)
        finals[str(leg["fixture_id"])] = final
    finals["2"]["away_team_id"] = 999
    write_json(comparison.RESULTS, finals)
    result = comparison.evaluate(now="2026-11-22T17:00:00Z")
    assert result["pairs_verified"] == 0
    assert result["exclusions"]["awaiting_both_verified_results"] == 1

    finals["2"]["away_team_id"] = second["away_team_id"]
    write_json(comparison.RESULTS, finals)
    result = comparison.evaluate(now="2026-11-22T17:00:00Z")
    assert result["pairs_verified"] == 1
    assert result["methods"]["original"]["wins"] == 1
