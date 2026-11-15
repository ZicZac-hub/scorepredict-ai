"""A later single forecast must never leak into an earlier two-match pick."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone

import numpy as np

import v21_historical_benchmark as benchmark


DAY = datetime(2035, 1, 1, tzinfo=timezone.utc)
GRID = np.pad(np.array([[.75, .10, .02], [.05, .03, .02], [.01, .01, .01]]),
              ((0, 2), (0, 2)))


def match(fid, kickoff, goals):
    return {"id": fid, "provider": "API-Football", "utcDate": kickoff.isoformat(),
            "status": "FINISHED", "provider_status": "FT",
            "homeTeam": {"id": fid*2, "name": f"Home {fid}"},
            "awayTeam": {"id": fid*2+1, "name": f"Away {fid}"},
            "competition": {"code": "TEST"},
            "score": {"fullTime": {"home": goals[0], "away": goals[1]}}}


def stub_models(monkeypatch, *, later_requires_morning_result=False):
    calls = []
    def evidence(fixture, history, as_of):
        # No forecast may see its own result or the actual scheduled status.
        assert fixture["status"] == "SCHEDULED"
        assert fixture["score"]["fullTime"] == {"home": None, "away": None}
        return {"fixture": fixture, "history": history, "league_history": history}
    def predict(fixture, history, *, league_history, as_of):
        ids = [row["id"] for row in history]
        calls.append((fixture["id"], as_of, ids))
        if fixture["id"] == 3 and later_requires_morning_result and 2 not in ids:
            return None
        # A deliberately informative morning final affects the evening model.
        # This is legitimate for the individual evening prediction, but not
        # for a pair chosen before the morning game has started.
        grid = GRID.copy()
        if fixture["id"] == 3 and 2 in ids:
            grid[:3, :3] = np.flip(GRID[:3, :3])
        return {"matrix": grid, "component_weights": {"one": .5, "two": .5}}
    monkeypatch.setattr(benchmark, "build_evidence", evidence)
    monkeypatch.setattr(benchmark, "predict_v8", predict)
    monkeypatch.setattr(benchmark, "variants", lambda grid: {
        key: grid for key in benchmark.NAMES if key != "v20_goal_form"})
    monkeypatch.setattr(benchmark, "goal_form_challenger", lambda *args: {"available": False})
    return calls


def test_double_rebuilt_before_earliest_kickoff_preserves_later_single_metric(monkeypatch):
    calls = stub_models(monkeypatch)
    source = [match(1, DAY-timedelta(days=1), (1, 0)),
              match(2, DAY+timedelta(hours=2), (0, 0)),
              match(3, DAY+timedelta(hours=8), (2, 2))]
    original = deepcopy(source)
    report = benchmark.replay(source, year=2035, max_eval=2, min_coverage=0)
    # Individual evaluation retains the original one-hour-before-each policy.
    assert report["single_match_metrics"]["baseline_v8"]["exact_hits"] == 2
    late_calls = [row for row in calls if row[0] == 3]
    assert len(late_calls) == 2
    assert late_calls[0][1] == DAY+timedelta(hours=7)
    assert 2 in late_calls[0][2]
    # Pair models are rebuilt at 01:00, before BOTH selected games kick off.
    assert late_calls[1][1] == DAY+timedelta(hours=1)
    assert 2 not in late_calls[1][2]
    pair = report["same_pair_exact_five"]
    assert pair["methods"]["baseline_v8"]["wins"] == 0
    assert pair["eligible_pair_days"] == 1
    assert pair["pair_models_rebuilt_at_common_cutoff"] is True
    assert pair["days"][0]["selected_forecasts_as_of_utc"] == [
        (DAY+timedelta(hours=1)).isoformat()]*2
    assert pair["days"][0]["all_candidate_forecasts_share_cutoff"] is True
    assert report["schema_version"] == 2
    assert source == original


def test_pair_eligibility_cannot_inherit_later_single_available_history(monkeypatch):
    stub_models(monkeypatch, later_requires_morning_result=True)
    source = [match(1, DAY-timedelta(days=1), (1, 0)),
              match(2, DAY+timedelta(hours=2), (0, 0)),
              match(3, DAY+timedelta(hours=8), (2, 2))]
    report = benchmark.replay(source, year=2035, max_eval=2, min_coverage=0)
    assert report["matched_fixtures"] == 2
    pair = report["same_pair_exact_five"]
    assert pair["eligible_pair_days"] == 0
    assert pair["days_abstained"] == 1
    assert pair["excluded_counts"]["original_research_history_insufficient"] == 1
    assert pair["days"][0]["source_qualified_forecast_count"] == 1
    assert pair["methods"]["baseline_v8"]["rate_per_issued"] is None


def test_three_hour_result_delay_applies_at_shared_pair_cutoff(monkeypatch):
    calls = stub_models(monkeypatch)
    # A previous-day match has kicked off, but has not reached the three-hour
    # result availability delay at this day's 01:00 slate cutoff.
    source = [match(1, DAY-timedelta(hours=1), (1, 0)),
              match(2, DAY+timedelta(hours=2), (0, 0)),
              match(3, DAY+timedelta(hours=8), (2, 2))]
    report = benchmark.replay(source, year=2035, max_eval=2, min_coverage=0)
    pair_calls = [row for row in calls if row[1] == DAY+timedelta(hours=1)]
    assert pair_calls and all(1 not in ids for _, _, ids in pair_calls)
    assert report["same_pair_exact_five"]["prior_result_settlement_delay_hours"] == 3
