from copy import deepcopy
from datetime import datetime, timedelta, timezone
import math

import numpy as np
import pytest

from ai_five_double_ui import archive_rows
from forecast_ledger import append_file_record
from v41_unified_ensemble import final_pair
from v42_research_engine import hypotheses
import v43_prediction as v43


NOW = datetime(2035, 1, 1, tzinfo=timezone.utc)


def _grid():
    return np.outer([.30, .35, .20, .10, .04, .01], [.40, .30, .17, .08, .04, .01])


def _original(fixture_id, matrix):
    return {
        "provider": "API-Football", "fixture_id": str(fixture_id),
        "home_team_id": str(fixture_id * 2), "away_team_id": str(fixture_id * 2 + 1),
        "home_team": f"Home {fixture_id}", "away_team": f"Away {fixture_id}",
        "model_key": "v8", "model_version": "original-v8", "input_digest": f"digest-{fixture_id}",
        "score_basis": "regulation", "competition": "E0", "score_matrix": matrix.tolist(),
        "top10_scores": v43.top_scores(matrix), "history_matches": 40,
        "captured_at_utc": (NOW - timedelta(hours=1)).isoformat(),
        "evidence_as_of_utc": (NOW - timedelta(hours=2)).isoformat(),
        "kickoff_utc": (NOW + timedelta(hours=fixture_id)).isoformat(),
    }


def test_v43_combines_fixed_families_and_generates_five_joint_probabilities():
    left, right = _grid(), _grid().T
    original_left = left.copy()
    result = v43.combine_grids([left, right], [{}, {}])
    assert result["status"] == "issued"
    controls, _ = final_pair([left, right], [{}, {}])
    for raw, control in zip(result["full_score_grids"], controls):
        expected = .90 * control + .025 * sum(hypotheses(control).values())
        assert np.allclose(raw, expected, rtol=0, atol=1e-12)
        assert np.asarray(raw).sum() == pytest.approx(1)
    assert np.array_equal(left, original_left)
    five = result["five_doubles"]
    assert len({(row["score1"], row["score2"]) for row in five}) == 5
    assert five[0]["joint_probability"] == pytest.approx(five[0]["first_probability"] * five[0]["second_probability"])
    assert result["estimated_coverage"] == pytest.approx(math.fsum(row["joint_probability"] for row in five))
    assert result["model_promoted"] is False


def test_v43_uses_optional_xg_only_when_both_legs_qualify():
    grids = [_grid(), _grid().T]
    xg = {"xg": {"value": {"ready": True, "expected_goal_indicators": [3, 1]},
                 "reference": {"snapshot_sha256": "xg-source"}}}
    ordinary = v43.combine_grids(grids, [{}, {}])
    one_sided = v43.combine_grids(grids, [xg, {}])
    both = v43.combine_grids(grids, [xg, xg])
    assert ordinary["full_score_grids"] == one_sided["full_score_grids"]
    assert one_sided["lineage"]["evidence_used"]["xg"] is False
    assert both["lineage"]["evidence_used"]["xg"] is True
    assert both["full_score_grids"] != ordinary["full_score_grids"]
    assert v43.combine_grids([grids[0] * .9, grids[1]], [{}, {}])["status"] == "abstained"


def test_pair_reads_original_grid_across_equivalent_utc_formats(tmp_path):
    for fixture_id in (1, 2):
        append_file_record(_original(fixture_id, _grid()), root=tmp_path)
    first, second = archive_rows(now=NOW, root=tmp_path)
    assert first["fixture"]["utcDate"].endswith("+00:00")
    assert first["source_forecast"]["kickoff_utc"].endswith(".000000Z")
    kwargs = {"as_of": NOW, "source_root": tmp_path, "provider_root": tmp_path / "evidence"}
    original = v43.forecast_pair(first, second, **kwargs)
    assert original["status"] == "issued"
    assert original["fixture_models"][0]["source_forecast_id"] == first["source_forecast"]["forecast_id"]
    edited = deepcopy(first)
    edited["result"]["matrix"] = _grid().T.tolist()
    assert v43.forecast_pair(edited, second, **kwargs) == original
    edited["source_forecast"]["input_digest"] = "wrong-digest"
    assert v43.forecast_pair(edited, second, **kwargs)["status"] == "abstained"
    assert v43.forecast_pair(first, first, **kwargs)["status"] == "abstained"
    assert v43.forecast_pair(first, second, **{**kwargs, "as_of": NOW - timedelta(hours=2)})["status"] == "abstained"

    single = v43.forecast_single(first, **kwargs)
    assert single["status"] == "issued"
    assert len(single["full_score_grids"]) == 1
    assert not single["five_doubles"]
    assert sum(single["fixture_models"][0]["model_probabilities"].values()) == pytest.approx(1)


def test_live_forecast_uses_league_history_with_shared_cutoff(monkeypatch, tmp_path):
    fixture = {
        "provider": "API-Football", "id": 100, "status": "SCHEDULED",
        "utcDate": (NOW + timedelta(hours=1)).isoformat(),
        "competition": {"id": 39, "code": "PL", "season": 2034},
        "homeTeam": {"id": 1, "name": "Home"}, "awayTeam": {"id": 2, "name": "Away"},
    }
    past = {
        **fixture, "id": 101, "status": "FINISHED", "score": {"fullTime": {"home": 1, "away": 0}},
        "utcDate": (NOW - timedelta(days=2)).isoformat(),
        "awayTeam": {"id": 3, "name": "Third"},
    }
    unavailable = {**past, "id": 102, "utcDate": (NOW - timedelta(hours=1)).isoformat()}

    def model(target, history, *, league_history, as_of):
        assert target["id"] == fixture["id"]
        assert as_of == NOW
        assert [row["id"] for row in history] == [past["id"]]
        assert league_history == history
        return {"matrix": _grid(), "model_version": "v8-test",
                "component_weights": {"test-baseline": .55, "test-fit": .45}}

    monkeypatch.setattr(v43, "predict_v8", model)
    row = {"fixture": fixture, "league": "Premier League", "league_history": [past, past, unavailable]}
    result = v43.forecast_single(row, as_of=NOW, source_root=tmp_path, provider_root=tmp_path)
    assert result["status"] == "issued"
    model = result["fixture_models"][0]
    assert model["league_history_matches"] == model["home_history_matches"] == 1
    assert model["away_history_matches"] == 0
    assert model["source_forecast_id"] is None
