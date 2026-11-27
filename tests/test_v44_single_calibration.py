from datetime import datetime, timedelta, timezone
import numpy as np
import pytest

import v44_single_calibration as single
from v44_family_pool_research import family_pool


def test_different_leg_support_preserves_probability_mass():
    a = np.full((5, 5), 1/25)
    b = np.full((7, 6), 1/42)
    out = family_pool({"anchor": {"v8": [a,b]}, "other":{"v": [a,b]}})
    assert out["anchor"].shape == (2,7,6)
    assert np.allclose(out["anchor"].sum(axis=(1,2)), [1,1])
    assert np.count_nonzero(out["anchor"][0,5:,:]) == 0


def test_inconsistent_same_leg_support_rejected():
    a = np.full((5,5), 1/25)
    b = np.full((6,6), 1/36)
    with pytest.raises(ValueError, match="invalid score distribution"):
        family_pool({"anchor":{"v8":[a,b]}, "other":{"v":[b,b]}})


def _record(i, delay=3):
    t = datetime(2025,1,1,tzinfo=timezone.utc) + timedelta(days=i)
    grid = np.full((11,11), 1/121)
    return {"decision_utc": t.isoformat(), "kickoff_utc": (t+timedelta(hours=1)).isoformat(),
            "final_known_utc": (t+timedelta(hours=delay)).isoformat(),
            "event": [str(i)], "forecast_id": str(i), "grid": grid, "actual":[1,0]}


def test_single_training_excludes_future_finals(monkeypatch):
    monkeypatch.setattr(single, "_combine", lambda grids, bundles:
                        {"status":"issued", "full_score_grids":grids})
    monkeypatch.setattr(single, "MIN_TRAIN_FIXTURES", 2)
    report = single.walk_forward([_record(0),_record(1,delay=100),_record(2),_record(3)])
    assert [r["training_fixtures"] for r in report["rows"]] == [0,1,1,2]
    assert report["eligible_comparison_fixtures"] == 1
    assert all(m["fixtures"] == 1 for m in report["comparisons"].values())


def test_outside_grid_is_penalized_not_dropped():
    metrics = single._metrics(np.full((5,5),1/25), [8,0])
    assert metrics["exact_log_loss"] > 30
    assert not metrics["top_score_hit"]
