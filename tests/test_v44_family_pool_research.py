from datetime import datetime, timedelta, timezone
import numpy as np
import pytest

from v44_family_pool_research import family_pool, walk_forward


def grid(good):
    p = np.full((5, 5), .01)
    p[1, 0] = .76 if good else .01
    p[0, 1] = .01 if good else .76
    return p


def row(i, delay=3):
    t = datetime(2025, 1, 1, tzinfo=timezone.utc) + timedelta(days=i)
    return {"decision_utc": t.isoformat(),
            "earliest_kickoff_utc": (t+timedelta(hours=1)).isoformat(),
            "finals_known_utc": (t+timedelta(hours=delay)).isoformat(),
            "fixture_ids": [f"{i}-a", f"{i}-b"],
            "verified": True, "actual": [[1, 0], [1, 0]],
            "families": {"good": {"one": [grid(True), grid(True)]},
                         "bad": {"one": [grid(False), grid(False)]}}}


def test_group_duplicate_variants_do_not_add_family_weight():
    r = row(0)
    before = family_pool(r["families"])["good"]
    r["families"]["good"]["duplicate"] = r["families"]["good"]["one"]
    assert np.allclose(before, family_pool(r["families"])["good"])


def test_weights_use_only_known_finals_and_improve_synthetic_log_loss():
    report = walk_forward([row(i) for i in range(35)])
    assert report["rows"][29]["weights"]["good"] == .5
    assert report["rows"][30]["weights"]["good"] > .5
    assert report["summary"]["learned"]["pairs"] == 5
    assert report["summary"]["learned"]["mean_log_loss"] < report["summary"]["equal"]["mean_log_loss"]
    assert not report["automatic_promotion"]


def test_unsettled_at_decision_excluded_from_training():
    report = walk_forward([row(i, delay=1000) for i in range(35)])
    assert report["rows"][-1]["training_pairs"] == 0
    assert report["summary"]["learned"]["hit_rate"] is None


def test_reject_duplicate_fixture_and_unverified_result():
    a, b = row(0), row(1)
    b["fixture_ids"][0] = a["fixture_ids"][0]
    with pytest.raises(ValueError, match="repeated fixture"):
        walk_forward([a, b])
    a["verified"] = False
    with pytest.raises(ValueError, match="unverified"):
        walk_forward([a])


def test_reject_invalid_distribution():
    r = row(0)
    r["families"]["bad"]["one"][0] = np.ones((5, 5))
    with pytest.raises(ValueError, match="invalid score distribution"):
        walk_forward([r])
