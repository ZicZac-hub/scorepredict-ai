"""Future-only V39 integrity tests: no hindsight or inflated denominators."""
import json
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

import v39_prospective_double_trials as trial
from v39_poisson_marginal import METHODS


def fixture_data():
    issued = datetime(2035, 1, 1, 10, tzinfo=timezone.utc)
    grid = np.full((6, 6), 1/36).tolist()
    a = {"key":["provider","101"], "teams":{"a","b"},
         "kickoff":issued+timedelta(hours=2), "grid":grid}
    b = {"key":["provider","202"], "teams":{"c","d"},
         "kickoff":issued+timedelta(hours=2), "grid":grid}
    original = {
        "policy":"FROZEN", "sydney_day":"2035-01-01",
        "actually_issued_at_utc":issued.isoformat(),
        "candidate_catalog_sha256":"unchanged-original-digest",
        "quality_selection":{
            "fixture_keys":[["provider","101"],["provider","202"]],
            "source_forecast_ids":["f101","f202"],
        },
    }
    return issued, original, [a,b]


def test_five_double_packet_same_fixtures_all_prespecified_methods(monkeypatch):
    issued, original, items = fixture_data()
    monkeypatch.setattr(trial, "_frozen_eligible", lambda *_: True)
    packet = trial.build_packet(original, items, issued)
    assert packet["predeclared_methods"] == list(METHODS)
    assert packet["original_catalog_sha256"] == original["candidate_catalog_sha256"]
    assert packet["research_only"] and packet["no_live_model_change"]
    assert packet["original_selected_fixture_keys"] == original["quality_selection"]["fixture_keys"]
    for arm in packet["policies"].values():
        assert len(arm["five_joint_scores"]) == 5
        assert len({(x["score1"],x["score2"]) for x in arm["five_joint_scores"]}) == 5
        assert all(np.isclose(np.asarray(grid).sum(), 1) for grid in arm["score_grids"])


def test_v39_packet_fails_late_and_missing_source(monkeypatch):
    issued, original, items = fixture_data()
    monkeypatch.setattr(trial, "_frozen_eligible", lambda *_: True)
    with pytest.raises(ValueError,match="too_late"):
        trial.build_packet(original, items, issued+timedelta(hours=1,minutes=30))
    with pytest.raises(ValueError,match="missing_immutable"):
        trial.build_packet(original, items[:1], issued)


def test_frozen_packet_is_write_once_and_tamper_detected(monkeypatch,tmp_path):
    issued, original, items = fixture_data()
    v28_root = tmp_path/"v28"
    decisions = v28_root/"decisions"
    decisions.mkdir(parents=True)
    (decisions/"2035-01-01.json").write_text("{}")
    monkeypatch.setattr(trial, "_frozen_eligible",lambda *_: True)
    monkeypatch.setattr(trial, "verify_decision",lambda *a,**k:(original,items))
    dest = tmp_path/"v39"
    first = trial.freeze(root=tmp_path,v28_root=v28_root,out=dest,now=issued)
    assert first["created_days"] == ["2035-01-01"]
    second = trial.freeze(root=tmp_path,v28_root=v28_root,out=dest,now=issued)
    assert second["created_days"] == []
    p = dest/"sets"/"2035-01-01.json"
    assert trial.verify_packet(p,root=tmp_path,v28_root=v28_root,clock=issued)
    saved = json.loads(p.read_text())
    saved["policies"]["baseline"]["five_joint_scores"][0]["score1"] = "9-9"
    p.write_text(json.dumps(saved))
    with pytest.raises(ValueError,match="tampered"):
        trial.verify_packet(p,root=tmp_path,v28_root=v28_root,clock=issued)


def test_v39_pending_not_a_loss_and_no_promotion():
    row = {"sydney_day":"2035-01-01","status":"pending",
           "original_selected_fixture_keys":[["p","1"],["p","2"]]}
    summary = trial.summarize([row],now=datetime(2035,1,1,tzinfo=timezone.utc))
    assert summary["pending_days"] == 1
    assert summary["verified_independent_sydney_days"] == 0
    assert summary["ready_for_separate_production_review"] == []
    assert all(summary["methods"][m]["observed_five_double_hit_rate"] is None for m in METHODS)


def test_v39_out_of_grid_final_is_counted_not_dropped():
    grid = np.full((5,5),1/25).tolist()
    proper = trial._describe(grid,(7,1))
    assert proper["outside_score_grid"]
    assert proper["exact_score_log_loss"] > 30
    assert proper["multiclass_score_brier"] > 1
    assert proper["over_3_5_brier"] >= 0

def test_v39_does_not_count_same_fixture_on_two_days():
    from v39_poisson_marginal import METHODS
    fixture_metrics = {
        "exact_score_log_loss":2.0, "multiclass_score_brier":0.7,
        "over_3_5_brier":0.2, "under_1_5_brier":0.1, "margin_3_plus_brier":0.2,
        "exact_score_probability":0.1, "outside_score_grid":False,
    }
    def row(day):
        return {
            "sydney_day":day, "status":"verified",
            "original_selected_fixture_keys":[["p","1"],["p","2"]],
            "actual_scores":["1-1","2-0"],
            "hits":{method:False for method in METHODS},
            "proper_scores":{method:[fixture_metrics,fixture_metrics] for method in METHODS},
        }
    with pytest.raises(ValueError,match="same_physical_fixture_repeated"):
        trial.summarize([row("2035-01-01"),row("2035-01-02")],
                        now=datetime(2035,1,3,tzinfo=timezone.utc))


def test_roundoff_only_never_reselects_frozen_tickets(monkeypatch):
    from copy import deepcopy
    from math import nextafter, inf
    issued, original, items = fixture_data()
    monkeypatch.setattr(trial, "_frozen_eligible", lambda *_: True)
    base = trial.build_packet(original, items, issued)
    near = deepcopy(base)
    arm = near["policies"]["conditional_margin_125"]
    grid = arm["score_grids"][0]
    grid[0][0] = nextafter(grid[0][0], inf)
    ticket = arm["five_joint_scores"][0]
    ticket["joint_probability"] = nextafter(ticket["joint_probability"], inf)
    assert trial._reproduces_packet(base, near)
    assert base == trial.build_packet(original, items, issued)

    for change in ("ticket", "order", "grid", "missing", "source", "nan", "type"):
        altered = deepcopy(near)
        arm = altered["policies"]["conditional_margin_125"]
        picks = arm["five_joint_scores"]
        if change == "ticket":
            picks[0]["score1"] = "99-99"
        elif change == "order":
            picks[0], picks[1] = picks[1], picks[0]
        elif change == "grid":
            arm["score_grids"][0][0][0] += 1e-6
            arm["score_grids"][0][0][1] -= 1e-6
        elif change == "missing":
            picks.pop()
        elif change == "source":
            altered["original_catalog_sha256"] = "changed"
        elif change == "nan":
            picks[0]["joint_probability"] = float("nan")
        else:
            picks[0]["joint_probability"] = str(picks[0]["joint_probability"])
        assert not trial._reproduces_packet(base, altered), change
    assert not trial._reproduces_packet(0.0, 1e-20)




def test_poisson_arm_reproduces_original_research_and_preserves_source():
    from v39_poisson_marginal import transform_score_grid
    from v21_goal_model_comparison import variants
    from v38_margin_calibration import transform_score_grid as old
    p = np.arange(1,43,dtype=float).reshape(6,7)
    p /= p.sum()
    before = p.copy()
    q = np.asarray(transform_score_grid(p,"poisson_marginal"))
    assert np.array_equal(p,before)
    assert np.allclose(q,variants(p)["poisson_marginal"],rtol=0,atol=1e-15)
    assert q.shape == p.shape
    assert np.isclose(q.sum(),1)
    assert np.all(q>=0)
    assert np.allclose(q,np.outer(q.sum(axis=1),q.sum(axis=0)),rtol=0,atol=1e-14)
    assert np.array_equal(transform_score_grid(p,"conditional_margin_125"),
                          old(p,"conditional_margin_125"))
    with pytest.raises(ValueError):
        transform_score_grid(p,"unregistered")


def test_poisson_outside_validated_rate_range_abstains():
    from v39_poisson_marginal import transform_score_grid
    p = np.zeros((6,6))
    p[0,0] = 1
    with pytest.raises(ValueError):
        transform_score_grid(p,"poisson_marginal")
