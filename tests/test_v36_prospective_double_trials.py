"""Future-only V36 integrity tests: no hindsight or inflated denominators."""
import json
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

import v36_prospective_double_trials as trial
from v36_tail_calibration_research import METHODS


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


def test_v36_packet_fails_late_and_missing_source(monkeypatch):
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
    dest = tmp_path/"v36"
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


def test_v36_pending_not_a_loss_and_no_promotion():
    row = {"sydney_day":"2035-01-01","status":"pending",
           "original_selected_fixture_keys":[["p","1"],["p","2"]]}
    summary = trial.summarize([row],now=datetime(2035,1,1,tzinfo=timezone.utc))
    assert summary["pending_days"] == 1
    assert summary["verified_independent_sydney_days"] == 0
    assert summary["ready_for_separate_production_review"] == []
    assert all(summary["methods"][m]["observed_five_double_hit_rate"] is None for m in METHODS)


def test_v36_out_of_grid_final_is_counted_not_dropped():
    grid = np.full((5,5),1/25).tolist()
    proper = trial._describe(grid,(7,1))
    assert proper["outside_score_grid"]
    assert proper["exact_score_log_loss"] > 30
    assert proper["multiclass_score_brier"] > 1
    assert proper["over_3_5_brier"] >= 0
