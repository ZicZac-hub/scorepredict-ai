"""V26 accuracy candidate: chronological as-of gates and immutable same-pair freeze."""
from datetime import datetime, timedelta, timezone
from pathlib import Path
import json

import numpy as np
import pytest

from professional_analyst_shadow_v25 import propose
from v26_temporal_score import (
    VERSION, TRAIN_MIN, calibrate, finished_samples, _compatible_training,
    original_rows, replay, freeze, evaluate,
)

START = datetime(2035, 1, 1, 12, tzinfo=timezone.utc)
SELECTED = datetime(2035, 2, 12, 10, tzinfo=timezone.utc)
MATRIX = np.outer([.22, .32, .24, .14, .08],
                  [.30, .38, .20, .09, .03]).tolist()


def forecast(fid, capture, kickoff):
    return {
        "provider": "API-Football", "fixture_id": str(fid),
        "forecast_id": f"original-{fid}", "home_team_id": str(fid*2),
        "away_team_id": str(fid*2+1),
        "home_team": f"Home {fid}", "away_team": f"Away {fid}",
        "model_key": "v8", "model_version": "archived-v8",
        "input_digest": f"verified-historical-evidence-{fid}",
        "mode": "prospective", "score_basis": "regulation",
        "history_matches": 55, "captured_at_utc": capture.isoformat(),
        "evidence_as_of_utc": (capture - timedelta(minutes=1)).isoformat(),
        "kickoff_utc": kickoff.isoformat(), "score_matrix": MATRIX,
    }


def final(model, score, verified_at):
    return {
        "provider": model["provider"],
        "fixture_id": model["fixture_id"],
        "home_team_id": model["home_team_id"],
        "away_team_id": model["away_team_id"],
        "kickoff_utc": model["kickoff_utc"],
        "status": "FT", "evidence_version": 2, "score_basis": "regulation",
        "verified_at_utc": verified_at.isoformat(),
        "home_goals": score[0], "away_goals": score[1],
    }


def put_source(root, model):
    folder = root / "forecasts" / model["provider"] / model["fixture_id"]
    folder.mkdir(parents=True, exist_ok=True)
    (folder / (model["forecast_id"] + ".json")).write_text(json.dumps(model))


def archive(tmp_path):
    root, prematch, sets, out = (tmp_path / p for p in (
        "original", "prematch", "v25sets", "v26"))
    root.mkdir()
    prematch.mkdir()
    verified = {}
    for i in range(31):
        kickoff = START + timedelta(days=i)
        model = forecast(1000+i, kickoff-timedelta(hours=4), kickoff)
        put_source(root, model)
        score = (4, 1) if i % 3 == 0 else (1, 1)
        verified[str(1000+i)] = final(model, score, kickoff+timedelta(hours=3))
    future = [
        forecast(2100, SELECTED-timedelta(days=1), SELECTED+timedelta(hours=3)),
        forecast(2200, SELECTED-timedelta(days=1), SELECTED+timedelta(hours=4)),
    ]
    for model in future:
        put_source(root, model)
    (prematch/"verified_results.json").write_text(json.dumps(verified))
    frozen = propose(future, SELECTED, min_envelope=0)
    packets = list(frozen["days"].values())
    assert len(packets) == 1 and packets[0]["decision"] == "FROZEN_RESEARCH_ONLY"
    day = packets[0]["sydney_date"]
    sets.mkdir()
    v25_file = sets / (day+".json")
    v25_file.write_text(json.dumps(packets[0]))
    return root, prematch, sets, out, v25_file, future, verified, day


def test_predefined_shrinkage_is_modest_and_normalized():
    source = np.asarray(MATRIX)
    older = [{"grid": source, "score": (4, 1)} for _ in range(TRAIN_MIN)]
    before = float(source[np.indices(source.shape).sum(axis=0) >= 4].sum())
    adjusted, diagnostic = calibrate(MATRIX, older)
    after = float(adjusted[np.indices(adjusted.shape).sum(axis=0) >= 4].sum())
    assert after > before
    assert adjusted.sum() == pytest.approx(1)
    assert 0 < diagnostic["total_variation_from_original_v8"] < 0.1
    assert np.array_equal(source, np.asarray(MATRIX))
    assert diagnostic["unverified_accuracy_improvement"] is True
    assert calibrate(MATRIX, older[:TRAIN_MIN-1]) is None


def test_late_verified_source_cannot_train_earlier_prediction(tmp_path):
    root, prematch, _, _, _, _, verified, _ = archive(tmp_path)
    sample, _ = finished_samples(root, prematch, now=SELECTED)
    assert len(sample) == 31
    future_verified = dict(verified["1030"])
    future_verified["verified_at_utc"] = (SELECTED+timedelta(days=10)).isoformat()
    stored = json.loads((prematch/"verified_results.json").read_text())
    stored["1030"] = future_verified
    (prematch/"verified_results.json").write_text(json.dumps(stored))
    early, _ = finished_samples(root, prematch, now=SELECTED)
    assert len(early) == 30
    # No future verification may influence the same source cutoff.
    training = _compatible_training(early, cutoff=SELECTED)
    assert len(training) == 30


def test_chronological_replay_never_reuses_target_or_future_results(tmp_path):
    root, prematch, _, _, _, _, _, _ = archive(tmp_path)
    report = replay(root, prematch, now=SELECTED)
    assert report["finished_original_v8_fixtures"] == 31
    assert report["verified_holdout_fixtures"] == 1
    assert report["holdout_source_receipts"][0]["fixture_id"] == "1030"
    assert report["holdout_source_receipts"][0]["training_count_before_source"] == 30
    assert report["baseline_v8"]["verified_leg_slots"] == 1
    assert report["v26_tail_candidate"]["verified_leg_slots"] == 1
    assert report["prospective_v26_accuracy_proven"] is False


def test_new_v25_pair_creates_write_once_v26_and_graded_verified_future(tmp_path):
    root, prematch, sets, out, v25_path, future, verified, day = archive(tmp_path)
    original = v25_path.read_bytes()
    clock = SELECTED + timedelta(minutes=1)
    first = freeze(root=root, prematch=prematch, sets=sets, out=out, now=clock)
    assert first["new_frozen_days"] == [day]
    frozen = out/"sets"/(day+".json")
    preserved = frozen.read_bytes()
    packet = json.loads(preserved)
    assert packet["version"] == VERSION
    assert len(packet["verified_training_receipts"]) == 31
    assert packet["70_percent_proven"] is False
    assert packet["original_v25_pair_untouched"] is True

    reattempt = freeze(root=root, prematch=prematch, sets=sets, out=out, now=clock)
    assert reattempt["new_frozen_days"] == []
    assert frozen.read_bytes() == preserved
    pending = evaluate(root=root, prematch=prematch, sets=sets, out=out, now=clock)
    assert pending["verified_pairs"] == 0 and pending["pending_pairs"] == 1

    a, b = packet["calibrated_five"][0]["score1"], packet["calibrated_five"][0]["score2"]
    scores = [tuple(map(int, entry.split("-"))) for entry in (a, b)]
    for model, score in zip(future, scores):
        verified[model["fixture_id"]] = final(
            model, score, datetime.fromisoformat(model["kickoff_utc"])+timedelta(hours=3))
    (prematch/"verified_results.json").write_text(json.dumps(verified))
    after = SELECTED+timedelta(hours=10)
    report = evaluate(root=root, prematch=prematch, sets=sets, out=out, now=after)
    assert report["verified_pairs"] == 1
    assert report["same_pair_five_hits"]["v26_calibrated"] == 1
    assert report["v26_minus_v8_log_loss"] is not None
    assert frozen.read_bytes() == preserved
    assert v25_path.read_bytes() == original

    broken = json.loads(preserved)
    broken["first"]["score_matrix"][0][0] += .001
    frozen.write_text(json.dumps(broken))
    invalid = evaluate(root=root, prematch=prematch, sets=sets, out=out, now=after)
    assert invalid["verified_pairs"] == 0
    assert invalid["invalid_or_tampered_sets_excluded"] == {
        "v26_frozen_matrix_not_reproducible": 1
    }


def test_cannot_retrofreeze_older_v25_selection(tmp_path):
    root, prematch, sets, out, _, _, _, _ = archive(tmp_path)
    result = freeze(root=root, prematch=prematch, sets=sets, out=out,
                    now=SELECTED+timedelta(minutes=21))
    assert result["new_frozen_days"] == []
    assert result["abstentions"]["v25_set_not_new_in_same_freeze_window"] == 1


def test_missing_provenance_or_early_verified_results_never_qualify(tmp_path):
    root, prematch, sets, out, _, _, verified, _ = archive(tmp_path)
    stored = json.loads((prematch/"verified_results.json").read_text())
    # The 31st result was available after the real decision, so it is not training.
    stored["1030"]["verified_at_utc"] = (SELECTED+timedelta(days=3)).isoformat()
    (prematch/"verified_results.json").write_text(json.dumps(stored))
    result = freeze(root=root, prematch=prematch, sets=sets, out=out,
                    now=SELECTED+timedelta(minutes=1))
    assert len(result["new_frozen_days"]) == 1
    frozen = next((out/"sets").glob("*.json"))
    assert len(json.loads(frozen.read_text())["verified_training_receipts"]) == 30
