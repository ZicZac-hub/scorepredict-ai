"""Only original pre-kickoff score grids and verified results may train calibration."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json

import numpy as np
import pytest

from forecast_ledger import append_file_record, read_file_records
from professional_double_optimizer import AUTO_SELECTION, POLICY, calibration_trials
import professional_double_shadow as shadow


NOW = datetime(2035, 1, 1, tzinfo=timezone.utc)


def sources(tmp_path):
    root = tmp_path/"original"
    marginal = np.asarray([.4, .3, .2, .075, .025])
    grid = np.outer(marginal, marginal)
    scores = sorted([[h, a, float(p)] for h, line in enumerate(grid)
                     for a, p in enumerate(line)], key=lambda r: (-r[2], r[0], r[1]))[:10]
    for fid in (1, 2, 3):
        model = {"mode": "prospective", "provider": "API-Football", "fixture_id": str(fid),
                 "model_key": "v8", "model_version": "original-test-v8", "input_digest": "digest"+str(fid),
                 "home_team_id": str(fid*2), "away_team_id": str(fid*2+1),
                 "home_team": f"Home {fid}", "away_team": f"Away {fid}",
                 "fixture_name": f"Home {fid} vs Away {fid}", "competition": "PL",
                 "score_matrix": grid.tolist(), "top10_scores": scores,
                 "captured_at_utc": (NOW-timedelta(hours=1)).isoformat(),
                 "evidence_as_of_utc": (NOW-timedelta(hours=1, minutes=5)).isoformat(),
                 "kickoff_utc": (NOW+timedelta(hours=3+fid)).isoformat()}
        assert append_file_record(model, root=root)
    return root


def freeze(tmp_path):
    original = sources(tmp_path)
    out = tmp_path/"shadow"
    status = shadow.freeze(root=out, source_root=original, now=NOW, clock=lambda: NOW)
    assert status["new_sets"] == ["2035-01-01"]
    path = out/"sets"/"2035-01-01.json"
    return original, out, path, json.loads(path.read_text())


def results(original, record, *, hit=True):
    by_id = {str(r["fixture_id"]): r for r in read_file_records(root=original)}
    output = {}
    selected = record["selection"]
    for number, side in enumerate(("first", "second"), 1):
        model = by_id[selected["source_forecast_refs"][side]["fixture_id"]]
        goals = list(map(int, selected["choices"][0][f"score{number}"].split("-"))) if hit else [9, 8]
        result = {key: model[key] for key in (
            "provider", "fixture_id", "home_team_id", "away_team_id", "kickoff_utc")}
        result.update({"status": "FT", "evidence_version": 2, "score_basis": "regulation",
                       "home_goals": goals[0], "away_goals": goals[1],
                       "verified_at_utc": (NOW+timedelta(hours=12)).isoformat()})
        output[model["fixture_id"]] = result
    (original/"verified_results.json").write_text(json.dumps(output))
    return output


def evaluate(original, out, **changes):
    return shadow.evaluate(root=out, source_root=original, prematch_root=out/"missing",
                           now=NOW+timedelta(days=1), **changes)


def test_freeze_is_write_once_source_preserving_and_unknown_results_pending(tmp_path):
    original, out, path, record = freeze(tmp_path)
    saved = path.read_bytes()
    originals = {p: p.read_bytes() for p in original.glob("forecasts/*/*/*.json")}
    retry = shadow.freeze(root=out, source_root=original, now=NOW+timedelta(minutes=1),
                          clock=lambda: NOW+timedelta(minutes=1))
    assert retry["new_sets"] == [] and retry["existing_sets_preserved"] == ["2035-01-01"]
    assert path.read_bytes() == saved
    assert all(p.read_bytes() == content for p, content in originals.items())
    pending = evaluate(original, out)
    assert pending["source_valid_sets"] == pending["pending_sets"] == 1
    assert pending["verified_sets"] == pending["hits"] == 0
    assert pending["observed_hit_rate"] is None
    assert pending["trials"] == []
    assert pending["target_70_percent_proven"] is False
    assert record["calibration_bootstrap_only_not_premium_recommendation"]


@pytest.mark.parametrize("hit", [True, False])
def test_hits_are_recomputed_from_both_full_time_results_not_serialized_flags(tmp_path, hit):
    original, out, path, record = freeze(tmp_path)
    results(original, record, hit=hit)
    record["selection"]["hit"] = not hit
    record["trials"] = [{"hit": not hit}]
    record["record_sha256"] = shadow._digest(record)
    path.write_text(json.dumps(record))
    report = evaluate(original, out)
    assert report["verified_sets"] == 1
    assert report["hits"] == int(hit)
    trial = report["trials"][0]
    assert trial["selection_scope"] == AUTO_SELECTION
    assert trial["policy_id"] == POLICY
    assert trial["hit"] is hit
    accepted, _ = calibration_trials(report["trials"], now=NOW+timedelta(days=2),
                                     model_versions=trial["model_versions"])
    assert len(accepted) == 1
    assert shadow.load_verified_trials(root=out, source_root=original,
                                       prematch_root=out/"none", now=NOW+timedelta(days=1)) == report["trials"]


@pytest.mark.parametrize("field,value", [("home_team_id", "wrong-team"), ("provider", "wrong-provider"),
    ("score_basis", "extra_time"), ("status", "NS"), ("home_goals", True),
    ("verified_at_utc", (NOW+timedelta(days=3)).isoformat())])
def test_wrong_or_future_result_evidence_remains_pending(tmp_path, field, value):
    original, out, _, record = freeze(tmp_path)
    finals = results(original, record)
    next(iter(finals.values()))[field] = value
    (original/"verified_results.json").write_text(json.dumps(finals))
    report = evaluate(original, out)
    assert report["verified_sets"] == 0
    assert report["pending_sets"] == 1


def test_conflicting_result_scores_are_excluded_from_calibration(tmp_path):
    original, out, _, record = freeze(tmp_path)
    finals = results(original, record)
    next(iter(finals.values()))["home_goals"] = 7
    other = out/"missing"
    other.mkdir()
    (other/"verified_results.json").write_text(json.dumps(finals))
    report = evaluate(original, out)
    assert report["conflicting_result_identities"] == 1
    assert report["pending_sets"] == 1
    assert report["trials"] == []


def test_probabilities_grid_and_source_digest_tampering_are_rejected(tmp_path):
    original, out, path, record = freeze(tmp_path)
    altered = deepcopy(record)
    altered["selection"]["choices"][0]["joint_probability"] += .1
    altered["record_sha256"] = shadow._digest(altered)
    path.write_text(json.dumps(altered))
    assert evaluate(original, out)["source_rejections"] == {"choice_probability_disagrees_with_original_grid": 1}
    altered = deepcopy(record)
    altered["selection"]["source_forecast_refs"]["first"]["input_digest"] = "changed"
    altered["record_sha256"] = shadow._digest(altered)
    path.write_text(json.dumps(altered))
    assert evaluate(original, out)["source_rejections"] == {"source_input_digest_mismatch": 1}
    path.write_text(json.dumps(record))
    ref = record["selection"]["source_forecast_refs"]["first"]
    source = original/"forecasts"/ref["provider"]/ref["fixture_id"]/(ref["forecast_id"]+".json")
    model = json.loads(source.read_text())
    model["score_matrix"][0][0] -= .01
    model["score_matrix"][0][1] += .01
    source.write_text(json.dumps(model))
    assert evaluate(original, out)["source_valid_sets"] == 0


def test_source_time_backfill_or_same_date_relabelling_is_not_a_trial(tmp_path):
    original, out, path, record = freeze(tmp_path)
    record["predicted_at_utc"] = (NOW-timedelta(hours=2)).isoformat()
    record["record_sha256"] = shadow._digest(record)
    path.write_text(json.dumps(record))
    assert evaluate(original, out)["source_rejections"] == {"source_created_after_pair_selection": 1}


def test_late_calculation_cannot_create_a_backdated_forecast(monkeypatch, tmp_path):
    original = sources(tmp_path)
    out = tmp_path/"shadow"
    at = [NOW]
    actual = shadow.auto_shadow
    def slow(*args, **kwargs):
        result = actual(*args, **kwargs)
        at[0] = NOW+timedelta(hours=5)
        return result
    monkeypatch.setattr(shadow, "auto_shadow", slow)
    report = shadow.freeze(root=out, source_root=original, now=NOW, clock=lambda: at[0])
    assert report["new_sets"] == []
    assert report["rejections"] == {"not_frozen_in_eligible_pre_match_window": 1}
    assert not list((out/"sets").glob("*.json"))


def test_relabelled_filename_or_policy_never_inflates_verified_trial_count(tmp_path):
    original, out, path, record = freeze(tmp_path)
    results(original, record)
    extra = deepcopy(record)
    extra["sydney_date"] = extra["selection"]["sydney_date"] = "2035-01-02"
    extra["record_sha256"] = shadow._digest(extra)
    (path.parent/"2035-01-02.json").write_text(json.dumps(extra))
    report = evaluate(original, out)
    assert report["verified_sets"] == 1
    assert report["source_rejections"] == {"same_sydney_date_required": 1}
