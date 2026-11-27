"""Use the real file ledger, original pair and verification code for V43."""
from datetime import datetime, timedelta, timezone
import json

import numpy as np

from forecast_ledger import append_file_record, read_file_records
from professional_double_shadow import freeze as freeze_original
from v43_prediction import top_scores
import v43_prospective as study


NOW = datetime(2035, 1, 1, tzinfo=timezone.utc)
DAY = "2035-01-01"


def setup_original_pair(directory):
    original, pair = directory / "original", directory / "pair"
    grid = np.outer([.4, .30, .20, .075, .025], [.4, .30, .20, .075, .025])
    for ident in (1, 2):
        append_file_record({
            "provider": "API-Football", "fixture_id": str(ident),
            "model_key": "v8", "model_version": "test-v8", "input_digest": f"digest-{ident}",
            "home_team_id": str(ident * 2), "away_team_id": str(ident * 2 + 1),
            "home_team": f"Home {ident}", "away_team": f"Away {ident}",
            "competition": "PL", "score_basis": "regulation",
            "captured_at_utc": (NOW - timedelta(hours=1)).isoformat(),
            "evidence_as_of_utc": (NOW - timedelta(hours=2)).isoformat(),
            "kickoff_utc": (NOW + timedelta(hours=ident + 1)).isoformat(),
            "score_matrix": grid.tolist(), "top10_scores": top_scores(grid),
        }, root=original)
    assert freeze_original(root=pair, source_root=original, now=NOW, clock=lambda: NOW)["new_sets"] == [DAY]
    return {"root": directory / "study", "source_root": original,
            "pair_root": pair, "provider_root": directory / "provider"}


def write_finals(source_root, receipt, count):
    raw = {}
    for idx, model in enumerate(sorted(read_file_records(source_root), key=lambda record: record["fixture_id"])[:count], 1):
        choice = receipt["arms"]["v43"]["five_doubles"][0]
        home, away = map(int, choice[f"score{idx}"].split("-"))
        raw[model["fixture_id"]] = {
            **{key: model[key] for key in ("provider", "fixture_id", "home_team_id", "away_team_id", "kickoff_utc")},
            "score_basis": "regulation", "status": "FT", "home_goals": home, "away_goals": away,
            "evidence_version": 2, "verified_at_utc": (NOW + timedelta(hours=6)).isoformat(),
        }
    (source_root / "verified_results.json").write_text(json.dumps(raw), encoding="utf-8")


def test_v43_freeze_once_and_settle_matched_pair(tmp_path):
    inputs = setup_original_pair(tmp_path)
    receipt = inputs["root"] / "trials" / f"{DAY}.json"
    frozen = study.freeze(**inputs, now=NOW, clock=lambda: NOW)
    assert frozen["new_trials"] == [DAY]
    original = receipt.read_bytes()
    assert study.freeze(**inputs, now=NOW, clock=lambda: NOW)["preserved_trials"] == [DAY]
    assert receipt.read_bytes() == original

    report = study.evaluate(**inputs, now=NOW)
    assert report["pending_kickoff_pairs"] == 1
    assert report["hit_rate"] is None

    as_of = NOW + timedelta(hours=7)
    write_finals(inputs["source_root"], json.loads(original), count=1)
    report = study.evaluate(**inputs, now=as_of, prematch_root=tmp_path / "prematch")
    assert report["pending_verification_pairs"] == 1
    assert report["rows"][0]["actual_scores"][0] is not None
    assert report["rows"][0]["actual_scores"][1] is None
    assert report["verified_pairs"] == report["hits"] == 0

    write_finals(inputs["source_root"], json.loads(original), count=2)
    report = study.evaluate(**inputs, now=as_of, prematch_root=tmp_path / "prematch")
    assert report["verified_pairs"] == report["hits"] == report["methods"]["v8_original"]["verified_pairs"] == 1
    assert report["pending_pairs"] == report["invalid_pairs"] == 0
    assert report["rows"][0]["v43_hit"] is True and report["review_gate_met"] is False
    assert receipt.read_bytes() == original


def test_post_window_run_leaves_no_receipt_and_reports_miss(tmp_path):
    inputs = setup_original_pair(tmp_path)
    late = NOW + timedelta(hours=4)
    rejected = study.freeze(**inputs, now=NOW, clock=lambda: late)
    assert rejected["new_trials"] == []
    assert rejected["rejections"] == {"calculation_finished_outside_decision_window": 1}
    assert study.freeze(**inputs, now=late, clock=lambda: late)["new_trials"] == []
    report = study.evaluate(**inputs, now=late, prematch_root=tmp_path / "prematch")
    assert report["missed_windows"] == 1
    assert report["verified_pairs"] == report["pending_pairs"] == report["abstained_pairs"] == 0


def test_receipt_reproduction_rejects_resigned_probability_tampering(tmp_path):
    inputs = setup_original_pair(tmp_path)
    study.freeze(**inputs, now=NOW, clock=lambda: NOW)
    path = inputs["root"] / "trials" / f"{DAY}.json"
    record = json.loads(path.read_text(encoding="utf-8"))
    record["arms"]["v43"]["five_doubles"][0]["joint_probability"] += .01
    record["record_sha256"] = study._digest(record, "record_sha256")
    path.write_text(json.dumps(record), encoding="utf-8")
    report = study.evaluate(**inputs, now=NOW)
    assert report["verified_pairs"] == 0
    assert report["invalid_pairs"] == 1
    assert report["rejections"] == {"v43_receipt_does_not_reproduce": 1}


def test_repeated_physical_fixtures_do_not_add_pending_or_verified_pairs(tmp_path, monkeypatch):
    inputs = setup_original_pair(tmp_path)
    study.freeze(**inputs, now=NOW, clock=lambda: NOW)
    receipt_dir = inputs["root"] / "trials"
    first = receipt_dir / f"{DAY}.json"
    second = receipt_dir / "2035-01-02.json"
    second.write_bytes(first.read_bytes())
    original_validate = study._validated
    monkeypatch.setattr(study, "_validated", lambda path, **kwargs: original_validate(first, **kwargs))
    write_finals(inputs["source_root"], json.loads(first.read_bytes()), count=2)
    report = study.evaluate(**inputs, now=NOW + timedelta(hours=7), prematch_root=tmp_path / "prematch")
    assert report["verified_pairs"] == 1 and report["invalid_pairs"] == 1
    assert report["rejections"] == {"repeated_fixture_or_cross_provider_event": 1}


def test_read_only_missing_archive_creates_no_registration(tmp_path):
    report = study.evaluate(root=tmp_path)
    assert report["status"] == "not_registered"
    assert report["registered_at_utc"] is None
    assert report["wilson_95"] is None
    assert not list(tmp_path.iterdir())


def test_fixed_review_uses_first_100_once_and_later_hits_do_not_replace_them():
    rows = [{"sydney_date": (NOW + timedelta(days=i)).date().isoformat(),
             "predicted_at_utc": (NOW + timedelta(days=i)).isoformat(),
             "status": "verified", "v43_hit": i < 78 or i >= 100}
            for i in range(115)]
    review = study._fixed_review(rows[::-1])
    assert review["issued"] == review["verified"] == 100
    assert review["status"] == "below_target" and review["hits"] == 78
    rows[78]["v43_hit"] = True
    assert study._fixed_review(rows)["status"] == "met"
    rows[78].update(status="pending_verification", v43_hit=None)
    review = study._fixed_review(rows)
    assert review["status"] == "waiting_for_verification" and review["verified"] == 99
    assert review["wilson_95"] is None
