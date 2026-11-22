"""Regression tests: same physical fixtures, sealed score grids, verified finals."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json

import pytest

from five_double_accuracy import joint_geometry
from v8_v25_matched_audit import _selection_mass, build_report

KICKOFF = datetime(2035, 1, 1, 14, tzinfo=timezone.utc)
SELECTED = KICKOFF - timedelta(hours=2)
CAPTURE = SELECTED - timedelta(hours=4)
NOW = KICKOFF + timedelta(hours=5)
DAY = "2035-01-02"


def grid():
    return [
        [.4, .18, 0, 0, 0],
        [.22, .1, 0, 0, 0],
        [.06, 0, .04, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
    ]


def forecast(fid, side):
    return {
        "provider": "API-Football", "fixture_id": fid, "forecast_id": side,
        "home_team_id": str(10 if side == "a" else 12),
        "away_team_id": str(11 if side == "a" else 13),
        "model_key": "v8", "model_version": "saved-v8",
        "input_digest": "real-frozen-input-digest",
        "mode": "prospective", "score_basis": "regulation",
        "history_matches": 55,
        "captured_at_utc": CAPTURE.isoformat(),
        "evidence_as_of_utc": (CAPTURE - timedelta(minutes=1)).isoformat(),
        "kickoff_utc": KICKOFF.isoformat(), "score_matrix": grid(),
    }


def verified(model, score):
    return {
        "provider": model["provider"], "fixture_id": model["fixture_id"],
        "home_team_id": model["home_team_id"],
        "away_team_id": model["away_team_id"],
        "kickoff_utc": model["kickoff_utc"],
        "status": "FT", "evidence_version": 2, "score_basis": "regulation",
        "verified_at_utc": (KICKOFF + timedelta(hours=3)).isoformat(),
        "home_goals": score[0], "away_goals": score[1],
    }


def setup(tmp_path, *, with_results=True):
    root, prematch, sets = (tmp_path / name for name in ("v10", "prematch", "sets"))
    a, b = forecast("100", "a"), forecast("200", "b")
    for original in (a, b):
        folder = root / "forecasts" / original["provider"] / original["fixture_id"]
        folder.mkdir(parents=True, exist_ok=True)
        (folder / (original["forecast_id"] + ".json")).write_text(json.dumps(original))
    options = joint_geometry(grid(), grid())["optimal_five"]
    baseline = [{"score1": x["score1"], "score2": x["score2"]} for x in options]
    challenger = baseline[:4] + [{"score1": "2-0", "score2": "1-1"}]
    packet = {
        "decision": "FROZEN_RESEARCH_ONLY",
        "model_version": "pro-analyst-v25-robust-sensitivity-shadow",
        "lower_envelope_not_a_calibrated_probability": True,
        "sydney_date": DAY, "selected_at_utc": SELECTED.isoformat(),
        "first": {k: a[k] for k in
                  ("provider", "fixture_id", "forecast_id", "home_team_id",
                   "away_team_id", "model_key", "model_version", "input_digest",
                   "captured_at_utc", "kickoff_utc")},
        "second": {k: b[k] for k in
                   ("provider", "fixture_id", "forecast_id", "home_team_id",
                    "away_team_id", "model_key", "model_version", "input_digest",
                    "captured_at_utc", "kickoff_utc")},
        "baseline_five": baseline, "five_doubles": challenger,
        "v8_optimal_five_model_mass": _selection_mass(
            [(tuple(map(int, x["score1"].split("-"))),
              tuple(map(int, x["score2"].split("-")))) for x in baseline],
            grid(), grid()
        ),
        "robust_five_v8_model_mass": _selection_mass(
            [(tuple(map(int, x["score1"].split("-"))),
              tuple(map(int, x["score2"].split("-")))) for x in challenger],
            grid(), grid()
        ),
    }
    sets.mkdir(parents=True)
    (sets / (DAY + ".json")).write_text(json.dumps(packet))
    if with_results:
        prematch.mkdir(parents=True)
        (prematch / "verified_results.json").write_text(json.dumps({
            "100": verified(a, (2, 0)),
            "200": verified(b, (1, 1)),
        }))
    return root, prematch, sets, packet, a, b


def overwrite_set(sets, packet):
    (sets / (DAY + ".json")).write_text(json.dumps(packet))


def test_same_pair_results_independently_reproduced(tmp_path):
    root, prematch, sets, packet, a, b = setup(tmp_path)
    original_bytes = (root / "forecasts/API-Football/100/a.json").read_bytes()
    report = build_report(root, prematch, sets, now=NOW)
    assert report["verified_matched_pairs"] == 1
    assert report["v8"]["five_double_hits"] == 0
    assert report["v25"]["five_double_hits"] == 1
    assert report["same_pair_discordance"]["v25_only_hits"] == 1
    assert report["same_pair_discordance"]["exact_two_sided_mcnemar_p"] == 1
    assert report["v8_exact_score_probability_calibration"]["evaluated_leg_slots"] == 2
    assert report["v25_probability_calibration"]["exact_score_log_loss"] is None
    assert report["production_model_promoted"] is False
    assert report["seventy_percent_accuracy_proven"] is False
    assert (root / "forecasts/API-Football/100/a.json").read_bytes() == original_bytes


def test_missing_final_is_pending_and_never_counted_as_loss(tmp_path):
    root, prematch, sets, *_ = setup(tmp_path, with_results=False)
    report = build_report(root, prematch, sets, now=NOW)
    assert report["source_valid_sets"] == 1
    assert report["pending_matched_pairs"] == 1
    assert report["verified_matched_pairs"] == 0
    assert report["v25"]["five_double_hit_rate"] is None


def test_conflicting_identity_verified_final_is_excluded(tmp_path):
    root, prematch, sets, _, a, _ = setup(tmp_path)
    root.mkdir(parents=True, exist_ok=True)
    other = verified(a, (4, 0))
    (root / "verified_results.json").write_text(json.dumps({"100": other}))
    report = build_report(root, prematch, sets, now=NOW)
    assert report["result_conflicts_excluded"] == 1
    assert report["pending_matched_pairs"] == 1
    assert report["verified_matched_pairs"] == 0


@pytest.mark.parametrize("changes", [
    {"evidence_version": 1},
    {"status": "NS", "home_goals": 3},
    {"home_goals": "2"},
    {"verified_at_utc": (NOW + timedelta(days=1)).isoformat(), "home_goals": 4},
])
def test_unqualified_rows_cannot_hide_a_verified_final(tmp_path, changes):
    root, prematch, sets, _, first, _ = setup(tmp_path)
    questionable = {"100": {**verified(first, (2, 0)), **changes}}
    (root / "verified_results.json").write_text(json.dumps(questionable))

    report = build_report(root, prematch, sets, now=NOW)

    assert report["result_source_rejections"] == {"invalid_or_unavailable_verified_final": 1}
    assert report["result_conflicts_excluded"] == 0
    assert report["verified_matched_pairs"] == 1
    assert report["v25"]["five_double_hits"] == 1


def test_post_kickoff_selection_is_not_reconstructed(tmp_path):
    root, prematch, sets, packet, *_ = setup(tmp_path)
    late = deepcopy(packet)
    late["selected_at_utc"] = (KICKOFF + timedelta(minutes=1)).isoformat()
    overwrite_set(sets, late)
    report = build_report(root, prematch, sets, now=NOW)
    assert report["source_valid_sets"] == 0
    assert report["source_exclusions"].get("late_or_backdated_source") == 1


def test_changed_original_digest_rejects_hindsight(tmp_path):
    root, prematch, sets, packet, a, _ = setup(tmp_path)
    path = root / "forecasts/API-Football/100/a.json"
    altered = deepcopy(a)
    altered["input_digest"] = "replaced-after-the-fact"
    path.write_text(json.dumps(altered))
    report = build_report(root, prematch, sets, now=NOW)
    assert report["source_valid_sets"] == 0
    assert report["source_exclusions"]["archived_source_identity_mismatch"] == 1


def test_saved_v8_choices_must_be_source_optimal(tmp_path):
    root, prematch, sets, packet, *_ = setup(tmp_path)
    changed = deepcopy(packet)
    changed["baseline_five"] = deepcopy(changed["five_doubles"])
    overwrite_set(sets, changed)
    report = build_report(root, prematch, sets, now=NOW)
    assert report["source_valid_sets"] == 0
    assert report["source_exclusions"]["saved_v8_baseline_not_source_optimal"] == 1


def test_wrong_saved_fixture_identity_and_invalid_v25_score_rejected(tmp_path):
    root, prematch, sets, packet, *_ = setup(tmp_path)
    wrong = deepcopy(packet)
    wrong["first"]["home_team_id"] = "999"
    overwrite_set(sets, wrong)
    assert build_report(root, prematch, sets, now=NOW)["source_exclusions"] == {
        "archived_source_identity_mismatch": 1
    }
    wrong = deepcopy(packet)
    wrong["five_doubles"][0]["score1"] = "fake"
    overwrite_set(sets, wrong)
    assert build_report(root, prematch, sets, now=NOW)["source_exclusions"] == {
        "invalid_selected_score": 1
    }


def test_v25_lower_envelope_not_misreported_as_calibration(tmp_path):
    root, prematch, sets, packet, *_ = setup(tmp_path)
    packet["lower_envelope_not_a_calibrated_probability"] = False
    overwrite_set(sets, packet)
    report = build_report(root, prematch, sets, now=NOW)
    assert report["source_valid_sets"] == 0
    assert report["source_exclusions"]["not_original_v25_frozen_set"] == 1


def test_frozen_v25_probabilities_use_same_verified_legs_and_preserve_five_double_grade(tmp_path):
    from professional_analyst_shadow_v25 import propose

    root, prematch, sets, packet, first, second = setup(tmp_path)
    for model in (first, second):
        model.update(home_team="Home", away_team="Away")
    frozen = propose([first, second], SELECTED, min_envelope=0)["days"][DAY]
    for leg in ("first", "second"):
        packet[leg]["v25_probability"] = frozen[leg]["v25_probability"]
    packet["mixture_five"] = frozen["mixture_five"]
    packet["mixture_five_policy"] = frozen["mixture_five_policy"]
    overwrite_set(sets, packet)

    pending = build_report(root, prematch, sets, now=SELECTED)
    assert pending["matched_mixture_five"]["pending_pairs"] == 1
    assert pending["matched_mixture_five"]["v25_mixture"]["five_double_hit_rate"] is None
    assert pending["matched_probability_calibration"]["pending_pairs"] == 1
    assert pending["v25_probability_calibration"]["evaluated_leg_slots"] == 0

    report = build_report(root, prematch, sets, now=NOW)
    comparison = report["matched_probability_calibration"]
    assert comparison["verified_pairs"] == 1
    assert comparison["v8_on_same_legs"]["evaluated_leg_slots"] == 2
    v25 = report["v25_probability_calibration"]
    assert v25["evaluated_leg_slots"] == 2
    assert v25["exact_score_log_loss"] != comparison["v8_on_same_legs"]["exact_score_log_loss"]
    assert report["v25"]["five_double_hits"] == 1
    assert report["matched_mixture_five"]["verified_pairs"] == 1
    assert report["matched_mixture_five"]["v25_robust"]["five_double_hits"] == 1
    assert report["rows"][0]["v25_mixture_five_model_mass"] >= (
        report["rows"][0]["v25_robust_five_mass_under_mixture"]
    )

    malformed = deepcopy(packet)
    malformed["mixture_five"][0]["joint_probability"] += 0.1
    overwrite_set(sets, malformed)
    rejected = build_report(root, prematch, sets, now=NOW)
    assert rejected["rows"][0]["v25_mixture_five_status"] == "invalid"
    assert rejected["matched_mixture_five"]["verified_pairs"] == 0
    assert rejected["matched_probability_calibration"]["verified_pairs"] == 1

    packet["first"]["v25_probability"]["frozen_at_utc"] = NOW.isoformat()
    overwrite_set(sets, packet)
    invalid = build_report(root, prematch, sets, now=NOW)
    assert invalid["rows"][0]["v25_probability_status"] == "invalid"
    assert invalid["v25_probability_calibration"]["evaluated_leg_slots"] == 0
    assert invalid["v25"]["five_double_hits"] == 1
