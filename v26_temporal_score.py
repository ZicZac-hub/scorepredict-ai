"""V26 research-only time-causal score-tail calibration and same-pair trial.

Only pre-match archived V8 matrices and *independently verified* earlier finals
may influence a challenger. A future V25 frozen pair gives the same two fixtures.
No odds/LLM requests, no rewriting either forecast, no production promotion.

The retrospective chronological comparison is EXPLORATORY; truly prospective
V26 results only arise after this version is registered and frozen before kickoff.
"""
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile
import hashlib
import json
import math

import numpy as np

from five_double_accuracy import _final, _validated_grid, joint_geometry, utc, wilson
from v10_prospective import ROOT as SOURCE_ROOT, read_latest
from verified_result_index import index_verified_results
from v8_v25_matched_audit import _frozen_set
from v25_score_accuracy_research import describe_score, summarize_scores

VERSION = "v26-temporal-tail-calibration-v1-research-only"
TRAIN_MIN = 30
PRIOR_EQUIVALENT = 80.0
RATIO_MIN = 0.8
RATIO_MAX = 1.2
CORRECTION_POWER = 0.65
MAX_SELECTION_AGE = timedelta(minutes=20)
MIN_LEAD = timedelta(minutes=45)
ORIGINAL_V25_SETS = Path("data/pro_analyst_shadow/sets")
OUT = Path("data/v26_temporal_calibration")
BANDS = ("0_1_goals", "2_3_goals", "4_plus_goals")


def _masks(shape):
    h, a = np.indices(shape)
    goals = h + a
    return (goals <= 1, (goals >= 2) & (goals <= 3), goals >= 4)


def _receipt(row):
    return {
        "provider": str(row["source"]["provider"]),
        "fixture_id": str(row["source"]["fixture_id"]),
        "forecast_id": str(row["source"]["forecast_id"]),
        "input_digest": str(row["source"]["input_digest"]),
        "verified_at_utc": utc(row["verified_at"]).isoformat(),
        "home_goals": row["score"][0],
        "away_goals": row["score"][1],
    }


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def original_rows(root=SOURCE_ROOT):
    """One earliest originally captured V8 forecast per physical fixture."""
    data = []
    for value in read_latest(root).values():
        try:
            if value.get("model_key") != "v8":
                continue
            if (value.get("mode") != "prospective" or
                    value.get("score_basis") != "regulation" or
                    not value.get("input_digest") or
                    int(value.get("history_matches") or 0) < 45):
                continue
            captured = utc(value["captured_at_utc"])
            evidence = utc(value["evidence_as_of_utc"])
            kickoff = utc(value["kickoff_utc"])
            if not evidence <= captured <= kickoff - MIN_LEAD:
                continue
            matrix = _validated_grid(value["score_matrix"])
            if min(len(matrix), len(matrix[0])) < 5 or max(
                    len(matrix), len(matrix[0])) > 25:
                continue
            if str(value["home_team_id"]) == str(value["away_team_id"]):
                continue
            data.append({"source": value, "grid": np.asarray(matrix),
                         "captured": captured, "kickoff": kickoff})
        except (KeyError, ValueError, TypeError, OverflowError):
            continue
    return sorted(data, key=lambda x: (
        x["captured"], str(x["source"]["provider"]),
        str(x["source"]["fixture_id"])))


def finished_samples(root=SOURCE_ROOT, prematch=Path("data/prematch"), *, now=None):
    """Only identity-matched, timestamped finals available at the cutoff."""
    clock = utc(now or datetime.now(timezone.utc))
    index, conflicts, rejected = index_verified_results(root, prematch, clock)
    rows = []
    for source in original_rows(root):
        model = source["source"]
        result = index.get((str(model["provider"]), str(model["fixture_id"])))
        if result is None:
            continue
        actual = _final(model, result, clock)
        if actual is None:
            continue
        try:
            verified_at = utc(result["verified_at_utc"])
            if not source["kickoff"] < verified_at <= clock:
                continue
        except (ValueError, KeyError, TypeError):
            continue
        rows.append({**source, "score": actual, "verified_at": verified_at})
    return rows, {"result_conflicts_excluded": len(conflicts),
                  "invalid_result_sources_excluded": dict(rejected)}


def calibrate(original, history):
    """Fixed shrinkage of three goal-total probability bins using PRIOR results.

    The baseline score grid is never refitted. Correction changes are small,
    constrained, predeclared and measured only by later verified outcomes.
    The class bins are not team, injury or lineup features.
    """
    base = np.asarray(_validated_grid(original), dtype=float)
    if len(history) < TRAIN_MIN:
        return None
    expected = np.zeros(3, dtype=float)
    observed = np.zeros(3, dtype=float)
    for row in history:
        older = row["grid"]
        if not isinstance(older, np.ndarray):
            older = np.asarray(_validated_grid(older), dtype=float)
        if (older.ndim != 2 or not np.isfinite(older).all() or
                np.any(older < 0) or not np.isclose(older.sum(), 1, atol=1e-6)):
            raise ValueError("invalid_prior_frozen_score_grid")
        expected += np.asarray([older[mask].sum() for mask in _masks(older.shape)])
        h, a = row["score"]
        if type(h) is not int or type(a) is not int or min(h, a) < 0:
            raise ValueError("invalid_independently_verified_goals")
        total = h + a
        observed[0 if total <= 1 else 1 if total <= 3 else 2] += 1
    # The fixed prior effectively shrinks historical observed-minus-forecast
    # bin differences towards zero. Do NOT optimize coefficients on outcomes.
    n = len(history)
    correction = (observed - expected) / (n + PRIOR_EQUIVALENT)
    predicted_rate = expected / n
    ratios = np.clip(
        (predicted_rate + correction) / np.maximum(predicted_rate, 1e-9),
        RATIO_MIN, RATIO_MAX,
    ) ** CORRECTION_POWER
    result = np.zeros_like(base)
    for mask, ratio in zip(_masks(base.shape), ratios):
        result[mask] = base[mask] * ratio
    mass = float(result.sum())
    if not math.isfinite(mass) or mass <= 0:
        raise ValueError("invalid_calibrated_probability_mass")
    result /= mass
    diagnostics = {
        "version": VERSION, "verified_training_fixtures": n,
        "prior_equivalent_fixtures": PRIOR_EQUIVALENT,
        "model_predicted_bin_share": dict(zip(BANDS, (expected/n).tolist())),
        "training_observed_bin_share": dict(zip(BANDS, (observed/n).tolist())),
        "bounded_bin_multipliers": dict(zip(BANDS, ratios.tolist())),
        "total_variation_from_original_v8": float(np.abs(result-base).sum()/2),
        "unverified_accuracy_improvement": True,
    }
    return result, diagnostics


def _compatible_training(rows, *, cutoff, excluded_fixtures=()):
    omit = set(excluded_fixtures)
    return sorted((
        row for row in rows
        if row["verified_at"] <= cutoff and row["kickoff"] < cutoff
        and (str(row["source"]["provider"]), str(row["source"]["fixture_id"])) not in omit
    ), key=lambda row: (
        row["verified_at"], str(row["source"]["provider"]),
        str(row["source"]["fixture_id"])))


def replay(root=SOURCE_ROOT, prematch=Path("data/prematch"), *, now=None):
    """Time-causal but retrospective *research method* on true past freezes.

    The algorithm is designed NOW: old backtest wins are not prospective V26
    wins. Each test input is an originally saved V8 forecast and calibration
    uses only results whose recorded verification time precedes its capture.
    """
    clock = utc(now or datetime.now(timezone.utc))
    samples, conflicts = finished_samples(root, prematch, now=clock)
    baseline, challenger = [], []
    tested = []
    for row in sorted(samples, key=lambda r: (r["captured"], r["source"]["fixture_id"])):
        training = _compatible_training(
            samples, cutoff=row["captured"],
            excluded_fixtures={(str(row["source"]["provider"]),
                                str(row["source"]["fixture_id"]))})
        result = calibrate(row["grid"].tolist(), training)
        if result is None:
            continue
        candidate, diagnostic = result
        baseline.append(describe_score(row["grid"], row["score"]))
        challenger.append(describe_score(candidate, row["score"]))
        tested.append({"fixture_id": str(row["source"]["fixture_id"]),
                       "source_forecast_id": str(row["source"]["forecast_id"]),
                       "source_captured_at_utc": row["captured"].isoformat(),
                       "training_count_before_source": len(training),
                       "verified_actual_score": f'{row["score"][0]}-{row["score"][1]}',
                       "baseline_log_loss": baseline[-1]["exact_score_log_loss"],
                       "calibrated_log_loss": challenger[-1]["exact_score_log_loss"]})
    ref = summarize_scores(baseline)
    cand = summarize_scores(challenger)
    return {
        "kind": "retrospective_chronological_replay_from_real_original_pre_match_forecasts",
        "version": VERSION,
        "finished_original_v8_fixtures": len(samples),
        "trained_before_each_source_capture": True,
        "minimum_prior_verified_matches": TRAIN_MIN,
        "verified_holdout_fixtures": len(tested),
        "baseline_v8": ref,
        "v26_tail_candidate": cand,
        "v26_minus_v8_log_loss": (
            cand["mean_exact_score_log_loss"] - ref["mean_exact_score_log_loss"]
            if tested else None),
        "v26_minus_v8_multiclass_brier": (
            cand["mean_multiclass_score_brier"] - ref["mean_multiclass_score_brier"]
            if tested else None),
        "smaller_proper_score_is_better": True,
        "holdout_source_receipts": tested,
        "original_verified_result_quality": conflicts,
        "original_forecasts_modified": False,
        "no_live_model_promotion": True,
        "prospective_v26_accuracy_proven": False,
        "limitations": [
            "The V26 adjustment was developed after these original matches.",
            "A chronological replay is not a genuine prospective V26 experiment.",
            "A small overlapping sample does not establish live five-double wins.",
        ],
    }


def _write_once(path, packet):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return False
    data = json.dumps(packet, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with NamedTemporaryFile(dir=path.parent, mode="w", encoding="utf-8",
                            delete=False) as temp:
        staged = Path(temp.name)
        try:
            temp.write(data)
            temp.flush()
            temp.close()
            try:
                path.hardlink_to(staged)
                return True
            except FileExistsError:
                return False
        finally:
            staged.unlink()


def freeze(*, root=SOURCE_ROOT, prematch=Path("data/prematch"),
           sets=ORIGINAL_V25_SETS, out=OUT, now=None):
    """Freeze a NEW v26 arm on the same newly-frozen V25 match pair.

    A V25 set saved prior to this run, even if games are still upcoming,
    cannot be retrofitted with a V26 probability distribution.
    """
    clock = utc(now or datetime.now(timezone.utc))
    historical, _ = finished_samples(root, prematch, now=clock)
    created, abstentions = [], Counter()
    for path in sorted(Path(sets).glob("*.json")):
        try:
            packet, originals, grids, _, _, _ = _frozen_set(path, root, clock)
            selected = utc(packet["selected_at_utc"])
            if not selected <= clock <= selected + MAX_SELECTION_AGE:
                abstentions["v25_set_not_new_in_same_freeze_window"] += 1
                continue
            kickoffs = [utc(source["kickoff_utc"]) for source in originals]
            if any(clock > kickoff - MIN_LEAD for kickoff in kickoffs):
                abstentions["too_close_to_kickoff"] += 1
                continue
            day = packet["sydney_date"]
            destination = Path(out) / "sets" / (day + ".json")
            if destination.exists():
                continue
            ids = {(str(src["provider"]), str(src["fixture_id"])) for src in originals}
            training = _compatible_training(historical, cutoff=selected,
                                           excluded_fixtures=ids)
            if len(training) < TRAIN_MIN:
                abstentions["fewer_than_30_before_cutoff_verified_sources"] += 1
                continue
            left, right = [], []
            for source, grid in zip(originals, grids):
                result = calibrate(grid, training)
                if result is None:
                    raise ValueError("qualified_calibration_unavailable")
                adjusted, diagnostic = result
                leg = {key: source[key] for key in (
                    "provider", "fixture_id", "forecast_id", "input_digest",
                    "home_team_id", "away_team_id", "kickoff_utc")}
                leg["score_matrix"] = adjusted.tolist()
                leg["calibration"] = diagnostic
                (left if not left else right).append(leg)
            receipts = [_receipt(row) for row in training]
            picks = joint_geometry(left[0]["score_matrix"], right[0]["score_matrix"])[
                "optimal_five"]
            saved = {
                "schema_version": 1, "decision": "V26_SHADOW_FROZEN",
                "version": VERSION, "sydney_date": day,
                "selected_at_utc": clock.isoformat(),
                "v25_original_selected_at_utc": selected.isoformat(),
                "first": left[0], "second": right[0],
                "calibrated_five": picks,
                "original_v8_five": packet["baseline_five"],
                "original_v25_robust_five": packet["five_doubles"],
                "verified_training_receipts": receipts,
                "training_receipts_sha256": hashlib.sha256(
                    _canonical(receipts).encode("utf-8")).hexdigest(),
                "training_available_by_v25_selection": True,
                "original_v25_pair_untouched": True,
                "no_new_provider_requests": True,
                "production_model_promoted": False,
                "70_percent_proven": False,
            }
            if _write_once(destination, saved):
                created.append(day)
        except (OSError, ValueError, KeyError, TypeError, OverflowError):
            abstentions["invalid_or_unreproducible_v25_source"] += 1
    return {"new_frozen_days": created, "abstentions": dict(abstentions),
            "model": VERSION, "live_model_changed": False}


def _score_hit(picks, actual):
    return any((p["score1"], p["score2"]) == tuple(
        f"{h}-{a}" for h, a in actual) for p in picks)


def evaluate(*, root=SOURCE_ROOT, prematch=Path("data/prematch"),
             sets=ORIGINAL_V25_SETS, out=OUT, now=None):
    """Audit frozen future V26 only; recalculate saved matrices from prior receipts."""
    clock = utc(now or datetime.now(timezone.utc))
    original_lookup = {
        (str(row["source"]["provider"]), str(row["source"]["fixture_id"]),
         str(row["source"]["forecast_id"])): row
        for row in original_rows(root)
    }
    known, _ = finished_samples(root, prematch, now=clock)
    sample_lookup = {
        (str(row["source"]["provider"]), str(row["source"]["fixture_id"]),
         str(row["source"]["forecast_id"])): row
        for row in known
    }
    index, _, _ = index_verified_results(root, prematch, clock)
    evaluated, invalid = [], Counter()
    for path in sorted((Path(out)/"sets").glob("*.json")):
        try:
            saved = json.loads(path.read_text(encoding="utf-8"))
            if (saved.get("decision") != "V26_SHADOW_FROZEN" or
                    saved.get("version") != VERSION or
                    saved.get("sydney_date") != path.stem):
                raise ValueError("wrong_model_or_day")
            source_file = Path(sets)/(path.stem+".json")
            orig, originals, grids, _, _, _ = _frozen_set(source_file, root, clock)
            freeze_at = utc(saved["selected_at_utc"])
            orig_at = utc(orig["selected_at_utc"])
            if (utc(saved["v25_original_selected_at_utc"]) != orig_at or
                    not orig_at <= freeze_at <= orig_at+MAX_SELECTION_AGE or
                    freeze_at > clock or
                    any(freeze_at > utc(m["kickoff_utc"])-MIN_LEAD
                        for m in originals)):
                raise ValueError("selection_time_ineligible")
            receipts = saved["verified_training_receipts"]
            if (not isinstance(receipts, list) or len(receipts) < TRAIN_MIN or
                    hashlib.sha256(_canonical(receipts).encode("utf-8")).hexdigest() !=
                    saved["training_receipts_sha256"]):
                raise ValueError("unverified_training_receipts")
            training, seen = [], set()
            pair_ids = {(str(x["provider"]), str(x["fixture_id"])) for x in originals}
            for receipt in receipts:
                key = (str(receipt["provider"]), str(receipt["fixture_id"]),
                       str(receipt["forecast_id"]))
                example = sample_lookup.get(key)
                if (example is None or key in seen or key[:2] in pair_ids or
                        example["verified_at"] > orig_at or
                        _receipt(example) != receipt):
                    raise ValueError("training_receipt_not_originally_verified")
                seen.add(key)
                training.append(example)
            ordered = _compatible_training(
                training, cutoff=orig_at, excluded_fixtures=pair_ids)
            if [_receipt(r) for r in training] != [_receipt(r) for r in ordered]:
                raise ValueError("training_receipts_order_or_cutoff_changed")
            for source, grid, side in zip(originals, grids, ("first", "second")):
                frozen = saved[side]
                if any(str(frozen.get(key)) != str(source[key]) for key in (
                        "provider", "fixture_id", "forecast_id", "input_digest",
                        "home_team_id", "away_team_id", "kickoff_utc")):
                    raise ValueError("new_shadow_source_identity_mismatch")
                candidate, _ = calibrate(grid, training)
                if not np.allclose(
                        np.asarray(frozen["score_matrix"], dtype=float),
                        candidate, atol=1e-11, rtol=0):
                    raise ValueError("v26_frozen_matrix_not_reproducible")
            expected = joint_geometry(
                saved["first"]["score_matrix"], saved["second"]["score_matrix"])[
                    "optimal_five"]
            if (len(saved["calibrated_five"]) != 5 or
                    any((p["score1"], p["score2"]) !=
                        (q["score1"], q["score2"]) or
                        not math.isclose(p["joint_probability"],
                                         q["joint_probability"], abs_tol=1e-12, rel_tol=0)
                        for p, q in zip(saved["calibrated_five"], expected))):
                raise ValueError("v26_saved_joint_score_mismatch")
            finals = [
                _final(source, index.get((str(source["provider"]),
                                          str(source["fixture_id"]))), clock)
                for source in originals
            ]
            pending = any(x is None for x in finals)
            row = {
                "sydney_date": path.stem,
                "status": "pending" if pending else "verified",
                "training_fixtures_before_selection": len(training),
                "original_v25_source_selected_at_utc": orig_at.isoformat(),
                "new_v26_selection_at_utc": freeze_at.isoformat(),
            }
            if not pending:
                row.update({
                    "actual_scores": [f"{h}-{a}" for h, a in finals],
                    "v26_five_hit": _score_hit(saved["calibrated_five"], finals),
                    "v25_robust_five_hit": _score_hit(orig["five_doubles"], finals),
                    "v8_same_pair_five_hit": _score_hit(orig["baseline_five"], finals),
                    "v26_leg_score_metrics": [
                        describe_score(saved[side]["score_matrix"], score)
                        for side, score in zip(("first", "second"), finals)],
                    "v8_leg_score_metrics": [
                        describe_score(grid, score)
                        for grid, score in zip(grids, finals)],
                })
            evaluated.append(row)
        except (OSError, ValueError, KeyError, TypeError, OverflowError) as error:
            invalid[str(error) if isinstance(error, ValueError)
                    else type(error).__name__] += 1
    complete = [r for r in evaluated if r["status"] == "verified"]
    v8_legs = [s for r in complete for s in r["v8_leg_score_metrics"]]
    v26_legs = [s for r in complete for s in r["v26_leg_score_metrics"]]
    base, challenger = summarize_scores(v8_legs), summarize_scores(v26_legs)
    return {
        "schema_version": 1, "version": VERSION,
        "study": "genuinely_frozen_v26_on_same_original_v25_fixture_pairs",
        "verified_pairs": len(complete),
        "pending_pairs": len(evaluated) - len(complete),
        "invalid_or_tampered_sets_excluded": dict(invalid),
        "same_pair_five_hits": {
            "v8": sum(r["v8_same_pair_five_hit"] for r in complete),
            "v25_robust": sum(r["v25_robust_five_hit"] for r in complete),
            "v26_calibrated": sum(r["v26_five_hit"] for r in complete),
        },
        "v26_five_hit_wilson_95": wilson(
            sum(r["v26_five_hit"] for r in complete), len(complete)),
        "same_legs_v8_score_metrics": base,
        "same_legs_v26_score_metrics": challenger,
        "v26_minus_v8_log_loss": (
            challenger["mean_exact_score_log_loss"] -
            base["mean_exact_score_log_loss"] if complete else None),
        "v26_minus_v8_multiclass_brier": (
            challenger["mean_multiclass_score_brier"] -
            base["mean_multiclass_score_brier"] if complete else None),
        "rows": evaluated,
        "no_historical_v26_probability_backfill": True,
        "no_model_promotion": True,
        "70_percent_proven": False,
    }


def run(*, root=SOURCE_ROOT, prematch=Path("data/prematch"),
        sets=ORIGINAL_V25_SETS, out=OUT, now=None, verify_only=False):
    clock = utc(now or datetime.now(timezone.utc))
    saved = ({"new_frozen_days": [], "abstentions": {},
              "model": VERSION, "live_model_changed": False}
             if verify_only else freeze(root=root, prematch=prematch,
                                        sets=sets, out=out, now=clock))
    report = {
        "version": VERSION,
        "evaluated_at_utc": clock.isoformat(),
        "prospective_freeze": saved,
        "chronological_research_replay": replay(root, prematch, now=clock),
        "prospective_verified": evaluate(root=root, prematch=prematch,
                                         sets=sets, out=out, now=clock),
        "no_provider_calls": True,
        "original_frozen_forecasts_changed": False,
        "production_model_promoted": False,
        "seventy_percent_proven": False,
    }
    file = Path(out)/"status.json"
    file.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if not file.exists() or file.read_text(encoding="utf-8") != data:
        file.write_text(data, encoding="utf-8")
    return report


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true",
                        help="Never generate a new V26 set in result-backfill jobs")
    args = parser.parse_args()
    report = run(verify_only=args.verify_only)
    print(json.dumps({
        "new_frozen": report["prospective_freeze"]["new_frozen_days"],
        "replay_holdout": report["chronological_research_replay"]["verified_holdout_fixtures"],
        "replay_log_loss_delta": report["chronological_research_replay"]["v26_minus_v8_log_loss"],
        "prospective_verified": report["prospective_verified"]["verified_pairs"],
        "prospective_pending": report["prospective_verified"]["pending_pairs"],
        "promoted": False,
    }, sort_keys=True))
