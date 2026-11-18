"""Read-only post-holdout calibration and provenance audit of frozen research.

No fitting, threshold selection, source replacement or model promotion occurs.
All comparisons below are descriptive analyses after outcomes were revealed.
"""
import argparse
from collections import Counter
from datetime import date, datetime, timedelta, timezone
import gzip
import hashlib
from itertools import combinations
import json
import math
from pathlib import Path

from five_double_accuracy import wilson

ROOT = Path("data/target70_research")
CACHE = Path("/workspace/shared/target70-data")
BIN_EDGES = (0., .05, .075, .10, .125, .15, .20, .30, .50, .70, 1.)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    path = Path(path)
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    compressed = Path(str(path)+".gz")
    with gzip.open(compressed, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def calibration(observations, *, edges=BIN_EDGES, constant_probability=None):
    """Binary set-hit probability quality; abstentions have no forecast score."""
    if len(edges) < 2 or edges[0] != 0 or edges[-1] != 1 or any(
            a >= b for a, b in zip(edges, edges[1:])):
        raise ValueError("invalid_reliability_bin_edges")
    checked = []
    for p, hit in observations:
        if type(p) not in (float, int) or not math.isfinite(p) or not 0 <= p <= 1:
            raise ValueError("invalid_set_probability")
        if type(hit) is not bool:
            raise ValueError("set_hit_must_be_boolean")
        checked.append((float(p), int(hit)))
    n = len(checked)
    hits = sum(hit for _, hit in checked)
    bins = []
    for i, (low, high) in enumerate(zip(edges, edges[1:])):
        matched = [(p, hit) for p, hit in checked if low <= p < high or
                   (i == len(edges)-2 and p == 1)]
        count = len(matched)
        wins = sum(hit for _, hit in matched)
        predicted = math.fsum(p for p, _ in matched)/count if count else None
        observed = wins/count if count else None
        bins.append({"low_inclusive": low, "high": high, "high_inclusive": i == len(edges)-2,
                     "issued": count, "hits": wins, "mean_prediction": predicted,
                     "observed_hit_rate": observed, "wilson_95_descriptive": wilson(wins, count),
                     "observed_minus_predicted": observed-predicted if count else None})
    mean = math.fsum(p for p, _ in checked)/n if n else None
    result = {"issued": n, "hits": hits, "observed_hit_rate": hits/n if n else None,
              "wilson_95_descriptive": wilson(hits, n), "mean_model_coverage": mean,
              "observed_minus_predicted": hits/n-mean if n else None,
              "binary_brier": math.fsum((p-hit)**2 for p, hit in checked)/n if n else None,
              "binary_log_loss": -math.fsum(math.log(max(1e-15, p if hit else 1-p))
                                            for p, hit in checked)/n if n else None,
              "fixed_bin_expected_calibration_error": math.fsum(
                  b["issued"]*abs(b["observed_minus_predicted"]) for b in bins if b["issued"])/n if n else None,
              "reliability_bins": bins, "calibration_fitted_or_applied": False,
              "bins_declared_after_holdout_for_description_only": True}
    if constant_probability is not None:
        if not 0 <= constant_probability <= 1:
            raise ValueError("invalid_validation_constant_probability")
        result["validation_prevalence_constant_brier"] = math.fsum(
            (constant_probability-hit)**2 for _, hit in checked)/n if n else None
        result["validation_prevalence_constant_probability"] = constant_probability
    return result


def _score(value):
    if not isinstance(value, list) or len(value) != 2 or any(type(g) is not int or g < 0 for g in value):
        raise ValueError("invalid_saved_exact_score")
    return tuple(value)


def validate_day_records(season):
    seen = set()
    for day in season["date_records"]:
        source_date = day["source_date"]
        date.fromisoformat(source_date)
        if source_date in seen:
            raise ValueError("duplicate_source_date")
        seen.add(source_date)
        for row in day["models"].values():
            if row is None:
                continue
            if len(row["match_ids"]) != 2 or len(set(row["match_ids"])) != 2:
                raise ValueError("two_distinct_matches_required")
            if len(row["fixtures"]) != 2 or any(f["date"] != source_date for f in row["fixtures"]):
                raise ValueError("pair_date_mismatch")
            teams = {(f["league"], f[side]) for f in row["fixtures"] for side in ("home", "away")}
            if len(teams) != 4:
                raise ValueError("four_distinct_teams_required")
            if len(row["fit_month_starts"]) != 2 or len(row["last_training_source_dates"]) != 2:
                raise ValueError("two_saved_training_cutoffs_required")
            for month, last in zip(row["fit_month_starts"], row["last_training_source_dates"]):
                month_start = date.fromisoformat(month)
                if month_start.day != 1 or month_start > date.fromisoformat(source_date) or not date.fromisoformat(last) < month_start-timedelta(days=1):
                    raise ValueError("saved_training_cutoff_violates_protocol")
            picks = row["five_doubles"]
            if len(picks) != 5 or len({(_score(p["score1"]), _score(p["score2"])) for p in picks}) != 5:
                raise ValueError("five_distinct_double_outcomes_required")
            probabilities = [p["joint_probability"] for p in picks]
            if any(type(p) not in (float, int) or not math.isfinite(p) or not 0 <= p <= 1 for p in probabilities):
                raise ValueError("invalid_saved_joint_probability")
            if not math.isclose(math.fsum(probabilities), row["estimated_coverage"], rel_tol=0, abs_tol=1e-10):
                raise ValueError("saved_coverage_disagrees_with_five_rows")
            if len(row["actual_scores"]) != 2:
                raise ValueError("two_actual_scores_required")
            actual = tuple(_score(s) for s in row["actual_scores"])
            observed = any((_score(p["score1"]), _score(p["score2"])) == actual for p in picks)
            if type(row["hit"]) is not bool or row["hit"] != observed:
                raise ValueError("saved_hit_disagrees_with_actual_same_row")


def validate_source_results(season, source_rows):
    """Bind scored pairs back to the pinned raw-source match identities."""
    source = {r["source_match_id"]: r for r in source_rows}
    for day in season["date_records"]:
        for pair in day["models"].values():
            if pair is None:
                continue
            for identity, fixture, actual in zip(pair["match_ids"], pair["fixtures"], pair["actual_scores"]):
                original = source.get(identity)
                if original is None or any(fixture[k] != original[field] for k, field in (
                        ("league", "competition"), ("date", "source_date"),
                        ("home", "home_team"), ("away", "away_team"))):
                    raise ValueError("saved_pair_fixture_not_in_pinned_source")
                if actual != [original["home_goals"], original["away_goals"]]:
                    raise ValueError("saved_actual_score_disagrees_with_pinned_source")


def _summary(days, model, threshold=0, *, constant_probability=None):
    issued = [d["models"][model] for d in days if d["models"].get(model) is not None and
              d["models"][model]["estimated_coverage"] >= threshold]
    return {"threshold": threshold, "source_date_opportunities": len(days),
            "issued_sets": len(issued), "abstained_dates": len(days)-len(issued),
            "wins_per_source_date_opportunity": sum(r["hit"] for r in issued)/len(days) if days else None,
            **calibration([(r["estimated_coverage"], r["hit"]) for r in issued],
                          constant_probability=constant_probability)}


def _paired_outcomes(left, right):
    if len(left) != len(right):
        raise ValueError("paired_outcome_lengths_differ")
    n = len(left)
    first_only = sum(a and not b for a, b in zip(left, right))
    second_only = sum(b and not a for a, b in zip(left, right))
    discordant = first_only+second_only
    p = min(1., 2*sum(math.comb(discordant, k) for k in range(min(first_only, second_only)+1))/2**discordant) if discordant else 1.
    return {"paired_dates": n, "first_wins": sum(left), "second_wins": sum(right),
            "first_only_wins": first_only, "second_only_wins": second_only,
            "first_minus_second_hit_rate": (sum(left)-sum(right))/n if n else None,
            "mcnemar_exact_two_sided_descriptive_p": p if n else None}


def policy_comparisons(season):
    """Compare declared policies on shared dates; each may choose other games."""
    days = season["date_records"]
    models = sorted(season["single_match_metrics"])
    records = []
    for first, second in combinations(models, 2):
        common = [d for d in days if d["models"].get(first) is not None and d["models"].get(second) is not None]
        same = [d for d in common if set(d["models"][first]["match_ids"]) ==
                set(d["models"][second]["match_ids"])]
        records.append({"first_policy": first, "second_policy": second,
            "all_source_dates_abstention_counts_as_no_winning_set": _paired_outcomes(
                [bool(d["models"].get(first) and d["models"][first]["hit"]) for d in days],
                [bool(d["models"].get(second) and d["models"][second]["hit"]) for d in days]),
            "both_policies_issued": _paired_outcomes(
                [d["models"][first]["hit"] for d in common], [d["models"][second]["hit"] for d in common]),
            "same_two_matches_subgroup": _paired_outcomes(
                [d["models"][first]["hit"] for d in same], [d["models"][second]["hit"] for d in same]),
            "different_pair_dates_among_joint_issued": len(common)-len(same),
            "post_holdout_exploration_not_model_selection": True,
            "independent_days_assumption_and_multiple_comparisons_unadjusted": True})
    return records


def source_audit(manifest, cache_dir, holdout_season):
    raw_total = played_total = 0
    rejected = Counter()
    season_counts = {}
    missing = []
    hashes = []
    scheduled_holdout_dates = set()
    missing_holdout_dates = Counter()
    cache = Path(cache_dir).resolve()
    for entry in manifest["source_files"]:
        path = (cache/entry["cache_relative_path"]).resolve()
        if cache not in path.parents:
            raise ValueError("source_path_outside_cache")
        if file_hash(path) != entry["sha256"]:
            raise ValueError("pinned_raw_file_hash_mismatch")
        hashes.append({"source_url": entry["source_url"], "sha256": entry["sha256"]})
        raw = load(path)["matches"]
        raw_total += len(raw)
        played_total += entry["normalized_played_rows"]
        rejected.update(entry["rejected_rows"])
        counts = season_counts.setdefault(entry["season"], {"raw_rows": 0, "normalized_played": 0})
        counts["raw_rows"] += len(raw)
        counts["normalized_played"] += entry["normalized_played_rows"]
        for row in raw:
            if entry["season"] == holdout_season:
                scheduled_holdout_dates.add(row["date"])
            if not isinstance((row.get("score") or {}).get("ft"), list):
                missing.append({"season": entry["season"], "league": entry["competition"],
                                "source_date": row["date"], "status": row.get("status", "unspecified")})
                if entry["season"] == holdout_season:
                    missing_holdout_dates[row["date"]] += 1
    return {"pinned_raw_files_hash_verified": len(hashes), "raw_rows": raw_total,
            "normalized_played_rows": played_total, "rejected_rows_by_reason": dict(rejected),
            "counts_by_season": season_counts,
            "missing_full_time_by_season_and_status": dict(Counter(
                r["season"]+"|"+r["status"] for r in missing)),
            "holdout_raw_scheduled_source_dates": len(scheduled_holdout_dates),
            "holdout_dates_with_missing_full_time": dict(sorted(missing_holdout_dates.items())),
            "holdout_missing_full_time_rows": sum(missing_holdout_dates.values()),
            "raw_source_references": hashes}, scheduled_holdout_dates, set(missing_holdout_dates)


def build_report(root=ROOT, cache_dir=CACHE, *, repository_root=None):
    root = Path(root)
    repository_root = Path(repository_root or Path(__file__).resolve().parent)
    manifest = load(root/"source_manifest.json")
    protocol = load(root/"protocol.json")
    selection = load(root/"selection_before_holdout.json")
    implementation = load(root/"implementation_before_holdout.json")
    validation = load(root/"validation.json")
    holdout_wrapper = load(root/"holdout.json")
    holdout = holdout_wrapper["holdout"]
    chosen = selection["selected_model"]
    # Reuse the frozen adapter and its canonical alias mapping, without ever
    # calling its fitter or replacing any of its saved observations.
    from target70_data import load_normalized
    from target70_benchmark import validate_rows
    normalized_source = load_normalized(root/"source_manifest.json", cache_dir)
    integrity = {
        "benchmark_code_still_matches_pre_holdout_hash": file_hash(repository_root/"target70_benchmark.py") == implementation["benchmark_implementation_sha256"],
        "selection_file_still_matches_pre_holdout_hash": file_hash(root/"selection_before_holdout.json") == implementation["selection_file_sha256"],
        "source_manifest_still_matches_pre_holdout_hash": file_hash(root/"source_manifest.json") == implementation["source_manifest_sha256"] == selection["source_manifest_sha256"],
        "protocol_digest_matches": digest(protocol) == selection["protocol_sha256"] == holdout_wrapper["protocol_sha256"],
        "selection_content_digest_matches": digest({k: v for k, v in selection.items() if k != "selection_sha256"}) == selection["selection_sha256"],
        "normalized_source_digest_matches_selection": digest(validate_rows(normalized_source)) == selection["source_digest"],
        "validation_report_digest_matches_selection": digest(validation) == selection["validation_report_sha256"],
        "holdout_selection_matches_saved_selection": holdout_wrapper["selection"] == selection,
        "selected_model_chosen_by_validation_log_loss": chosen == min(
            (r["exact_score_log_loss"], name) for name, r in validation["single_match_metrics"].items()
            if r["evaluated"] and r["exact_score_log_loss"] is not None)[1]}
    if not all(integrity.values()):
        raise ValueError("frozen_experiment_integrity_check_failed")
    validate_day_records(validation)
    validate_day_records(holdout)
    validate_source_results(validation, normalized_source)
    validate_source_results(holdout, normalized_source)
    provenance, scheduled_dates, incomplete_dates = source_audit(manifest, cache_dir, protocol["holdout_season"])
    results = {}
    for phase, season in (("validation", validation), ("holdout", holdout)):
        days = season["date_records"]
        val_issued = [d["models"][chosen] for d in validation["date_records"] if d["models"].get(chosen)]
        constant = sum(r["hit"] for r in val_issued)/len(val_issued) if val_issued else None
        declared = {name: [_summary(days, name, threshold,
                                   constant_probability=constant if phase == "holdout" and name == chosen else None)
                           for threshold in protocol["thresholds"]] for name in protocol["models"]}
        # Recompute every saved threshold result independently from its rows.
        for name, summaries in declared.items():
            for saved, fresh in zip(season["five_double_thresholds"][name], summaries):
                if (saved["opportunities"], saved["issued"], saved["hits"], saved["abstained"]) != (
                        fresh["source_date_opportunities"], fresh["issued_sets"], fresh["hits"], fresh["abstained_dates"]):
                    raise ValueError("saved_threshold_denominator_mismatch")
        results[phase] = {"season": season["season"], "source_played_matches": season["source_matches"],
                          "common_fixture_evaluated": season["common_fixture_evaluated"],
                          "single_fixture_exclusions": season["excluded"],
                          "source_date_opportunities": len(days), "model_threshold_calibration": declared,
                          "paired_policy_comparisons": policy_comparisons(season)}
    held_days = holdout["date_records"]
    held_dates = {d["source_date"] for d in held_days}
    complete_days = [d for d in held_days if d["source_date"] not in incomplete_dates]
    live = load(repository_root/"data/tracking/status.json")
    pro_path = repository_root/"data/pro_analyst_shadow/status.json"
    return {"schema_version": 1, "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
            "analysis_type": "post_holdout_read_only_diagnostics_no_model_refit",
            "integrity": integrity,
            "integrity_limits": ["The timestamp order is attested by preserved local artifacts, not an external preregistration.",
                                 "Only the benchmark module was hashed before holdout; dependency hashes were not fully frozen.",
                                 "This verifies saved artifacts and raw source bytes, not an independent original result provider."],
            "selected_model": chosen, "license_declared_by_source_manifest": manifest["license"],
            "license_url": manifest["license_url"], "source_revision": manifest["source_revision"],
            "source_completeness": {**provenance,
                "missing_only_holdout_dates_absent_from_original_opportunity_denominator": sorted(scheduled_dates-held_dates),
                "heldout_record_dates_affected_by_partial_missing_slate": sorted(held_dates & incomplete_dates),
                "complete_source_date_subset_descriptive_only": _summary(complete_days, chosen),
                "complete_case_selection_bias": "Original pair selection omitted fixtures without saved outcomes. Missing-only dates disappear; missing fixtures on partly complete dates could change the chosen pair. Direction and magnitude cannot be established from this archive."},
            "phases": results,
            "live_comparison": {"public_tracking_checked_at_utc": live.get("checked_at_utc"),
                "original_five_doubles": live.get("five_doubles"),
                "v25_status": load(pro_path) if pro_path.is_file() else None,
                "comparable_head_to_head_with_target70": False,
                "reason": "Different models, seasons, leagues, fixture slates, calendar-day conventions and prospective/retrospective designs; do not pool hit counts or infer live V8 improvement."},
            "roi": {"available": False, "estimate": None,
                "reason": "No archived executable correct-score double prices, actual stakes, commissions, voids or returns are present; hit rate alone cannot establish betting profit."},
            "next_data_priorities": [
                "Complete originally scheduled fixture slates and independently checked final scores with provider identities and exact UTC kickoff times; retain missing, postponed and void records.",
                "Immutable source-timestamped pre-match bookmaker correct-score, winner and totals quotes with available prices and settlement rules; necessary for price comparison and ROI.",
                "Licensed historical shot-based xG/xGA and player/confirmed-XI/absence packets timestamped before each forecast; test incremental value on future matched fixtures."],
            "target_70_percent_established": False, "model_promoted": False,
            "analysis_caveats": [
                "Reliability bins and policy comparisons were added after holdout; they are diagnostics, not a newly untouched validation set.",
                "Wilson and paired exact tests assume suitable independent observations; repeated teams, dates and league conditions can cause dependence.",
                "Different pair policies may predict different events; binary Brier comparisons are descriptive and not common-event probability comparisons.",
                "No recalibration, threshold tuning or automatic promotion is justified by these diagnostics.",
                "Abstentions have no probability forecast and are excluded from Brier while remaining in the date-opportunity counts.",
                "Five exact-score double set accuracy differs from single exact-score, top-five single-score, 1X2 and double-chance accuracy."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--cache-dir", type=Path, default=CACHE)
    parser.add_argument("--output", type=Path, default=ROOT/"evaluation_audit.json")
    args = parser.parse_args()
    report = build_report(args.root, args.cache_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, sort_keys=True, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "selected_model": report["selected_model"],
                      "integrity": report["integrity"], "roi": report["roi"]}, indent=2))


if __name__ == "__main__":
    main()
