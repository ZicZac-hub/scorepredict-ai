"""Measure one football data input at a time on original match archives."""

from collections import Counter
from datetime import datetime, timedelta, timezone
import json
import math
from pathlib import Path

from five_double_accuracy import _final, wilson
from v10_prospective import utc
import v42_late_research_trial as late_study
import v42_research_trial as early_study
from v43_prediction import _combine, _source_bundle
from v44_family_pool_research import _grid
import v44_fixture_evidence as inventory
from v44_late_evidence_audit import LEAD, _candidate_models
import v44_market_ablation as market_archive
from v44_single_calibration import _metrics
from v44_v9_reliability import checked_projection
from v44_xg_age_research import _find as find_xg
from verified_result_index import index_verified_results


SOURCE_KEYS = {"market": "market", "historical_xg": "xg"}
REVIEW_FIXTURES = 30
REVIEW_PAIRS = early_study.PROTOCOL["minimum_review_verified_pairs"]
ARMS = ("baseline", "with_source")
BAD_RECORD = (ValueError, TypeError, KeyError, OSError, AttributeError, ArithmeticError)


def _comparison(metrics, counts, *, pairs=False):
    minimum = REVIEW_PAIRS if pairs else REVIEW_FIXTURES
    hit_key = "hit" if pairs else "top_score_hit"
    loss_key = "score_log_loss" if pairs else "exact_log_loss"
    loss_fields = (loss_key, "outcome_brier") if pairs else (loss_key, "outcome_brier", "score_brier")
    n = len(metrics)
    if n >= minimum:
        status = "descriptive_replay_available"
    elif n:
        status = "insufficient_matched_results"
    elif counts["qualified_inputs"]:
        status = "no_scoreable_matches"
    elif counts["opportunities"]:
        status = "no_qualified_source_evidence"
    else:
        status = "no_eligible_archives"
    models = {}
    for arm in ARMS:
        rows = [row[arm] for row in metrics]
        hits = sum(r[hit_key] for r in rows)
        models[arm] = {"hits": hits, "hit_rate": hits / n if n else None,
                       "wilson_95": wilson(hits, n),
                       **{key: math.fsum(r[key] for r in rows) / n if n else None
                          for key in loss_fields}}
    loss_differences = [r["with_source"][loss_key] - r["baseline"][loss_key] for r in metrics]
    better = sum(delta < -1e-12 for delta in loss_differences)
    worse = sum(delta > 1e-12 for delta in loss_differences)
    return {"status": status, "opportunities": counts["opportunities"],
            "qualified_inputs": counts["qualified_inputs"],
            "missing_source_evidence": counts["opportunities"] - counts["qualified_inputs"],
            "missing_verified_final": counts["missing_verified_final"],
            "transform_abstentions": counts["transform_abstentions"],
            "matched": n, "minimum_descriptive_sample": minimum, "models": models,
            "paired_delta_with_source_minus_baseline": {
                key: math.fsum(r["with_source"][key] - r["baseline"][key] for r in metrics) / n
                if n else None for key in (*loss_fields, hit_key)},
            "lower_exact_score_loss": {"with_source": better, "baseline": worse,
                                       "ties": n - better - worse}}


def _record(models, bundles, actual, comparisons):
    """Keep the score engine fixed and toggle only the tested data component."""
    pair = len(models) == 2
    grids = [model["score_matrix"] for model in models]
    baseline = None
    for source, kind in SOURCE_KEYS.items():
        metrics, counts = comparisons[source]
        counts["opportunities"] += 1
        if any(bundle.get(kind) is None for bundle in bundles):
            continue
        counts["qualified_inputs"] += 1
        if any(score is None for score in actual):
            counts["missing_verified_final"] += 1
            continue
        if baseline is None:
            baseline = _combine(grids, [{} for _ in grids])
        with_source = _combine(grids, [{kind: bundle[kind]} for bundle in bundles])
        if (baseline["status"] != "issued" or with_source["status"] != "issued"
                or with_source["lineage"]["evidence_used"].get(kind) is not True):
            counts["transform_abstentions"] += 1
            continue
        arms = dict(zip(ARMS, (baseline, with_source)))
        if pair:
            metrics.append({name: early_study._metric(arm, actual) for name, arm in arms.items()})
        else:
            metrics.append({name: _metrics(_grid(arm["full_score_grids"][0]), actual[0])
                            for name, arm in arms.items()})


def _empty_comparisons():
    return {source: ([], Counter()) for source in SOURCE_KEYS}


def _singles(archive, *, source_root, provider_root, finals, clock):
    comparisons = _empty_comparisons()
    rejections = Counter()
    for row in archive["rows"]:
        cutoff = utc(row["kickoff_utc"]) - LEAD
        models, rejected = _candidate_models(row, source_root, cutoff)
        rejections["invalid_forecast_records"] += rejected
        original = next((m for m in models if m["forecast_id"] == row["forecast_id"]), None)
        if original is None:
            rejections["missing_frozen_baseline_at_cutoff"] += 1
            continue
        identity = (original["home_team_id"], original["away_team_id"])
        models = [m for m in models if (m["home_team_id"], m["away_team_id"]) == identity]
        base = max(models, key=lambda m: (utc(m["captured_at_utc"]), m["forecast_id"]))
        try:
            bundle = _source_bundle(models, cutoff, provider_root)
            _, extended_xg, _ = find_xg(models, provider_root, cutoff)
            # Historical xG can use the validated 24h window. Our market and
            # lineup receipts retain the original three-hour snapshot rule.
            bundle["xg"] = ({"reference": extended_xg[1], "value": extended_xg[2]}
                            if extended_xg else None)
            actual = _final(base, finals.get(inventory._identity(base)), clock)
            _record([base], [bundle], [actual], comparisons)
        except BAD_RECORD:
            rejections["unusable_source_or_comparison"] += 1
    return comparisons, {k: v for k, v in rejections.items() if v}


def _doubles(protocol, root, *, pair_root, source_root, provider_root, finals, clock):
    comparisons = _empty_comparisons()
    rejections = Counter()
    ids, events = set(), set()
    files = sorted((Path(root) / "trials").glob("*.json"))
    for path in files:
        try:
            record = protocol._load(path)
            if record["sydney_date"] != path.stem:
                raise ValueError("study_filename_day_mismatch")
            models = protocol._validated(record, root=root, pair_root=pair_root, source_root=source_root,
                                         provider_root=provider_root, now=clock)
            current_ids = {inventory._identity(model) for model in models}
            current_events = {inventory._event(model) for model in models}
            if len(current_ids) != 2 or len(current_events) != 2 or ids & current_ids or events & current_events:
                raise ValueError("repeated_fixture")
            decision = utc(record["decision_as_of_utc"])
            bundles = []
            for model in models:
                bundle = protocol._components(model, provider_root=provider_root, cutoff=decision)
                if bundle["xg"] is not None:
                    # Require typed historical xG from the exact receipt the
                    # original protocol selected. Keep its stricter 3h window.
                    strict, _, _ = find_xg([model], provider_root, decision)
                    reference = bundle["xg"]["reference"]
                    if strict is None or any(strict[1].get(k) != v for k, v in reference.items()):
                        bundle["xg"] = None
                bundles.append(bundle)
            ids.update(current_ids)
            events.update(current_events)
            actual = [_final(m, finals.get(inventory._identity(m)), clock) for m in models]
            _record(models, bundles, actual, comparisons)
        except BAD_RECORD as exc:
            rejections[str(exc) if isinstance(exc, ValueError) else type(exc).__name__] += 1
    return comparisons, {"archive_files": len(files), "rejections": dict(rejections)}


def _archived_market(*, source_root, finals, clock):
    """Keep older V9 quote receipts in a distinct, fixed projection cohort."""
    matches, counts, rejections = market_archive.match_archive(source_root, clock)
    metrics = []
    counts["opportunities"] = counts["qualified_inputs"] = len(matches)
    for baseline, market in matches:
        final = finals.get(inventory._identity(baseline))
        actual = _final(baseline, final, clock)
        if actual is None or _final(market, final, clock) != actual:
            counts["missing_verified_final"] += 1
            continue
        try:
            base = _grid(baseline["score_matrix"])
            enhanced, _ = checked_projection(base, market_archive.qualify_market(market))
            metrics.append({"baseline": _metrics(base, actual), "with_source": _metrics(enhanced, actual)})
        except BAD_RECORD:
            counts["transform_abstentions"] += 1
    return {**_comparison(metrics, counts), "archive_counts": dict(counts),
            "archive_rejections": dict(rejections)}


def evaluate(*, source_root=early_study.SOURCE_ROOT, prematch_root=early_study.PREMATCH_ROOT,
             provider_root=early_study.PROVIDER_ROOT, early_root=early_study.ROOT,
             late_root=late_study.ROOT, pair_root=early_study.PAIR_ROOT, now=None):
    clock = utc(now or datetime.now(timezone.utc))
    finals, conflicts, rejected = index_verified_results(source_root, prematch_root, clock)
    archive = inventory.evaluate(source_root=source_root, prematch_root=prematch_root,
                                 provider_root=provider_root, now=clock)
    singles, single_rejections = _singles(archive, source_root=source_root, provider_root=provider_root,
                                          finals=finals, clock=clock)
    context = dict(pair_root=pair_root, source_root=source_root, provider_root=provider_root,
                   finals=finals, clock=clock)
    early, early_diagnostics = _doubles(early_study, early_root, **context)
    late, late_diagnostics = _doubles(late_study, late_root, **context)
    comparisons = {source: {"single_fixtures": _comparison(*singles[source]),
                            "early_five_doubles": _comparison(*early[source], pairs=True),
                            "late_five_doubles": _comparison(*late[source], pairs=True)}
                   for source in SOURCE_KEYS}
    comparisons["market"]["archived_v9_fixtures"] = _archived_market(
        source_root=source_root, finals=finals, clock=clock)
    return {"study_type": "v45-matched-source-impact-replay-v1", "evaluated_at_utc": clock.isoformat(),
            "comparisons": comparisons,
            "cohorts": {"single_fixtures": {"cutoff_minutes_before_kickoff": LEAD.total_seconds() / 60,
                                           "inventory_verified_matches": archive["matched_verified_fixtures"],
                                           "inventory_upcoming": archive["pending_kickoff"],
                                           "inventory_past_without_verified_final": archive["past_without_verified_final"],
                                           "rejections": single_rejections},
                        "early_five_doubles": early_diagnostics, "late_five_doubles": late_diagnostics},
            "context_only": {"confirmed_lineups": "no_validated_numerical_lineup_model",
                             "ai_reviews": "no_numerical_forecast_weight",
                             "recovered_results_history": "no_matched_same_cutoff_history_only_arms"},
            "result_conflicts": len(conflicts), "result_rejections": dict(rejected),
            "external_requests": 0, "automatic_promotion": False,
            "limitations": ["The source is toggled independently on the same archived probability grid.",
                            "Original early and late pair cohorts can overlap and are reported separately.",
                            "Settlement is verified separately from the forecasts. The legacy result receipt does not prove agreement between providers.",
                            "Only input-qualified and result-verified matches contribute to each comparison.",
                            "The archived V9 comparison uses its own older quote receipts and fixed projection.",
                            "These are retrospective comparisons. Prospective forecast improvement requires separate pre-kickoff trials."]}


if __name__ == "__main__":
    print(json.dumps(evaluate(), indent=2, allow_nan=False))
