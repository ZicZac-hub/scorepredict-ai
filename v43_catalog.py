"""Read-only lineage, data coverage and comparable research for V43."""
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path

from five_double_accuracy import utc
from verified_result_index import index_verified_results

ROOT = Path(__file__).resolve().parent

# Versions are project releases. A version number in a provider, baseline or
# later study name is not evidence that the matching project release existed.
VERSIONS = (
    (6, "Venue-form baseline", "Forecast", "engine.py", "forecast_model.py"),
    (7, "Reference-informed goals and shared evidence", "Forecast", "reference_model.py", "forecast_model.py", "EVIDENCE_LEDGER.md"),
    (8, "Evidence ensemble and score-pair ranking", "Forecast and selection", "research_ensemble_v8.py", "smart_double_v8.py", "RESEARCH_V8.md"),
    (9, "Market-informed score grid", "Forecast", "market_score_v9.py", "MARKET_MODEL_V9.md"),
    (10, "Multi-bookmaker consensus and prospective ledger", "Forecast and collection", "multi_market_v10.py", "v10_prospective.py", "V10_UPGRADE.md"),
    (11, "Near-kickoff score refresh and new doubles", "Capture timing", "V11_FRESH_SCORES_AND_DOUBLES.md", "v10_prospective.py"),
    (12, "Score-shape and league calibration", "Calibration", "score_calibration_v12.py", "V12_SCORE_CALIBRATION.md"),
    (13, "Hybrid score grids and five doubles", "Forecast and selection", "hybrid_five_v13.py", "V13_HYBRID_FIVE_DOUBLES.md"),
    (14, "V7/V8 robust agreement and independent pairs", "Selection and evaluation", "robust_double_v14.py", "v14_robust_backtest.py", "V14_FIVE_DOUBLE_ACCURACY_RESEARCH.md"),
    (15, "External score forecasts and source audit", "Research and evaluation", "v15_real_source_audit.py", "V15_SIX_AI_DOUBLE_ACCURACY_AUDIT.md"),
    (16, "Source-bound original-pair AI research", "Research and evaluation", "v16_pair_research.py", "V16_PAIRED_AI_EVIDENCE.md"),
    (17, "Verified confidence bins and joint coverage", "Calibration and evaluation", "v17_confidence.py", "V17_EVIDENCE_CONFIDENCE_RESEARCH.md"),
    (18, "Prospective accuracy and time-safe calibration", "Calibration and evaluation", "v18_accuracy_audit.py", "V18_REAL_ACCURACY_EVIDENCE.md"),
    (19, "Source-verified match and market intelligence", "Research and collection", "v19_match_intelligence.py", "V19_VERIFIED_MATCH_INTELLIGENCE.md"),
    (20, "Home/away goal form, head-to-head and venue form", "Forecast", "goal_research_v20.py", "tests/test_goal_research_v20.py"),
    (21, "Verified provider store and chronological model replay", "Collection and evaluation", "v21_capture_sources.py", "v21_goal_model_comparison.py", "V21_RESEARCH_PIPELINE.md"),
    (22, "Market and xG models with paired AI evaluation", "Forecast and evaluation", "v22_evidence_models.py", "v22_ai_gate.py", "v22_prospective.py"),
    (23, "Near-kickoff market and goal-form challengers", "Forecast and collection", "v23_market_form.py", "v23_prospective.py", "V23_MARKET_GOAL_RESEARCH.md"),
    (24, "Date-wide market catalog", "Collection", "v24_bulk_market_capture.py", "V24_BROAD_DATA_CAPTURE.md"),
    (25, "Upcoming slate and anchored score challenger", "Forecast and selection", "v25_upcoming.py", "v25_anchored_challenger.py", "V25_ANCHORED_SCORE_RESEARCH.md"),
    (26, "Chronological score calibration", "Calibration", "v26_temporal_score.py", "V26_CHRONOLOGICAL_CALIBRATION.md"),
    (27, "Full-slate pair quality gate", "Selection", "v27_pair_selection.py", "V27_FULL_SLATE_PAIR_RESEARCH.md"),
    (28, "Independent prospective pair trial", "Selection and evaluation", "v28_independent_pair_trial.py", "V28_INDEPENDENT_PAIR_RESEARCH.md"),
    (29, "Distinct pair sensitivity and result readiness", "Evaluation", "v29_pair_sensitivity.py", "V29_PAIR_SENSITIVITY_AND_READINESS.md"),
    (30, "Prospective full-slate rank audit", "Selection and evaluation", "v30_full_slate_audit.py", "V30_FULL_SLATE_RANKING_RESEARCH.md"),
    (31, "Verified result coverage for original slates", "Collection and evaluation", "v31_slate_settlement.py", "V31_ORIGINAL_SLATE_RESULT_COVERAGE.md"),
    (32, "Rank discrimination and lift", "Evaluation", "v32_rank_discrimination.py", "V32_RANK_DISCRIMINATION_RESEARCH.md"),
    (33, "Early settlement rank checkpoints", "Evaluation", "v33_early_rank_checkpoints.py", "V33_EARLY_SETTLED_RANK_CHECKPOINTS.md"),
    (34, "Exact-score error and goal-mass decomposition", "Evaluation", "v34_score_error_research.py", "V34_SCORE_ERROR_RESEARCH.md"),
    (35, "Temperature and goal-tilt score shapes", "Forecast and evaluation", "v35_continuous_double_trials.py"),
    (36, "Tail calibration and fixed-set trials", "Calibration and evaluation", "v36_tail_calibration_research.py", "v36_prospective_double_trials.py"),
    (37, "Sparse-history UEFA challenger", "Forecast and evaluation", "v37_sparse_history_research.py", "v37_sparse_evaluation.py"),
    (38, "Conditional score-margin calibration", "Calibration and evaluation", "v38_margin_calibration.py", "v38_prospective_double_trials.py"),
    (39, "Poisson marginal challenger", "Forecast and evaluation", "v39_poisson_marginal.py", "v39_prospective_double_trials.py"),
    (40, "Paired Poisson and shot-xG ablation", "Forecast and evaluation", "v40_poisson_xg_ablation.py"),
    (41, "Fixed-weight grouped ensemble and weight audit", "Forecast and evaluation", "v41_unified_ensemble.py", "v41_unified_trial.py", "v41_weight_audit.py"),
    (42, "V41 control, shape challenges, and matched cutoff trials", "Forecast and evaluation", "v42_research_engine.py", "v42_research_trial.py", "v42_cutoff_comparison.py", "V42_RESEARCH.md"),
)

LEGACY_LINEAGE = (
    ("Original worldwide Poisson score forecasts and walk-forward testing", "57337a6fe", ("engine.py", "advanced.py", "app.py")),
    ("Venue form, head-to-head and a dual-method goals ensemble", "4f2ea1ad7", ("engine.py",)),
    ("League shrinkage, preserved baseline and calendar recency", "aa7de4bc1", ("engine.py",)),
    ("Dixon–Coles low-score dependence and venue-aware estimates", "45941c267", ("engine.py",)),
    ("Opponent-adjusted attack and defence experiments", "c50b67ae3", ("engine.py",)),
)

SOURCES = (
    ("football-data.org", "Fixtures and completed competition history", "Live adapter", "data.py", "history_pipeline.py"),
    ("API-Football", "Fixtures, history, verified finals, odds, lineup, injury and data-plan-dependent inputs", "Live adapter and archives", "api_football.py", "data/prematch", "data/v21_sources", "data/v24_market_catalog"),
    ("Sportmonks", "Fixture discovery, optional licensed shot xG and correct-score probability research", "Live adapters and archives", "sportmonks.py", "sportmonks_xg_research.py", "sportmonks_probability_research.py", "data/sportmonks_xg", "data/independent_football"),
    ("OpenFootball", "Pinned CC0 completed scores, league and season coverage for historical replay", "Historical archive", "target70_data.py", "data/target70_research/source_manifest.json"),
    ("OpenAI and Claude", "Optional analysis of the same fixture evidence and statistical forecast", "Opt-in research adapters", "llm_research_panel.py", "on_demand_ai.py", "data/llm_research"),
    ("Additional AI providers", "Gemini, Grok, DeepSeek, Perplexity, Mistral, Qwen, GroqCloud and Cohere", "Opt-in research adapters", "llm_research_panel.py", "TEN_AI_AND_SPORTMONKS_SETUP.md"),
    ("Verified result stores", "Identity-matched regulation scores used to evaluate frozen forecasts", "Archived", "verified_result_index.py", "data/prematch/verified_results.json", "data/v10_prospective/verified_results.json"),
    ("Central forecast ledger", "PostgreSQL archive and file-backed prospective evidence", "Storage adapters and archives", "central_archive.py", "forecast_ledger.py", "data/prospective_ledger"),
)

METHODS = (
    ("Poisson and league baselines", "Goal rates, venue form, competition priors and opponent strength", "engine.py"),
    ("Dixon–Coles", "Low-score dependence with optional time decay", "dynamic_dixon_coles.py"),
    ("Bivariate Poisson and negative binomial", "Shared scoring dependence and alternative dispersion", "v21_goal_model_comparison.py"),
    ("Market calibration", "Verified bookmaker prices, overround removal and bounded adjustment", "v22_evidence_models.py"),
    ("Shot xG ablation", "Qualified paired historical shot xG versus same-cohort controls", "v40_poisson_xg_ablation.py"),
    ("Temperature, tail, margin and upset sensitivity", "Score-grid shape hypotheses and fixed-family mixtures", "v42_research_engine.py"),
    ("Five exact-score doubles", "Five distinct joint score pairs, using products of individual fixture probabilities", "professional_double_optimizer.py"),
    ("Historical tournament", "Same-evidence walk-forward controls with common-cohort score metrics", "all_methods_tournament.py"),
)


def _object(value):
    return value if isinstance(value, dict) else {}


def _report(root, artifact):
    try:
        value = json.loads((root / artifact).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return _object(value)


def _archive_coverage(root):
    index = _report(root, "data/prediction_evidence_index.json").get("paths")
    indexed = Counter()
    if isinstance(index, list):
        for raw in set(path for path in index if isinstance(path, str)):
            parts = Path(raw).parts
            if len(parts) >= 3 and parts[0] == "data" and ".." not in parts:
                indexed[parts[1]] += 1
    data = root / "data"
    if not data.is_dir():
        return []
    return [
        {"archive": directory.name, "artifact": directory.relative_to(root).as_posix(),
         "files": sum(name.endswith((".json", ".json.gz")) for _, _, files in os.walk(directory) for name in files),
         "indexed_files": indexed[directory.name]}
        for directory in sorted(data.iterdir()) if directory.is_dir() and not directory.is_symlink()
    ]


def _evidence_results(root, as_of):
    results, conflicts, rejected = index_verified_results(
        root / "data/v10_prospective", root / "data/prematch", as_of
    )
    dates = sorted(utc(row["kickoff_utc"]) for row in results.values())
    return {
        "verified_fixtures": len(results),
        "verified_result_conflicts": len(conflicts),
        "verified_result_rejections": dict(rejected),
        "verified_fixture_start_utc": dates[0].isoformat() if dates else None,
        "verified_fixture_end_utc": dates[-1].isoformat() if dates else None,
        "verified_fixtures_by_provider": dict(sorted(Counter(provider for provider, _ in results).items())),
    }


def _benchmarks(root):
    benchmarks, metrics = [], []

    def add(artifact, benchmark, kind, unit, report, matched, source_matches=None, notes=None):
        if report:
            benchmarks.append({
                "benchmark": benchmark, "kind": kind, "matched": matched, "unit": unit,
                "source_matches": source_matches,
                "as_of": report.get("evaluated_at_utc") or report.get("generated_date"),
                "artifact": artifact, "notes": notes,
            })

    def scores(benchmark, arms, n, unit="fixtures"):
        if type(n) is not int or n < 1:
            return
        for method, row in arms.items():
            if not isinstance(row, dict):
                continue
            # Only rows with the common denominator belong in a cohort table.
            count = row.get("evaluated", row.get("verified", row.get("verified_sets", row.get("fixtures", n))))
            if count != n:
                continue
            metrics.append({
                "benchmark": benchmark, "method": method, "matched": n, "unit": unit,
                "hits": row.get("exact_hits", row.get("hits")),
                "hit_rate": row.get("exact_hit_rate", row.get("exact_accuracy", row.get("exact_hit", row.get("hit_rate")))),
                "score_log_loss": row.get("score_log_loss", row.get("single_score_log_loss", row.get("exact_score_log_loss"))),
                "outcome_brier": row.get("outcome_brier"),
                "multiclass_score_brier": row.get("multiclass_brier"),
            })

    artifact = "data/research_v8/extended_historical_validation.json"
    report = _report(root, artifact)
    name = "V6 versus V8 expanded replay"
    n = report.get("matched")
    add(artifact, name, "Historical replay", "fixtures", report, n, report.get("source_results"), report.get("method"))
    scores(name, {key: report.get(key) for key in ("preserved_v6", "research_v8")}, n)

    artifact = "data/accuracy_diagnosis/historical_replay.json"
    report = _report(root, artifact)
    name = "Chronological goal-distribution replay"
    n = report.get("matched_fixtures")
    add(artifact, name, "Historical replay", "fixtures", report, n, report.get("archived_verified_results"), report.get("note"))
    scores(name, _object(report.get("single_match_metrics")), n)

    paired = _object(report.get("paired_v8_v20_same_fixtures"))
    name = "V8 versus V20 venue goal form"
    n = paired.get("common_fixtures")
    add(artifact, name, "Matched historical replay", "fixtures", paired, n,
        report.get("archived_verified_results"), "Identical replay fixtures for both score grids.")
    scores(name, {"V8": paired.get("original_v8_on_eligible_fixtures"), "V20": paired.get("v20_form_on_same_fixtures")}, n)

    artifact = "data/target70_research/summary.json"
    report = _report(root, artifact)
    source = _object(report.get("source_dataset"))
    holdout = _object(report.get("holdout_summary"))
    name = "OpenFootball season holdout"
    n = holdout.get("common_fixture_evaluated")
    add(artifact, name, "Historical holdout", "fixtures", report, n, source.get("normalized_played_matches"),
        "Five leagues and six seasons of date-only source results. Entire candidate date excluded from history.")
    scores(name, _object(holdout.get("single_match_metrics")), n)

    artifact = "data/research_model_comparison_v1/summary.json"
    report = _report(root, artifact)
    name = "Expanded model comparison on the OpenFootball cohort"
    arms = _object(report.get("development_single_match_metrics"))
    counts = {row.get("evaluated") for row in arms.values() if isinstance(row, dict) and type(row.get("evaluated")) is int}
    n = next(iter(counts)) if len(counts) == 1 else None
    add(artifact, name, "Development analysis", "fixtures", report, n, report.get("source_matches"),
        "Reuses the previously explored OpenFootball development cohort. Market, shot xG and AI were unavailable.")
    scores(name, arms, n)

    artifact = "data/benchmark/summary.json"
    report = _report(root, artifact)
    name = "Independent frozen score benchmark"
    paired = _object(report.get("paired"))
    n = paired.get("shared_fixtures")
    add(artifact, name, "Prospective", "fixtures", report, n,
        notes="Common verified original forecasts. Multiclass score Brier scores concern the score grid.")
    scores(name, _object(paired.get("models")), n)

    artifact = "data/accuracy_diagnosis/v8_v25_matched.json"
    report = _report(root, artifact)
    name = "V8 versus V25 frozen pairs"
    n = report.get("verified_matched_pairs")
    add(artifact, name, "Prospective", "pairs", report, n, notes="Same original fixtures and regulation results for both methods.")
    scores(name, {key: {"hits": row.get("five_double_hits"), "hit_rate": row.get("five_double_hit_rate")}
                  for key in ("v8", "v25") if isinstance(row := report.get(key), dict)}, n, "pairs")

    artifact = "data/v42_research/performance.json"
    report = _report(root, artifact)
    name = "V8 versus V42 final arm"
    paired = _object(_object(report.get("paired_comparisons")).get("unified_final"))
    n = paired.get("verified_common_pairs")
    add(artifact, name, "Prospective", "pairs", report, n,
        notes="Common verified pairs between the production baseline and the V42 research arm.")
    scores(name, _object(paired.get("same_pair_metrics")), n, "pairs")

    artifact = "data/v42_late_research/cutoff_comparison.json"
    report = _report(root, artifact)
    name = "V42 early versus late common-cutoff cohorts"
    n = report.get("matched_verified_issued_pairs")
    add(artifact, name, "Prospective", "pairs", report, n,
        notes="Identical verified pairs issued at both cutoffs. Each run uses evidence available at its own cutoff.")
    scores(name, {key: report.get(key) for key in ("matched_early", "matched_late")}, n, "pairs")

    return benchmarks, metrics, {
        "openfootball_matches": source.get("normalized_played_matches"),
        "openfootball_seasons": _object(source.get("coverage_by_season")),
        "openfootball_competitions": _object(source.get("coverage_by_competition")),
    }


def build_catalog(root=None, *, as_of=None):
    """Combine archive coverage and studies. The cutoff applies to verified finals."""
    root = ROOT if root is None else Path(root)
    now = datetime.now(timezone.utc) if as_of is None else utc(as_of)
    versions = [
        {"version": version, "title": "Legacy release label unavailable", "role": "Legacy lineage",
         "status": "Unverified release label", "artifacts": []}
        for version in range(1, 6)
    ]
    for version, title, role, *files in VERSIONS:
        artifacts = [path for path in files if (root / path).exists()]
        versions.append({"version": version, "title": title, "role": role,
                         "status": "Documented" if artifacts else "Artifact unavailable", "artifacts": artifacts})
    sources = []
    for title, role, status, *files in SOURCES:
        artifacts = [path for path in files if (root / path).exists()]
        sources.append({"source": title, "role": role, "status": status if artifacts else "Artifact unavailable", "artifacts": artifacts})
    methods = [
        {"method": title, "role": role, "artifact": artifact}
        for title, role, artifact in METHODS if (root / artifact).exists()
    ]
    archives = _archive_coverage(root)
    benchmarks, metrics, historical = _benchmarks(root)
    return {
        "versions": versions,
        "legacy_lineage": [
            {"title": title, "commit": commit, "artifacts": list(artifacts)}
            for title, commit, artifacts in LEGACY_LINEAGE
        ],
        "sources": sources,
        "methods": methods,
        "archives": archives,
        "benchmarks": benchmarks,
        "metrics": metrics,
        "summary": {
            "documented_versions": sum(row["status"] == "Documented" for row in versions),
            "unavailable_version_labels": [row["version"] for row in versions if row["status"] == "Unverified release label"],
            "archive_families": len(archives),
            "archive_files": sum(row["files"] for row in archives),
            "indexed_files": sum(row["indexed_files"] for row in archives),
            **historical,
            **_evidence_results(root, now),
        },
    }
