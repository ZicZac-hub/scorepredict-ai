"""New, research-only comparison of score models on source-date histories.

The November 15 target70 experiment is immutable. This is a separate version:
2024-25 has already been explored and is explicitly DEVELOPMENT retrospective
data, not an independent unseen test. No fitted model is promoted to live use.
"""
import argparse
from collections import Counter, defaultdict
from datetime import date, timedelta
import hashlib
import importlib.util
import json
import math
from pathlib import Path

import numpy as np

import target70_benchmark as old
from engine import _grid


CORE_MODELS = ("league_poisson", "time_weighted_strength_poisson",
               "dixon_coles_strength", "dixon_coles_regularized_r18",
               "bivariate_poisson_c012", "negative_binomial_r3")
BOOST = "gradient_boost_poisson"
ENSEMBLE = "validation_fitted_calibrated_ensemble"
PROTOCOL = {
    "version": "development-score-model-comparison-v1",
    "preserved_original_experiment": "target70-cc0-date-safe-v1",
    "development_test_season_already_explored": "2024-25",
    "independent_unseen_test": False,
    "training_label_source_dates_before": "2023-07-01",
    "boost_training_source_dates_from": "2021-07-01",
    "weight_fit_selection_season": "2023-24",
    "weight_fit_source_dates_before": "2024-01-01",
    "temperature_calibration_source_dates_from": "2024-01-01",
    "temperature_candidates": [.8, 1., 1.2],
    "model_selection_metric": "common_fixture_exact_score_log_loss",
    "core_models": list(CORE_MODELS),
    "optional_gradient_boost": {
        "estimator": "sklearn.HistGradientBoostingRegressor",
        "loss": "poisson", "learning_rate": .05, "max_iter": 100,
        "max_leaf_nodes": 7, "min_samples_leaf": 30,
        "l2_regularization": 10., "random_state": 17, "early_stopping": False},
    "ensemble_weight_fit": "positive_simplex_exponentiated_gradient_log_loss_uniform_ridge_0.01",
    "minimum_weight_fit_rows": 50,
    "minimum_calibration_rows": 50,
    "monthly_training_boundary": "strictly_before_month_start_minus_one_source_day",
    "pair_candidate_cap_per_source_date": 18,
    "thresholds": [0., .1, .2, .3, .5, .7],
    "pair_policy": "highest_five_joint_score_mass_with_four_distinct_teams",
    "time_precision": "source_dates_only_not_UTC_kickoffs_or_Sydney_matchdays",
    "future_prospective_validation_required": True,
    "live_model_promotion": False}
FEATURES = ("strength_home_mean", "strength_away_mean", "home_prior_count", "away_prior_count",
            "home_last10_scored", "home_last10_conceded", "away_last10_scored", "away_last10_conceded",
            "home_venue_scored", "home_venue_conceded", "away_venue_scored", "away_venue_conceded",
            "home_rest_source_days", "away_rest_source_days", "league_home_mean", "league_away_mean",
            "league_en", "league_de", "league_es", "league_it", "league_fr")


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _marginal(rate, *, shape=None):
    if not math.isfinite(rate) or not 0 <= rate <= 20:
        raise ValueError("goal_rate_outside_research_support")
    if shape is not None and (not math.isfinite(shape) or shape <= 0):
        raise ValueError("positive_negative_binomial_shape_required")
    out = [math.exp(-rate) if shape is None else (shape/(shape+rate))**shape]
    while len(out) < 100 and math.fsum(out) < 1-1e-10:
        k = len(out)
        out.append(out[-1]*rate/k if shape is None else
                   out[-1]*(k-1+shape)/k*rate/(shape+rate))
    if math.fsum(out) < 1-1e-8:
        raise ValueError("insufficient_goal_tail_support")
    return np.asarray(out)/math.fsum(out)


def bivariate_grid(home_mean, away_mean, covariance=.12):
    shared = min(covariance, .8*min(home_mean, away_mean))
    h, a, z = (_marginal(x) for x in (home_mean-shared, away_mean-shared, shared))
    size = max(len(h), len(a))+len(z)-1
    result = np.zeros((size, size))
    for k, probability in enumerate(z):
        result[k:k+len(h), k:k+len(a)] += probability*np.outer(h, a)
    return result/result.sum()


def negative_binomial_grid(home_mean, away_mean, shape=3.):
    return np.outer(_marginal(home_mean, shape=shape), _marginal(away_mean, shape=shape))


def _means(row, state):
    h, a = (state["index"][old._team(row, side)] for side in ("home", "away"))
    return (math.exp(state["intercept_home"]+state["attack"][h]+state["defence"][a]),
            math.exp(state["intercept_away"]+state["attack"][a]+state["defence"][h]))


class ResearchContext:
    """Date-safe cached monthly fits; current fixture outcomes are not features."""
    def __init__(self, rows):
        self.rows = old.validate_rows(rows)
        self.by_league = defaultdict(list)
        for row in self.rows:
            self.by_league[row["league"]].append(row)
        self.states = {}

    def state(self, row, config_name):
        month = old._date(row).replace(day=1)
        key = (row["league"], month, config_name)
        if key not in self.states:
            self.states[key] = old.fit_state(self.by_league[row["league"]], row["league"], month,
                                             old.PROTOCOL["models"][config_name])
        return self.states[key]

    def forecast(self, row):
        baseline = self.state(row, "league_poisson")
        strength = self.state(row, "opponent_dc_r6_h240")
        conservative = self.state(row, "opponent_dc_r18_h365")
        if any(old.matrix_for(row, state) is None for state in (baseline, strength, conservative)):
            return None
        h, a = _means(row, strength)
        grids = {"league_poisson": old.matrix_for(row, baseline),
                 "time_weighted_strength_poisson": _grid(h, a),
                 "dixon_coles_strength": old.matrix_for(row, strength),
                 "dixon_coles_regularized_r18": old.matrix_for(row, conservative),
                 "bivariate_poisson_c012": bivariate_grid(h, a),
                 "negative_binomial_r3": negative_binomial_grid(h, a)}
        states = {name: strength for name in grids}
        states["league_poisson"] = baseline
        states["dixon_coles_regularized_r18"] = conservative
        feature, source = self.features(row, strength)
        return {"row": row, "models": grids, "state": states,
                "features": feature, "feature_provenance": source}

    def features(self, row, state):
        boundary = date.fromisoformat(state["strict_training_date_boundary"])
        past = [r for r in self.by_league[row["league"]] if old._date(r) < boundary]
        histories = []
        for team, venue in ((row["home"], "home"), (row["away"], "away")):
            overall, specific = [], []
            for item in reversed(past):
                if team not in (item["home"], item["away"]):
                    continue
                at_home = item["home"] == team
                goals = (item["home_goals"], item["away_goals"]) if at_home else (
                    item["away_goals"], item["home_goals"])
                if len(overall) < 10:
                    overall.append((goals, item["date"]))
                if (at_home == (venue == "home")) and len(specific) < 10:
                    specific.append((goals, item["date"]))
                if len(overall) == len(specific) == 10:
                    break
            if not overall:
                raise ValueError("verified_prior_team_history_required")
            # A missing venue sample shrinks to real observed overall games.
            specific = specific or overall
            average = lambda sample, i: math.fsum(x[0][i] for x in sample)/len(sample)
            histories.append({"overall": [average(overall, i) for i in (0, 1)],
                              "venue": [average(specific, i) for i in (0, 1)],
                              "rest": min(120, (old._date(row)-date.fromisoformat(overall[0][1])).days)})
        h, a = _means(row, state)
        count = [state["counts"][state["index"][old._team(row, side)]] for side in ("home", "away")]
        feature = [h, a, *count, *histories[0]["overall"], *histories[1]["overall"],
                   *histories[0]["venue"], *histories[1]["venue"],
                   histories[0]["rest"], histories[1]["rest"],
                   math.fsum(x["home_goals"] for x in past)/len(past),
                   math.fsum(x["away_goals"] for x in past)/len(past),
                   *(int(row["league"] == league) for league in old.PROTOCOL["leagues"])]
        if len(feature) != len(FEATURES) or not np.isfinite(feature).all():
            raise ValueError("invalid_prior_result_feature_vector")
        return np.asarray(feature, dtype=float), {
            "source_date_precision": "calendar_date_only",
            "strict_training_date_boundary": boundary.isoformat(),
            "latest_feature_result_source_date": max(r["date"] for r in past),
            "current_fixture_goals_used_as_features": False,
            "feature_names": list(FEATURES)}


def train_boost(context, *, enabled=False, estimator_factory=None):
    if not enabled:
        return None, {"ready": False, "reason": "optional_gradient_boost_not_requested"}
    if estimator_factory is None:
        if importlib.util.find_spec("sklearn") is None:
            return None, {"ready": False, "reason": "optional_sklearn_dependency_unavailable"}
        from sklearn.ensemble import HistGradientBoostingRegressor
        estimator_factory = HistGradientBoostingRegressor
    observations = []
    for row in context.rows:
        if not (PROTOCOL["boost_training_source_dates_from"] <= row["date"] <
                PROTOCOL["training_label_source_dates_before"]):
            continue
        forecast = context.forecast(row)
        if forecast:
            observations.append(forecast)
    if len(observations) < 100:
        return None, {"ready": False, "reason": "insufficient_date_safe_training_rows",
                      "training_rows": len(observations)}
    config = {k: v for k, v in PROTOCOL["optional_gradient_boost"].items() if k != "estimator"}
    x = np.vstack([r["features"] for r in observations])
    models = []
    for side in ("home", "away"):
        y = np.asarray([r["row"][side+"_goals"] for r in observations], dtype=float)
        model = estimator_factory(**config)
        model.fit(x, y)
        models.append(model)
    return models, {"ready": True, "training_rows": len(observations),
                    "latest_training_label_source_date": max(r["row"]["date"] for r in observations),
                    "all_features_are_earlier_monthly_results": True,
                    "fixed_estimator_settings": config,
                    "feature_names": list(FEATURES)}


def add_boost(forecast, models):
    if models is None:
        return
    rates = [max(.05, float(model.predict(forecast["features"][None, :])[0])) for model in models]
    forecast["models"][BOOST] = _grid(*rates)
    forecast["state"][BOOST] = forecast["state"]["dixon_coles_strength"]


def fit_ensemble_weights(forecasts, names):
    if len(forecasts) < PROTOCOL["minimum_weight_fit_rows"]:
        raise ValueError("insufficient_ensemble_weight_fit_rows")
    probability = np.asarray([[
        float(r["models"][name][r["row"]["home_goals"], r["row"]["away_goals"]])
        if r["row"]["home_goals"] < r["models"][name].shape[0] and
        r["row"]["away_goals"] < r["models"][name].shape[1] else 1e-15
        for name in names] for r in forecasts])
    weights = np.ones(len(names))/len(names)
    for _ in range(400):
        denominator = np.maximum(probability@weights, 1e-15)
        gradient = -(probability/denominator[:, None]).mean(axis=0)+.02*(weights-1/len(names))
        weights *= np.exp(np.clip(-.15*gradient, -10, 10))
        weights /= weights.sum()
    return {name: float(w) for name, w in zip(names, weights)}


def mixture(grids, weights, temperature=1.):
    if not weights or any(not math.isfinite(w) or w < 0 for w in weights.values()) or not math.isclose(sum(weights.values()), 1., abs_tol=1e-8):
        raise ValueError("valid_simplex_weights_required")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("positive_calibration_temperature_required")
    shape = tuple(max(grids[name].shape[i] for name in weights) for i in (0, 1))
    result = np.zeros(shape)
    for name, w in weights.items():
        grid = np.asarray(grids[name], dtype=float)
        if grid.ndim != 2 or not np.isfinite(grid).all() or np.any(grid < 0) or not np.isclose(grid.sum(), 1., atol=1e-7):
            raise ValueError("normalized_finite_component_grid_required")
        result[:grid.shape[0], :grid.shape[1]] += w*grid
    result = np.power(result, 1/temperature)
    return result/result.sum()


def fit_temperature(forecasts, weights):
    if len(forecasts) < PROTOCOL["minimum_calibration_rows"]:
        raise ValueError("insufficient_separate_calibration_rows")
    losses = {str(t): math.fsum(old._single(mixture(r["models"], weights, t), r["row"])
                                        ["exact_score_log_loss"] for r in forecasts)/len(forecasts)
              for t in PROTOCOL["temperature_candidates"]}
    chosen = min(PROTOCOL["temperature_candidates"], key=lambda t: (losses[str(t)], abs(t-1), t))
    return chosen, losses


def generate(context, targets, booster):
    forecasts, excluded = [], Counter()
    for row in targets:
        try:
            forecast = context.forecast(row)
            if forecast is None:
                excluded["insufficient_common_prior_team_or_league_history"] += 1
                continue
            add_boost(forecast, booster)
            forecasts.append(forecast)
        except (ValueError, OverflowError, FloatingPointError):
            excluded["unusable_numerical_goal_or_feature_support"] += 1
    return forecasts, dict(excluded)


def evaluate(forecasts, targets, names):
    observations = {name: [] for name in names}
    by_day = {r["date"]: [] for r in targets}
    for forecast in forecasts:
        if not all(name in forecast["models"] for name in names):
            continue
        by_day[forecast["row"]["date"]].append(forecast)
        for name in names:
            observations[name].append(old._single(forecast["models"][name], forecast["row"]))
    days = [{"source_date": day, "common_candidate_fixtures": len(rows),
             "models": {name: old.choose_pair(rows, name) for name in names}}
            for day, rows in sorted(by_day.items())]
    return {"source_matches": len(targets),
            "common_fixture_count": len(next(iter(observations.values()), [])),
            "model_names": list(names),
            "single_match_metrics": {name: old._summarize_singles(rows) for name, rows in observations.items()},
            "five_double_thresholds": {name: [old.threshold_summary(days, name, t)
                                               for t in PROTOCOL["thresholds"]] for name in names},
            "pair_comparison_scope": "each_model_selects_its_own_pair_from_same_common_source_date_cohort",
            "source_dates_not_Sydney_days": True,
            "date_records": days}


def _save_once(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("x", encoding="utf-8") as output:
        json.dump(value, output, sort_keys=True, indent=2, allow_nan=False)
        output.write("\n")


def run(rows, output, *, include_gradient_boost=False, max_development_fixtures=None,
        source_manifest_sha256=None, source_coverage=None, incomplete_source_dates=None):
    output = Path(output)
    context = ResearchContext(rows)
    protocol = {**PROTOCOL, "include_gradient_boost": bool(include_gradient_boost),
                "max_development_fixtures": max_development_fixtures,
                "implementation_sha256": file_sha(__file__),
                "preserved_fitter_sha256": file_sha(old.__file__),
                "source_digest": old.digest(context.rows),
                "source_manifest_sha256": source_manifest_sha256}
    _save_once(output/"protocol.json", protocol)
    booster, boost_status = train_boost(context, enabled=include_gradient_boost)
    names = list(CORE_MODELS)+([BOOST] if booster is not None else [])
    validation_targets = [r for r in context.rows if r["season"] == "2023-24"]
    validation, validation_exclusions = generate(context, validation_targets, booster)
    weight_rows = [r for r in validation if r["row"]["date"] < PROTOCOL["weight_fit_source_dates_before"]]
    calibration_rows = [r for r in validation if r["row"]["date"] >= PROTOCOL["temperature_calibration_source_dates_from"]]
    weights = fit_ensemble_weights(weight_rows, names)
    temperature, calibration_losses = fit_temperature(calibration_rows, weights)
    selection_metrics = {name: old._summarize_singles([old._single(r["models"][name], r["row"])
                                                     for r in weight_rows]) for name in names}
    selected = min(names, key=lambda name: (selection_metrics[name]["exact_score_log_loss"], name))
    fitted = {"protocol_sha256": old.digest(protocol), "source_digest": protocol["source_digest"],
              "selected_single_model_by_earlier_validation": selected,
              "selection_single_metrics": selection_metrics,
              "weight_fit_rows": len(weight_rows), "weight_fit_latest_source_date": max(r["row"]["date"] for r in weight_rows),
              "ensemble_weights": weights, "calibration_rows": len(calibration_rows),
              "calibration_first_source_date": min(r["row"]["date"] for r in calibration_rows),
              "calibration_latest_source_date": max(r["row"]["date"] for r in calibration_rows),
              "temperature": temperature, "calibration_loss_by_temperature": calibration_losses,
              "gradient_boost_status": boost_status,
              "development_results_consulted_during_fit": False}
    _save_once(output/"fitted_before_development.json", fitted)
    # No development-target forecasting or scores are evaluated before the
    # protocol and all learned calibration choices have been written above.
    development_targets = [r for r in context.rows if r["season"] == "2024-25"]
    if max_development_fixtures is not None:
        if type(max_development_fixtures) is not int or max_development_fixtures < 2:
            raise ValueError("development_fixture_limit_must_be_at_least_two")
        development_targets = development_targets[-max_development_fixtures:]
    development, development_exclusions = generate(context, development_targets, booster)
    for forecast in validation+development:
        forecast["models"][ENSEMBLE] = mixture(forecast["models"], weights, temperature)
        forecast["state"][ENSEMBLE] = forecast["state"]["dixon_coles_strength"]
    all_names = names+[ENSEMBLE]
    validation_report = evaluate(validation, validation_targets, all_names)
    development_report = evaluate(development, development_targets, all_names)
    report = {"protocol": protocol, "fitted_before_development": fitted,
              "validation_and_calibration_diagnostics_not_out_of_sample_for_ensemble": validation_report,
              "development_retrospective_comparison": development_report,
              "validation_exclusions": validation_exclusions,
              "development_exclusions": development_exclusions,
              "complete_case_source_coverage": source_coverage or {
                  "raw_schedule_completeness_checked": False},
              "source_slate_policy": "played_result_complete_case_development_only_missing_outcome_fixtures_not_reconstructed",
              "source_dependent_models": {
                  "shot_quality_xg": {"ready": False, "reason": "licensed_identity_matched_historical_shot_xg_not_in_CC0_results"},
                  "market_adjusted": {"ready": False, "reason": "timestamped_historical_bookmaker_score_or_market_prices_not_in_source"},
                  "independent_AI": {"ready": False, "reason": "no_matched_pre_match_provider_predictions_in_this_dataset"}},
              "target_70_percent_proven": False, "live_model_promoted": False,
              "requires_independent_future_prospective_validation": True,
              "limitations": [
                  "2024-25 has already been inspected in the November 15 experiment; this is development research.",
                  "Validation/calibration fits are not out-of-sample evidence for the ensemble.",
                  "Core models share historical goals and fitted means; they are not independent witnesses.",
                  "Different model policies can select different pairs on the same source dates.",
                  "Single exact-score or top-five accuracy does not measure five-double set success.",
                  "All five-double mass estimates assume independent matches and are not guaranteed true probabilities.",
                  "Date-only source histories cannot establish actual UTC kickoff availability or Sydney matchday cohorts.",
                  "Missing final scores can omit whole dates or alter candidate slates; complete-case pair selection is subject to this unresolved bias.",
                  "Additional data and successful software tests do not establish 70% football success."]}
    if incomplete_source_dates is not None:
        complete_days = [r for r in development_report["date_records"] if r["source_date"] not in incomplete_source_dates]
        report["complete_source_date_subset_descriptive_only"] = {
            name: [old.threshold_summary(complete_days, name, t) for t in PROTOCOL["thresholds"]]
            for name in all_names}
    _save_once(output/"comparison.json", report)
    summary = {"version": PROTOCOL["version"], "prediction_type": old.PROTOCOL["prediction_target"],
               "development_test_already_explored": True,
               "source_matches": len(context.rows), "optional_gradient_boost": boost_status,
               "fitted_ensemble": {"weights": weights, "temperature": temperature},
               "development_single_match_metrics": development_report["single_match_metrics"],
               "development_five_double_thresholds": development_report["five_double_thresholds"],
               "unavailable_source_models": report["source_dependent_models"],
               "complete_case_source_coverage": report["complete_case_source_coverage"],
               "source_slate_policy": report["source_slate_policy"],
               "target_70_percent_proven": False, "live_model_promoted": False,
               "full_comparison_file": "comparison.json"}
    _save_once(output/"summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/target70_research/source_manifest.json"))
    parser.add_argument("--output", type=Path, default=Path("data/research_model_comparison_v1"))
    parser.add_argument("--include-gradient-boost", action="store_true")
    parser.add_argument("--max-development-fixtures", type=int)
    args = parser.parse_args()
    from target70_data import load_normalized
    rows = load_normalized(args.manifest)
    from target70_postholdout_audit import source_audit, CACHE
    coverage, scheduled_dates, incomplete_dates = source_audit(
        json.loads(args.manifest.read_text()), CACHE, "2024-25")
    coverage["raw_schedule_completeness_checked"] = True
    coverage["raw_scheduled_source_dates"] = sorted(scheduled_dates)
    result = run(rows, args.output, include_gradient_boost=args.include_gradient_boost,
                 max_development_fixtures=args.max_development_fixtures,
                 source_manifest_sha256=file_sha(args.manifest),
                 source_coverage=coverage, incomplete_source_dates=incomplete_dates)
    print(json.dumps(result, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
