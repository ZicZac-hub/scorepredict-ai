"""Research models remain source-date safe, comparable and unpromoted."""
from copy import deepcopy
from datetime import date, timedelta
import json

import numpy as np
import pytest

import model_comparison_research as research
from target70_benchmark import digest


def match(day, *, home="A", away="B", hg=1, ag=0, season="2022-23"):
    return {"league": "en.1", "date": str(day), "home": home, "away": away,
            "home_goals": hg, "away_goals": ag, "season": season}


def history(start=date(2021, 1, 1), count=90):
    return [match(start+timedelta(days=i), home="A" if i % 2 else "B",
                  away="B" if i % 2 else "A", hg=i % 4, ag=i % 3)
            for i in range(count)]


def moments(grid):
    h, a = np.arange(grid.shape[0])[:, None], np.arange(grid.shape[1])[None, :]
    mh, ma = float((grid*h).sum()), float((grid*a).sum())
    return mh, ma, float((grid*(h-mh)**2).sum()), float((grid*(h-mh)*(a-ma)).sum())


def test_distribution_families_have_valid_mass_and_distinct_dependence():
    nb = research.negative_binomial_grid(1.8, 1.2)
    bi = research.bivariate_grid(1.8, 1.2)
    for grid in (nb, bi):
        assert np.isfinite(grid).all()
        assert (grid >= 0).all()
        assert grid.sum() == pytest.approx(1)
        assert moments(grid)[:2] == pytest.approx((1.8, 1.2), abs=1e-7)
    assert moments(nb)[2] == pytest.approx(1.8+1.8**2/3, abs=1e-6)
    assert moments(nb)[3] == pytest.approx(0, abs=1e-8)
    assert moments(bi)[3] == pytest.approx(.12, abs=1e-8)
    with pytest.raises(ValueError, match="positive_negative_binomial"):
        research.negative_binomial_grid(1.8, 1.2, shape=0)


def test_core_models_and_features_ignore_current_and_future_scores():
    rows = history()
    target = match("2022-04-10")
    original = research.ResearchContext(rows+[target]).forecast(target)
    altered = deepcopy(rows)+[{**target, "home_goals": 99, "away_goals": 99},
                              match("2025-12-31", hg=99, ag=99)]
    second = research.ResearchContext(altered).forecast(target)
    assert set(original["models"]) == set(research.CORE_MODELS)
    assert np.array_equal(original["features"], second["features"])
    for name in research.CORE_MODELS:
        assert np.array_equal(original["models"][name], second["models"][name])
    assert original["feature_provenance"]["latest_feature_result_source_date"] < "2022-03-31"
    assert original["feature_provenance"]["current_fixture_goals_used_as_features"] is False
    assert len(original["features"]) == len(research.FEATURES)


def test_month_boundary_excludes_entire_previous_source_day():
    rows = history(date(2022, 1, 1), count=90)
    target = match("2022-03-15")
    first = research.ResearchContext(rows).forecast(target)
    for row in rows:
        if row["date"] >= "2022-02-28":
            row["home_goals"], row["away_goals"] = 77, 77
    second = research.ResearchContext(rows).forecast(target)
    assert first["feature_provenance"]["latest_feature_result_source_date"] == "2022-02-27"
    assert np.array_equal(first["features"], second["features"])


def test_optional_boost_abstains_without_request_or_dependency(monkeypatch):
    context = research.ResearchContext(history())
    model, status = research.train_boost(context)
    assert model is None
    assert status["reason"] == "optional_gradient_boost_not_requested"
    monkeypatch.setattr(research.importlib.util, "find_spec", lambda name: None)
    model, status = research.train_boost(context, enabled=True)
    assert model is None
    assert status["reason"] == "optional_sklearn_dependency_unavailable"


def test_boost_training_excludes_validation_and_development_labels():
    rows = history(date(2021, 1, 1), count=90)+history(date(2022, 1, 1), count=140)
    rows += [match("2023-08-01", hg=99, ag=99, season="2023-24"),
             match("2024-08-01", hg=99, ag=99, season="2024-25")]
    captures = []
    class Estimator:
        def __init__(self, **config):
            self.config = config
        def fit(self, x, y):
            captures.append((x.copy(), y.copy()))
            assert np.max(y) < 99
            self.rate = float(y.mean())
        def predict(self, x):
            return np.full(len(x), self.rate)
    models, status = research.train_boost(research.ResearchContext(rows), enabled=True,
                                          estimator_factory=Estimator)
    assert models is not None
    assert len(captures) == 2
    assert status["latest_training_label_source_date"] < "2023-07-01"
    assert status["training_rows"] >= 100
    assert captures[0][0].shape[1] == len(research.FEATURES)


def test_mixture_preserves_normalization_and_temperature_is_fitted_separately():
    grids = {"first": np.array([[.4, .1], [.3, .2]]),
             "second": np.array([[.2, .2, .1], [.2, .2, .1]])}
    mix = research.mixture(grids, {"first": .3, "second": .7})
    assert mix.sum() == pytest.approx(1)
    assert mix.shape == (2, 3)
    assert mix[0, 0] == pytest.approx(.3*.4+.7*.2)
    for temperature in research.PROTOCOL["temperature_candidates"]:
        adjusted = research.mixture(grids, {"first": .3, "second": .7}, temperature)
        assert adjusted.sum() == pytest.approx(1)
        assert (adjusted >= 0).all()
    with pytest.raises(ValueError, match="simplex"):
        research.mixture(grids, {"first": .3, "second": .3})
    with pytest.raises(ValueError, match="temperature"):
        research.mixture(grids, {"first": .3, "second": .7}, 0)


def test_ensemble_weight_fit_is_positive_simplex_and_uses_actual_probabilities():
    forecasts = [{"row": match("2023-08-01", hg=0, ag=0), "models": {
        "first": np.array([[.8, .1], [.05, .05]]),
        "second": np.array([[.1, .3], [.3, .3]])}} for _ in range(50)]
    weights = research.fit_ensemble_weights(forecasts, ["first", "second"])
    assert sum(weights.values()) == pytest.approx(1)
    assert all(w >= 0 for w in weights.values())
    assert weights["first"] > weights["second"]
    with pytest.raises(ValueError, match="insufficient_ensemble"):
        research.fit_ensemble_weights([], ["first", "second"])


def test_calibration_missing_evidence_never_means_default_success():
    with pytest.raises(ValueError, match="insufficient_separate_calibration"):
        research.fit_temperature([], {"first": 1.})


def test_development_cannot_start_before_choices_are_saved(monkeypatch, tmp_path):
    rows = history()+[match("2023-08-15", season="2023-24"),
                     match("2024-02-15", season="2023-24"),
                     match("2024-08-15", season="2024-25")]
    monkeypatch.setitem(research.PROTOCOL, "minimum_weight_fit_rows", 1)
    monkeypatch.setitem(research.PROTOCOL, "minimum_calibration_rows", 1)
    original_generate = research.generate
    def checked(context, targets, booster):
        if any(r["season"] == "2024-25" for r in targets):
            assert (tmp_path/"protocol.json").is_file()
            assert (tmp_path/"fitted_before_development.json").is_file()
            fitted = json.loads((tmp_path/"fitted_before_development.json").read_text())
            assert fitted["weight_fit_latest_source_date"] < "2024-01-01"
            assert fitted["calibration_first_source_date"] >= "2024-01-01"
            assert fitted["development_results_consulted_during_fit"] is False
        return original_generate(context, targets, booster)
    monkeypatch.setattr(research, "generate", checked)
    original_hash = research.file_sha(research.old.__file__)
    result = research.run(rows, tmp_path, source_coverage={"holdout_missing_full_time_rows": 20},
                          incomplete_source_dates={"2024-08-15"})
    assert research.file_sha(research.old.__file__) == original_hash
    assert result["target_70_percent_proven"] is False
    assert result["live_model_promoted"] is False
    assert result["development_test_already_explored"] is True
    assert result["complete_case_source_coverage"]["holdout_missing_full_time_rows"] == 20
    assert all(status["ready"] is False for status in result["unavailable_source_models"].values())


def test_same_fixture_cohort_and_zero_pairs_remain_abstentions():
    context = research.ResearchContext(history())
    row = match("2024-08-15", season="2024-25")
    forecast = context.forecast(row)
    result = research.evaluate([forecast], [row], research.CORE_MODELS)
    assert result["common_fixture_count"] == 1
    assert len({m["evaluated"] for m in result["single_match_metrics"].values()}) == 1
    for thresholds in result["five_double_thresholds"].values():
        assert thresholds[0]["issued"] == 0
        assert thresholds[0]["rate_per_issued"] is None
        assert thresholds[0]["target_70_proven"] is False


def test_protocol_does_not_claim_another_unseen_2024_holdout():
    assert research.PROTOCOL["independent_unseen_test"] is False
    assert research.PROTOCOL["development_test_season_already_explored"] == "2024-25"
    assert research.PROTOCOL["future_prospective_validation_required"] is True
