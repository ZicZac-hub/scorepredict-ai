"""No metric substitution, date leakage or holdout-based model selection."""
from copy import deepcopy
from datetime import date, timedelta
import json

import numpy as np
import pytest

import target70_benchmark as bench


def match(day, *, home="A", away="B", hg=1, ag=0, season="2022-23", ident=None):
    return {"league": "en.1", "date": str(day), "home": home, "away": away,
            "home_goals": hg, "away_goals": ag, "season": season,
            "match_id": ident or f"en.1|{day}|{home}|{away}"}


def history():
    return [match(date(2022, 1, 1)+timedelta(days=i),
                  home=("A" if i % 2 else "B"), away=("B" if i % 2 else "A"),
                  hg=i % 3, ag=i % 2) for i in range(60)]


def test_monthly_fit_excludes_target_month_and_entire_preceding_source_day():
    rows = history()
    month = date(2022, 2, 1)
    config = bench.PROTOCOL["models"]["opponent_dc_r6_h240"]
    fitted = bench.fit_state(rows, "en.1", month, config)
    assert fitted["training_matches"] == 30
    assert fitted["last_training_source_date"] == "2022-01-30"
    modified = deepcopy(rows)
    for row in modified:
        if row["date"] >= "2022-01-31":
            row["home_goals"], row["away_goals"] = 90, 90
    second = bench.fit_state(modified, "en.1", month, config)
    for field in ("attack", "defence", "counts"):
        assert np.array_equal(fitted[field], second[field])
    assert fitted["rho"] == second["rho"]
    assert fitted["intercept_home"] == second["intercept_home"]


def test_matrix_is_valid_and_unknown_or_thin_team_abstains():
    fitted = bench.fit_state(history(), "en.1", date(2022, 4, 1),
                             bench.PROTOCOL["models"]["opponent_dc_r6_h240"])
    predicted = bench.matrix_for(match("2022-04-15"), fitted)
    assert np.isfinite(predicted).all()
    assert (predicted >= 0).all()
    assert predicted.sum() == pytest.approx(1)
    assert bench.matrix_for(match("2022-04-15", home="newly promoted"), fitted) is None


def test_source_date_adapter_does_not_invent_utc_or_sydney():
    row = {"source": "openfootball", "competition": "en.1", "source_date": "2024-08-16",
           "home_team": "A", "away_team": "B", "home_goals": 1, "away_goals": 0,
           "source_match_id": "source1", "season": "2024-25",
           "time_precision": "source_calendar_date_only"}
    parsed = bench.validate_rows([row])[0]
    assert parsed["date"] == row["source_date"]
    assert parsed["match_id"] == "source1"
    assert "kickoff_utc" not in parsed
    assert "sydney_date" not in parsed


def test_conflicting_and_non_integer_results_never_silently_enter_research():
    row = match("2024-08-16")
    with pytest.raises(ValueError, match="conflicting_result"):
        bench.validate_rows([row, {**row, "home_goals": 2}])
    with pytest.raises(ValueError, match="invalid_regulation"):
        bench.validate_rows([{**row, "home_goals": True}])
    assert len(bench.validate_rows([row, dict(row)])) == 1


def test_five_maximize_the_original_product_mass():
    first = np.array([[.3, .2], [.4, .1]])
    second = np.array([[.15, .25], [.35, .25]])
    chosen = bench.five(first, second)
    brute = sorted((float(p*q) for p in first.ravel() for q in second.ravel()), reverse=True)[:5]
    assert [row["joint_probability"] for row in chosen] == pytest.approx(brute)
    assert len({(tuple(row["score1"]), tuple(row["score2"])) for row in chosen}) == 5


def forecast(row, grid):
    return {"row": row, "models": {"chosen": grid}, "state": {"chosen": {
        "month_start": "2024-08-01", "last_training_source_date": "2024-07-30"}}}


def test_pair_choice_never_uses_outcomes_and_requires_both_scores_in_one_row():
    grid = np.array([[.35, .2], [.3, .15]])
    first = forecast(match("2024-08-16", home="A", away="B"), grid)
    second = forecast(match("2024-08-16", home="C", away="D"), grid)
    pair = bench.choose_pair([first, second], "chosen")
    picks = {(tuple(p["score1"]), tuple(p["score2"])) for p in pair["five_doubles"]}
    a = {s[0] for s in picks}; b = {s[1] for s in picks}
    absent = next((x, y) for x in a for y in b if (x, y) not in picks)
    first["row"]["home_goals"], first["row"]["away_goals"] = absent[0]
    second["row"]["home_goals"], second["row"]["away_goals"] = absent[1]
    changed = bench.choose_pair([first, second], "chosen")
    assert changed["five_doubles"] == pair["five_doubles"]
    assert changed["match_ids"] == pair["match_ids"]
    assert changed["hit"] is False
    second["row"]["home"] = "A"
    assert bench.choose_pair([first, second], "chosen") is None


def test_threshold_abstentions_never_count_as_seventy_percent():
    days = [{"models": {"chosen": {"estimated_coverage": .07, "hit": True}}},
            {"models": {"chosen": None}}]
    base = bench.threshold_summary(days, "chosen", 0)
    assert base["issued"] == 1
    assert base["rate_per_issued"] == 1
    assert base["rate_per_opportunity"] == .5
    assert base["target_70_proven"] is False
    strict = bench.threshold_summary(days, "chosen", .70)
    assert strict["issued"] == 0
    assert strict["abstained"] == 2
    assert strict["rate_per_issued"] is None
    assert strict["target_70_proven"] is False
    assert strict["zero_issued_does_not_achieve_target"] is True


def test_selection_uses_validation_log_loss_and_is_hashed_before_holdout():
    validation = {"season": "2023-24", "single_match_metrics": {
        "league_poisson": {"evaluated": 10, "exact_score_log_loss": 4., "exact_hit": 1.},
        "opponent_dc_r6_h240": {"evaluated": 10, "exact_score_log_loss": 3., "exact_hit": 0.}}}
    selected = bench.select_from_validation(validation, "source")
    assert selected["selected_model"] == "opponent_dc_r6_h240"
    assert selected["holdout_evaluated_at_selection"] is False
    bench.validate_selection(selected, "source")
    with pytest.raises(ValueError, match="changed_pre_holdout"):
        bench.validate_selection(selected, "changed-source")
    selected["selected_model"] = "league_poisson"
    with pytest.raises(ValueError, match="changed_pre_holdout"):
        bench.validate_selection(selected, "source")


def test_validation_phase_does_not_pass_holdout_results_to_forecasting(monkeypatch, tmp_path):
    rows = [match("2022-01-01"), match("2023-08-16", season="2023-24"),
            match("2024-08-16", season="2024-25", hg=99)]
    seen = []
    def evaluate(data, season):
        seen.extend(data)
        return {"season": season, "single_match_metrics": {
            "league_poisson": {"evaluated": 1, "exact_score_log_loss": 3.}}}
    monkeypatch.setattr(bench, "evaluate_season", evaluate)
    selected = bench.run_validation(rows, tmp_path)
    assert all(row["season"] != "2024-25" for row in seen)
    saved = json.loads((tmp_path/"selection_before_holdout.json").read_text())
    assert saved == selected
    with pytest.raises(FileExistsError):
        bench.run_validation(rows, tmp_path)


def test_holdout_cannot_run_without_existing_frozen_selection(tmp_path):
    with pytest.raises(FileNotFoundError):
        bench.run_holdout(history(), tmp_path)


def test_holdout_rejects_changed_protocol_before_evaluating(monkeypatch, tmp_path):
    validation = {"season": "2023-24", "single_match_metrics": {
        "league_poisson": {"evaluated": 1, "exact_score_log_loss": 3.}}}
    rows = bench.validate_rows(history())
    selected = bench.select_from_validation(validation, bench.digest(rows))
    (tmp_path/"selection_before_holdout.json").write_text(json.dumps(selected))
    (tmp_path/"protocol.json").write_text(json.dumps({"version": "changed"}))
    monkeypatch.setattr(bench, "evaluate_season", lambda *args: pytest.fail("holdout must not run"))
    with pytest.raises(ValueError, match="protocol_changed"):
        bench.run_holdout(rows, tmp_path)


def test_common_fixture_denominators_and_source_date_opportunities():
    rows = history()+[match("2024-08-16", season="2024-25"),
                     match("2024-08-16", home="C", away="D", season="2024-25")]
    report = bench.evaluate_season(rows, "2024-25")
    assert report["common_fixture_evaluated"] == 1
    assert report["source_matches"] == 2
    assert report["source_date_opportunities"] == 1
    assert len({m["evaluated"] for m in report["single_match_metrics"].values()}) == 1
    assert all(stats[0]["abstained"] == 1 for stats in report["five_double_thresholds"].values())


def test_holdout_rejects_rehashed_model_change_against_validation(monkeypatch, tmp_path):
    rows = bench.validate_rows(history())
    validation = {"season": "2023-24", "single_match_metrics": {
        "league_poisson": {"evaluated": 10, "exact_score_log_loss": 4.},
        "opponent_dc_r6_h240": {"evaluated": 10, "exact_score_log_loss": 3.}}}
    selected = bench.select_from_validation(validation, bench.digest(rows))
    selected["selected_model"] = "league_poisson"
    selected.pop("selection_sha256")
    selected["selection_sha256"] = bench.digest(selected)
    (tmp_path/"selection_before_holdout.json").write_text(json.dumps(selected))
    (tmp_path/"protocol.json").write_text(json.dumps(bench.PROTOCOL))
    (tmp_path/"validation.json").write_text(json.dumps(validation))
    monkeypatch.setattr(bench, "evaluate_season", lambda *args: pytest.fail("holdout must not run"))
    with pytest.raises(ValueError, match="frozen_validation_evidence"):
        bench.run_holdout(rows, tmp_path)
