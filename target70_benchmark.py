"""Predeclared, date-safe CC0-data research for five exact-score doubles.

This standalone experiment never changes a live scorer. Source dates are not
UTC kickoff instants or Sydney matchdays. A monthly fit uses only source dates
strictly before the preceding calendar day. Validation chooses one model by
proper exact-score log loss before the holdout is evaluated.
"""
import argparse
from collections import Counter, defaultdict
from datetime import date, timedelta
import hashlib
from itertools import combinations
import json
import math
from pathlib import Path

import numpy as np

from engine import _grid
from five_double_accuracy import wilson


PROTOCOL = {
    "version": "target70-cc0-date-safe-v1",
    "prediction_target": "one_of_five_combinations_matches_both_regulation_scores",
    "training_seasons": ["2019-20", "2020-21", "2021-22", "2022-23"],
    "validation_season": "2023-24", "holdout_season": "2024-25",
    "leagues": ["en.1", "de.1", "es.1", "it.1", "fr.1"],
    "models": {
        "league_poisson": {"ridge": None, "half_life_days": 365},
        "opponent_dc_r6_h240": {"ridge": 6., "half_life_days": 240},
        "opponent_dc_r18_h365": {"ridge": 18., "half_life_days": 365}},
    "selection_metric": "validation_common_fixture_exact_score_log_loss",
    "selection_tie_break": "model_name_ascending",
    "monthly_refit": True,
    "training_rule": "source_date_strictly_before_month_start_minus_one_calendar_day",
    "minimum_league_history": 30, "minimum_team_history": 5,
    "maximum_pair_candidates_per_source_date": 18,
    "pair_choice": "greatest_top_five_independent_joint_model_mass",
    "four_distinct_teams": True,
    "thresholds": [0., .10, .20, .30, .50, .70],
    "thresholds_selected_using_holdout": False,
    "abstentions_count_in_opportunity_denominator": True,
    "interpretation": "retrospective_source_date_reconstruction_not_live_or_Sydney_accuracy",
    "no_paid_data_or_AI_calls": True, "live_model_promotion": False}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _date(row):
    return date.fromisoformat(row["date"])


def _team(row, side):
    return str(row["league"]) + ":" + str(row[side])


def _id(row):
    return str(row.get("match_id") or "|".join(
        str(row[k]) for k in ("league", "date", "home", "away")))


def validate_rows(rows):
    """Require completed, unambiguous observations; never resolve conflicts by wins."""
    output = {}
    for original in rows:
        row = dict(original)
        # The data module deliberately labels source dates rather than
        # supplying invented kickoff instants. Retain that provenance here.
        for short, original_key in (("league", "competition"), ("date", "source_date"),
                                    ("home", "home_team"), ("away", "away_team"),
                                    ("match_id", "source_match_id")):
            if short not in row and original_key in row:
                row[short] = row[original_key]
        if row["league"] not in PROTOCOL["leagues"]:
            raise ValueError("unsupported_league")
        _date(row)
        if not row["home"] or not row["away"] or row["home"] == row["away"]:
            raise ValueError("invalid_teams")
        if any(type(row[k]) is not int or row[k] < 0 for k in ("home_goals", "away_goals")):
            raise ValueError("invalid_regulation_scores")
        key = _id(row)
        if key in output and any(output[key][k] != row[k] for k in
                                  ("league", "date", "home", "away", "home_goals", "away_goals")):
            raise ValueError("conflicting_result_for_match_identity")
        output[key] = row
    return sorted(output.values(), key=lambda row: (row["date"], _id(row)))


def fit_state(rows, league, month_start, config):
    """Adapt the established reference-model fitter, keeping complete fitted ratings."""
    boundary = month_start - timedelta(days=1)
    history = [r for r in rows if r["league"] == league and _date(r) < boundary]
    if len(history) < PROTOCOL["minimum_league_history"]:
        return None
    teams = sorted({_team(r, side) for r in history for side in ("home", "away")})
    index = {team: i for i, team in enumerate(teams)}
    home = np.array([index[_team(r, "home")] for r in history], dtype=int)
    away = np.array([index[_team(r, "away")] for r in history], dtype=int)
    gh = np.array([r["home_goals"] for r in history], dtype=float)
    ga = np.array([r["away_goals"] for r in history], dtype=float)
    ages = np.array([(month_start-_date(r)).days for r in history], dtype=float)
    weight = 2.**(-ages/config["half_life_days"])
    ih = math.log(max(.05, float(np.average(gh, weights=weight))))
    ia = math.log(max(.05, float(np.average(ga, weights=weight))))
    attack, defence = np.zeros(len(teams)), np.zeros(len(teams))
    penalty = config["ridge"]
    rho = 0.
    if penalty is not None:
        for _ in range(35):
            old = np.r_[attack.copy(), defence.copy(), ih, ia]
            hm = np.exp(np.clip(ih+attack[home]+defence[away], -8, 5))
            am = np.exp(np.clip(ia+attack[away]+defence[home], -8, 5))
            gradient = (np.bincount(home, weights=weight*(gh-hm), minlength=len(teams))
                        +np.bincount(away, weights=weight*(ga-am), minlength=len(teams))
                        -penalty*attack)
            curvature = (np.bincount(home, weights=weight*hm, minlength=len(teams))
                         +np.bincount(away, weights=weight*am, minlength=len(teams))+penalty)
            attack += np.clip(gradient/np.maximum(curvature, 1e-9), -.5, .5)
            hm = np.exp(np.clip(ih+attack[home]+defence[away], -8, 5))
            am = np.exp(np.clip(ia+attack[away]+defence[home], -8, 5))
            gradient = (np.bincount(away, weights=weight*(gh-hm), minlength=len(teams))
                        +np.bincount(home, weights=weight*(ga-am), minlength=len(teams))
                        -penalty*defence)
            curvature = (np.bincount(away, weights=weight*hm, minlength=len(teams))
                         +np.bincount(home, weights=weight*am, minlength=len(teams))+penalty)
            defence += np.clip(gradient/np.maximum(curvature, 1e-9), -.5, .5)
            ih = math.log(max(.05, float(np.sum(weight*gh)/np.sum(
                weight*np.exp(np.clip(attack[home]+defence[away], -8, 5))))))
            ia = math.log(max(.05, float(np.sum(weight*ga)/np.sum(
                weight*np.exp(np.clip(attack[away]+defence[home], -8, 5))))))
            if np.max(np.abs(np.r_[attack, defence, ih, ia]-old)) < 1e-6:
                break
        hm = np.exp(np.clip(ih+attack[home]+defence[away], -8, 5))
        am = np.exp(np.clip(ia+attack[away]+defence[home], -8, 5))
        low = max(-.2, float(np.max(-.99/np.maximum(hm, am))))
        high = min(.2, float(np.min(.99/(hm*am))))
        candidates = np.linspace(low, high, 81)
        likelihood = -20*candidates*candidates
        for goals, factor in (((0, 0), -hm*am), ((0, 1), hm), ((1, 0), am)):
            mask = (gh == goals[0]) & (ga == goals[1])
            likelihood += np.sum(weight[mask, None]*np.log(
                1+factor[mask, None]*candidates[None, :]), axis=0)
        likelihood += weight[(gh == 1) & (ga == 1)].sum()*np.log(1-candidates)
        rho = float(candidates[int(np.argmax(likelihood))])
    return {"league": league, "month_start": month_start.isoformat(),
            "strict_training_date_boundary": boundary.isoformat(),
            "last_training_source_date": max(r["date"] for r in history),
            "training_matches": len(history), "index": index,
            "counts": np.bincount(np.r_[home, away], minlength=len(teams)),
            "attack": attack, "defence": defence,
            "intercept_home": ih, "intercept_away": ia, "rho": rho}


def matrix_for(row, state):
    if state is None:
        return None
    index = state["index"]
    if any(_team(row, side) not in index for side in ("home", "away")):
        return None
    h, a = (index[_team(row, side)] for side in ("home", "away"))
    if min(state["counts"][h], state["counts"][a]) < PROTOCOL["minimum_team_history"]:
        return None
    lh = math.exp(state["intercept_home"]+state["attack"][h]+state["defence"][a])
    la = math.exp(state["intercept_away"]+state["attack"][a]+state["defence"][h])
    rho = min(max(state["rho"], -.99/max(lh, la)), .99/(lh*la))
    return _grid(lh, la, rho=rho)


def _top(grid, limit=5):
    return sorted(((float(p), h, a) for h, row in enumerate(grid)
                   for a, p in enumerate(row)), key=lambda item: (-item[0], item[1:]))[:limit]


def five(first, second):
    return sorted(({
        "score1": [h, a], "score2": [j, k], "joint_probability": p*q}
        for p, h, a in _top(first) for q, j, k in _top(second)),
        key=lambda item: (-item["joint_probability"], item["score1"], item["score2"]))[:5]


def choose_pair(forecasts, model):
    ranked = sorted(forecasts, key=lambda row: (-_top(row["models"][model], 1)[0][0],
                                               _id(row["row"])))
    ranked = ranked[:PROTOCOL["maximum_pair_candidates_per_source_date"]]
    best = None
    for first, second in combinations(ranked, 2):
        teams = {_team(item["row"], side) for item in (first, second) for side in ("home", "away")}
        if len(teams) != 4:
            continue
        picks = five(first["models"][model], second["models"][model])
        mass = math.fsum(p["joint_probability"] for p in picks)
        key = (-mass, _id(first["row"]), _id(second["row"]))
        if best is None or key < best[0]:
            best = (key, first, second, picks)
    if best is None:
        return None
    _, first, second, picks = best
    actual = [[item["row"]["home_goals"], item["row"]["away_goals"]] for item in (first, second)]
    return {"match_ids": [_id(item["row"]) for item in (first, second)],
            "fixtures": [{k: item["row"][k] for k in ("league", "date", "home", "away")}
                         for item in (first, second)],
            "fit_month_starts": [item["state"][model]["month_start"] for item in (first, second)],
            "last_training_source_dates": [item["state"][model]["last_training_source_date"]
                                           for item in (first, second)],
            "five_doubles": picks, "estimated_coverage": math.fsum(
                p["joint_probability"] for p in picks), "actual_scores": actual,
            "hit": any(p["score1"] == actual[0] and p["score2"] == actual[1] for p in picks),
            "assumption": "independent_matches"}


def _single(grid, row):
    h, a = row["home_goals"], row["away_goals"]
    p = float(grid[h, a]) if h < grid.shape[0] and a < grid.shape[1] else 0.
    top = _top(grid)
    outcome = 0 if h > a else 1 if h == a else 2
    outcome_p = (float(np.tril(grid, -1).sum()), float(np.trace(grid)),
                 float(np.triu(grid, 1).sum()))
    return {"exact_hit": (h, a) == top[0][1:],
            "top5_hit": any((h, a) == cell[1:] for cell in top),
            "exact_score_log_loss": -math.log(max(p, 1e-15)),
            "outcome_brier": sum((value-int(i == outcome))**2 for i, value in enumerate(outcome_p))}


def _summarize_singles(rows):
    n = len(rows)
    return {"evaluated": n, "exact_hits": sum(row["exact_hit"] for row in rows),
            "top5_hits": sum(row["top5_hit"] for row in rows),
            **{key: math.fsum(row[key] for row in rows)/n if n else None for key in
               ("exact_hit", "top5_hit", "exact_score_log_loss", "outcome_brier")}}


def threshold_summary(days, model, threshold):
    issued = [day["models"][model] for day in days if day["models"].get(model) is not None
              and day["models"][model]["estimated_coverage"] >= threshold]
    wins = sum(row["hit"] for row in issued)
    n, opportunities = len(issued), len(days)
    return {"minimum_model_coverage": threshold, "opportunities": opportunities,
            "issued": n, "abstained": opportunities-n, "hits": wins,
            "rate_per_issued": wins/n if n else None,
            "rate_per_opportunity": wins/opportunities if opportunities else None,
            "wilson_95_assuming_independent_sets": wilson(wins, n),
            "mean_model_coverage": math.fsum(r["estimated_coverage"] for r in issued)/n if n else None,
            "maximum_model_coverage": max((r["estimated_coverage"] for r in issued), default=None),
            "zero_issued_does_not_achieve_target": n == 0,
            "prospective_validation_required": True,
            "target_70_proven": False}


def evaluate_season(rows, season):
    rows = validate_rows(rows)
    targets = [r for r in rows if r["season"] == season]
    cache = {}
    excluded = Counter()
    singles = {name: [] for name in PROTOCOL["models"]}
    per_league = {name: defaultdict(list) for name in PROTOCOL["models"]}
    by_day = {r["date"]: [] for r in targets}
    for row in targets:
        at = _date(row)
        month = at.replace(day=1)
        states, models = {}, {}
        for name, config in PROTOCOL["models"].items():
            key = (row["league"], month, name)
            if key not in cache:
                cache[key] = fit_state(rows, row["league"], month, config)
            state = cache[key]
            matrix = matrix_for(row, state)
            if matrix is None:
                break
            states[name], models[name] = state, matrix
        if len(models) != len(PROTOCOL["models"]):
            excluded["insufficient_common_history_or_team_evidence"] += 1
            continue
        for name, grid in models.items():
            observation = _single(grid, row)
            singles[name].append(observation)
            per_league[name][row["league"]].append(observation)
        by_day[row["date"]].append({"row": row, "models": models, "state": states})
    days = [{"source_date": day, "qualified_candidate_fixtures": len(forecasts),
             "models": {name: choose_pair(forecasts, name) for name in PROTOCOL["models"]}}
            for day, forecasts in sorted(by_day.items())]
    return {"season": season, "source_matches": len(targets),
            "common_fixture_evaluated": len(singles["league_poisson"]),
            "excluded": dict(excluded), "source_date_opportunities": len(days),
            "single_match_metrics": {name: _summarize_singles(observations)
                                     for name, observations in singles.items()},
            "single_metrics_by_league": {name: {league: _summarize_singles(obs)
                                                for league, obs in sorted(leagues.items())}
                                         for name, leagues in per_league.items()},
            "five_double_thresholds": {name: [threshold_summary(days, name, threshold)
                                               for threshold in PROTOCOL["thresholds"]]
                                        for name in PROTOCOL["models"]},
            "model_pair_selection_comparison": "each_policy_chooses_its_own_pair_on_same_source_dates",
            "date_records": days}


def _save_once(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as output:
        output.write(json.dumps(value, sort_keys=True, indent=2, allow_nan=False)+"\n")


def select_from_validation(validation, source_digest):
    if validation["season"] != PROTOCOL["validation_season"]:
        raise ValueError("selection_requires_declared_validation_season")
    available = [(m["exact_score_log_loss"], name) for name, m in
                 validation["single_match_metrics"].items()
                 if m["evaluated"] and m["exact_score_log_loss"] is not None]
    if not available:
        raise ValueError("no_common_validation_forecasts")
    selection = {"protocol_sha256": digest(PROTOCOL), "source_digest": source_digest,
                 "validation_report_sha256": digest(validation),
                 "selected_model": min(available)[1],
                 "selection_criterion": PROTOCOL["selection_metric"],
                 "holdout_evaluated_at_selection": False,
                 "validation_metrics": validation["single_match_metrics"]}
    selection["selection_sha256"] = digest(selection)
    return selection


def validate_selection(selection, source_digest):
    body = {k: v for k, v in selection.items() if k != "selection_sha256"}
    if (selection.get("selection_sha256") != digest(body)
            or selection.get("protocol_sha256") != digest(PROTOCOL)
            or selection.get("source_digest") != source_digest
            or selection.get("selected_model") not in PROTOCOL["models"]
            or selection.get("holdout_evaluated_at_selection") is not False):
        raise ValueError("invalid_or_changed_pre_holdout_selection")


def run_validation(rows, output, *, source_manifest_sha256=None):
    output = Path(output)
    # Restrict the selection process to prior seasons before it sees outcomes
    # from the held-out season; artifact hashes still bind the complete source.
    normalized = validate_rows(rows)
    training_allowed = set(PROTOCOL["training_seasons"]+[PROTOCOL["validation_season"]])
    earlier = [r for r in normalized if r["season"] in training_allowed]
    protocol_path = output/"protocol.json"
    if protocol_path.exists():
        if json.loads(protocol_path.read_text()) != PROTOCOL:
            raise ValueError("existing_protocol_differs")
    else:
        _save_once(protocol_path, PROTOCOL)
    validation = evaluate_season(earlier, PROTOCOL["validation_season"])
    selection = select_from_validation(validation, digest(normalized))
    if source_manifest_sha256 is not None:
        selection.pop("selection_sha256")
        selection["source_manifest_sha256"] = source_manifest_sha256
        selection["selection_sha256"] = digest(selection)
    _save_once(output/"validation.json", validation)
    _save_once(output/"selection_before_holdout.json", selection)
    return selection


def run_holdout(rows, output, *, validation_output=None, source_manifest_sha256=None):
    output = Path(output)
    # The selection artifact must already be persisted before holdout starts.
    validation_output = Path(validation_output) if validation_output is not None else output
    selection = json.loads((validation_output/"selection_before_holdout.json").read_text())
    normalized = validate_rows(rows)
    validate_selection(selection, digest(normalized))
    if selection.get("source_manifest_sha256") != source_manifest_sha256:
        raise ValueError("source_manifest_changed_after_selection")
    if json.loads((validation_output/"protocol.json").read_text()) != PROTOCOL:
        raise ValueError("protocol_changed_after_selection")
    validation = json.loads((validation_output/"validation.json").read_text())
    expected = select_from_validation(validation, digest(normalized))
    if source_manifest_sha256 is not None:
        expected.pop("selection_sha256")
        expected["source_manifest_sha256"] = source_manifest_sha256
        expected["selection_sha256"] = digest(expected)
    if selection != expected:
        raise ValueError("selection_does_not_match_frozen_validation_evidence")
    result = evaluate_season(normalized, PROTOCOL["holdout_season"])
    chosen = selection["selected_model"]
    report = {"protocol_sha256": digest(PROTOCOL), "selection": selection,
              "selected_model": chosen, "holdout": result,
              "chosen_model_threshold_results": result["five_double_thresholds"][chosen],
              "target_70_percent_proven": False, "live_model_promoted": False,
              "new_data_is_not_prospective_evidence": True,
              "limitations": [
                  "Retrospective source-date reconstruction; no exact kickoff timestamps are inferred.",
                  "Date groups are not Sydney matchdays and must not be merged with live original accuracy.",
                  "Monthly updating uses earlier results with a full preceding-source-day exclusion.",
                  "Model probability and observed hit rate remain separate.",
                  "The holdout season is new to this experiment, not guaranteed untouched by all prior developer research.",
                  "Cross-day football outcomes need not be independent; Wilson intervals are descriptive.",
                  "No zero-prediction threshold is credited as success.",
                  "Source data lack archived odds, real shot xG and pre-match lineups; no such feature is fabricated."]}
    _save_once(output/"holdout.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--manifest", type=Path, default=Path("data/target70_research/source_manifest.json"))
    parser.add_argument("--output", type=Path, default=Path("data/target70_research"))
    parser.add_argument("--validation-output", type=Path)
    parser.add_argument("--holdout-output", type=Path)
    parser.add_argument("--phase", choices=("validate", "holdout"), required=True)
    args = parser.parse_args()
    if args.input is not None:
        rows = json.loads(args.input.read_text())
        manifest_hash = None
    else:
        from target70_data import load_normalized
        rows = load_normalized(args.manifest)
        manifest_hash = hashlib.sha256(args.manifest.read_bytes()).hexdigest()
    validation_output = args.validation_output or args.output
    result = (run_validation(rows, validation_output, source_manifest_sha256=manifest_hash)
              if args.phase == "validate" else run_holdout(
                  rows, args.holdout_output or args.output, validation_output=validation_output,
                  source_manifest_sha256=manifest_hash))
    print(json.dumps({k: v for k, v in result.items() if k not in ("holdout", "validation_metrics")},
                     sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
