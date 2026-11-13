"""Leakage-safe calibration of a Baseline/Venue Contrast probability hybrid.

Choose the blend using 2025 fixtures only; evaluate once on 2026.
Weights mix complete score probability matrices, not hand-picked scorelines.
"""
import argparse
import json
import math
from collections import defaultdict
import numpy as np
from engine import predict, finished, _date, mix_score_matrices

WEIGHTS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0)
EPS = 1e-12

def metrics(rows, weight):
    total = len(rows)
    if not total:
        return {"evaluated": 0}
    exact = top3 = outcome = one_one = 0
    loss = brier = 0.0
    for baseline, contrast, home, away in rows:
        matrix = mix_score_matrices([baseline,contrast],[1.0-weight,weight])
        ranked = np.argsort(matrix.ravel())[::-1][:3]
        predicted = np.unravel_index(int(ranked[0]), matrix.shape)
        exact += int(predicted == (home, away))
        top3 += int(home * matrix.shape[1] + away in ranked)
        one_one += int(predicted == (1, 1))
        loss -= math.log(max(float(matrix[home, away]) if home < matrix.shape[0] and away < matrix.shape[1] else EPS, EPS))
        probs = (float(np.tril(matrix, -1).sum()), float(np.trace(matrix)),
                 float(np.triu(matrix, 1).sum()))
        actual = 0 if home > away else 1 if home == away else 2
        outcome += int(int(np.argmax(probs)) == actual)
        brier += sum((p - int(i == actual)) ** 2 for i, p in enumerate(probs))
    return {"evaluated": total, "exact_hits": exact, "exact_hit_rate": exact / total,
            "top3_hit_rate": top3 / total, "outcome_accuracy": outcome / total,
            "predicted_1_1_share": one_one / total,
            "score_log_loss": loss / total, "outcome_brier": brier / total}

def evaluate(matches, calibration_year=2025, holdout_year=2026, min_history=30):
    ordered = sorted((m for m in matches if finished(m) and _date(m.get("utcDate"))),
                     key=lambda m: _date(m["utcDate"]))
    histories = defaultdict(list)
    calibration, holdout = [], []
    for fixture in ordered:
        league = (fixture.get("competition") or {}).get("code") or "unknown"
        kickoff = _date(fixture["utcDate"])
        year = kickoff.year
        if year in (calibration_year, holdout_year):
            past = [m for m in histories[league] if _date(m["utcDate"]) < kickoff]
            if len(past) >= min_history:
                baseline = predict(fixture, past, variant="baseline")
                contrast = predict(fixture, past, variant="venue_contrast")
                if baseline and contrast:
                    score = fixture["score"]["fullTime"]
                    row = (baseline["matrix"], contrast["matrix"], score["home"], score["away"])
                    (calibration if year == calibration_year else holdout).append(row)
        histories[league].append(fixture)
    if not calibration or not holdout:
        return {"error": "Insufficient chronological calibration or holdout matches",
                "calibration_matches": len(calibration), "holdout_matches": len(holdout)}
    calibration_results = {str(w): metrics(calibration, w) for w in WEIGHTS}
    # Select on proper probability scoring rule; tie breaks towards baseline.
    selected = min(WEIGHTS, key=lambda w: (calibration_results[str(w)]["score_log_loss"], w))
    holdout_results = {str(w): metrics(holdout, w) for w in (0.0, selected, 1.0)}
    return {"calibration_year": calibration_year, "holdout_year": holdout_year,
            "selected_venue_contrast_weight": selected,
            "calibration": calibration_results, "holdout": holdout_results,
            "note": "Weights chosen exclusively on 2025 score log loss; 2026 remains untouched for selection. A single holdout does not prove future gains. Live baseline unchanged."}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("matches_json")
    parser.add_argument("--output", default="hybrid_calibration_results.json")
    args = parser.parse_args()
    with open(args.matches_json, encoding="utf-8") as source:
        matches = json.load(source)
    report = evaluate(matches)
    with open(args.output, "w", encoding="utf-8") as output:
        json.dump(report, output, indent=2)
    print(json.dumps(report, indent=2))
    if "error" in report:
        raise SystemExit(report["error"])

