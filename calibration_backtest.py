"""Out-of-sample temperature calibration of ZicZac exact-score probabilities.

Fit a single score-grid temperature using only pre-2026 matches; evaluate
unchanged on 2026 matches. Never use the held-out results for fitting.
No live model is modified by this experiment.
"""
import argparse
import json
import math
from collections import defaultdict
from datetime import datetime
import numpy as np
from engine import predict, finished, _date

TEMPERATURES = (0.70, 0.80, 0.90, 1.00, 1.10, 1.20, 1.35, 1.50)
EPS = 1e-12

def calibrate(matrix, temperature):
    if temperature <= 0:
        raise ValueError("Temperature must be positive")
    logits = np.log(np.maximum(matrix, EPS)) / temperature
    logits -= logits.max()
    adjusted = np.exp(logits)
    return adjusted / adjusted.sum()

def metrics(predictions, temperature):
    n = len(predictions)
    if not n:
        return {"evaluated": 0}
    hits = top3 = outcome_hits = 0
    log_loss = brier = 0.0
    for matrix, home, away in predictions:
        grid = calibrate(matrix, temperature)
        flat = np.argsort(grid.ravel())[::-1]
        actual_index = home * grid.shape[1] + away if home < grid.shape[0] and away < grid.shape[1] else -1
        hits += int(flat[0] == actual_index)
        top3 += int(actual_index in flat[:3])
        p = float(grid[home, away]) if actual_index >= 0 else EPS
        log_loss -= math.log(max(p, EPS))
        probs = [float(np.tril(grid, -1).sum()), float(np.trace(grid)), float(np.triu(grid, 1).sum())]
        outcome = 0 if home > away else 1 if home == away else 2
        outcome_hits += int(int(np.argmax(probs)) == outcome)
        brier += sum((v - int(i == outcome)) ** 2 for i, v in enumerate(probs))
    return {"evaluated": n, "exact_hit_rate": hits/n, "top3_hit_rate": top3/n,
            "outcome_accuracy": outcome_hits/n, "outcome_brier": brier/n,
            "score_log_loss": log_loss/n}

def evaluate(matches, cutoff_year=2026, min_history=30):
    ordered = sorted((m for m in matches if finished(m) and _date(m.get("utcDate"))),
                     key=lambda m: _date(m["utcDate"]))
    by_league = defaultdict(list)
    train, test = [], []
    for match in ordered:
        code = (match.get("competition") or {}).get("code")
        previous = by_league[code]
        if len(previous) >= min_history:
            forecast = predict(match, previous, variant="baseline")
            if forecast is not None:
                actual = match["score"]["fullTime"]
                record = (forecast["matrix"], actual["home"], actual["away"])
                (train if _date(match["utcDate"]).year < cutoff_year else test).append(record)
        # Matches at the same kickoff must not enter each other's history.
        previous.append(match)
    if not train or not test:
        raise ValueError("Both pre-cutoff training and post-cutoff test fixtures are required")
    train_scores = {str(t): metrics(train, t) for t in TEMPERATURES}
    best = min(TEMPERATURES, key=lambda t: (train_scores[str(t)]["score_log_loss"], abs(t-1)))
    return {"cutoff_year": cutoff_year, "training_matches": len(train),
            "holdout_matches": len(test), "selection_metric": "training score log loss",
            "candidate_temperatures": list(TEMPERATURES),
            "training": train_scores, "chosen_temperature": best,
            "holdout": {"baseline": metrics(test, 1.0),
                        "calibrated": metrics(test, best)},
            "note": "One global temperature chosen solely on pre-cutoff matches; 2026 held out. Historical training predictions are chronological. This is experimental and not deployed."}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("matches_json")
    parser.add_argument("--cutoff-year", type=int, default=2026)
    parser.add_argument("--output", default="calibration_results.json")
    args = parser.parse_args()
    with open(args.matches_json, encoding="utf-8") as f:
        matches = json.load(f)
    report = evaluate(matches, cutoff_year=args.cutoff_year)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps({"training_matches": report["training_matches"],
                      "holdout_matches": report["holdout_matches"],
                      "chosen_temperature": report["chosen_temperature"],
                      "holdout": report["holdout"]}, indent=2))
