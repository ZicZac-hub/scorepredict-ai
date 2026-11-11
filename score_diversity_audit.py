"""Audit repeated exact-score modes without manipulating predictions.

Poisson score modes often cluster at 1-1 when both expected-goal rates
are between 1 and 2. Report frequency, entropy and actual accuracy.
"""
import argparse
import json
import math
from collections import Counter, defaultdict
from engine import predict, finished, _date

def summarize(rows):
    counts = Counter(row["score"] for row in rows)
    n = len(rows)
    return {
        "predictions": n,
        "top_score_frequencies": dict(counts.most_common(10)),
        "share_predicted_1_1": counts["1-1"] / n if n else None,
        "unique_top_scores": len(counts),
        "mean_top_score_probability": sum(row["prob"] for row in rows)/n if n else None,
        "exact_score_hit_rate": sum(row.get("hit", 0) for row in rows)/n if n else None,
        "one_one_subgroup": {
            "count": counts["1-1"],
            "hits": sum(row.get("hit",0) for row in rows if row["score"] == "1-1"),
            "accuracy": (sum(row.get("hit",0) for row in rows if row["score"] == "1-1")/counts["1-1"]) if counts["1-1"] else None
        }
    }

def historical_audit(matches, cutoff_year=2026, min_history=30):
    ordered = sorted((m for m in matches if finished(m) and _date(m.get("utcDate"))),
                     key=lambda m: _date(m["utcDate"]))
    histories = defaultdict(list)
    data = {"baseline": [], "venue_arithmetic": [], "team_strength": [], "opponent_adjusted": []}
    for fixture in ordered:
        code = (fixture.get("competition") or {}).get("code") or "unknown"
        kickoff = _date(fixture["utcDate"])
        past = [m for m in histories[code] if _date(m["utcDate"]) < kickoff]
        if kickoff.year >= cutoff_year and len(past) >= min_history:
            actual = fixture["score"]["fullTime"]
            for variant in data:
                result = predict(fixture, past, variant=variant)
                if not result:
                    continue
                h, a, p = result["scores"][0]
                data[variant].append({"score": f"{h}-{a}", "prob": p,
                                      "hit": int((h,a)==(actual["home"],actual["away"]))})
        histories[code].append(fixture)
    return {variant: summarize(rows) for variant, rows in data.items()}

def snapshot_audit(paths):
    rows = []
    for path in paths:
        with open(path, encoding="utf-8") as source:
            for item in json.load(source).get("records", []):
                score = item.get("predicted_score") or []
                if len(score) != 2:
                    continue
                rows.append({"score": f"{score[0]}-{score[1]}",
                             "prob": item.get("predicted_score_probability", 0)})
    return summarize(rows)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("matches_json")
    parser.add_argument("--output", default="score_diversity_results.json")
    args = parser.parse_args()
    with open(args.matches_json, encoding="utf-8") as source:
        matches = json.load(source)
    report = {
        "historical_2026_holdout": historical_audit(matches),
        "interpretation": "Compare 1-1 frequency and accuracy on the same chronological holdout. Higher diversity alone is not evidence of better predictions. Production baseline unchanged."
    }
    with open(args.output, "w", encoding="utf-8") as target:
        json.dump(report, target, indent=2)
    print(json.dumps(report, indent=2))
