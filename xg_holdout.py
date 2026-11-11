"""Chronological holdout comparison of expected-goals estimators.

Evaluate fixed candidates on 2026 only; pre-2026 data is used solely as
historical input. No candidate is chosen or fitted on the holdout.
"""
import argparse
import json
import math
from collections import defaultdict
from engine import predict, finished, _date

VARIANTS = ("baseline", "venue_arithmetic", "team_strength", "opponent_adjusted", "venue_contrast", "league_opponent")
EPS = 1e-12

def evaluate(matches, cutoff_year=2026, min_history=30):
    ordered = sorted((m for m in matches if finished(m) and _date(m.get("utcDate"))),
                     key=lambda m: _date(m["utcDate"]))
    histories = defaultdict(list)
    counts = defaultdict(lambda: defaultdict(float))
    for fixture in ordered:
        league = (fixture.get("competition") or {}).get("code") or "unknown"
        prior = [m for m in histories[league]
                 if _date(m["utcDate"]) < _date(fixture["utcDate"])]
        if _date(fixture["utcDate"]).year >= cutoff_year and len(prior) >= min_history:
            actual = fixture["score"]["fullTime"]
            h, a = actual["home"], actual["away"]
            outcome = 0 if h > a else 1 if h == a else 2
            for variant in VARIANTS:
                result = predict(fixture, prior, variant=variant)
                if result is None:
                    continue
                matrix = result["matrix"]
                top = result["scores"]
                probability = float(matrix[h,a]) if h < matrix.shape[0] and a < matrix.shape[1] else EPS
                probs = [result["home_win"], result["draw"], result["away_win"]]
                for group in ("overall", league):
                    stat = counts[(group, variant)]
                    stat["evaluated"] += 1
                    stat["exact_hits"] += int((top[0][0],top[0][1]) == (h,a))
                    stat["top3_hits"] += int(any((x,y)==(h,a) for x,y,_ in top[:3]))
                    stat["outcome_hits"] += int(max(range(3),key=lambda i:probs[i]) == outcome)
                    stat["brier"] += sum((p-int(i==outcome))**2 for i,p in enumerate(probs))
                    stat["log_loss"] -= math.log(max(probability,EPS))
        histories[league].append(fixture)
    def summary(stat):
        n = stat["evaluated"]
        if not n:
            return {"evaluated":0}
        return {"evaluated":int(n),"exact_hit_rate":stat["exact_hits"]/n,
                "top3_hit_rate":stat["top3_hits"]/n,
                "outcome_accuracy":stat["outcome_hits"]/n,
                "outcome_brier":stat["brier"]/n,
                "score_log_loss":stat["log_loss"]/n}
    return {"holdout_start_year":cutoff_year,
            "variants":list(VARIANTS),
            "overall":{v:summary(counts[("overall",v)]) for v in VARIANTS},
            "by_league":{league:{v:summary(counts[(league,v)]) for v in VARIANTS}
                         for league in sorted({key[0] for key in counts if key[0]!="overall"})},
            "note":"Experimental fixed estimators, evaluated only on 2026 and later; earlier completed matches are used as historical context. No production model changed."}

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("matches_json")
    parser.add_argument("--output",default="xg_holdout_results.json")
    args=parser.parse_args()
    with open(args.matches_json,encoding="utf-8") as f:
        matches=json.load(f)
    report=evaluate(matches)
    with open(args.output,"w",encoding="utf-8") as f:
        json.dump(report,f,indent=2)
    print(json.dumps(report["overall"],indent=2))
