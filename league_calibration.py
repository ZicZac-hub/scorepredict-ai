"""League-specific home/away xG calibration, selected on 2025 only.

Global fallback and league coefficients are fitted without seeing 2026
results. Small league samples shrink towards the global coefficient.
"""
import argparse
import json
from collections import defaultdict
from engine import predict, finished, _date
from home_away_calibration import SCALES, evaluate_rows

MIN_LEAGUE_CALIBRATION = 40
SHRINKAGE_MATCHES = 100

def best_scales(rows):
    candidates = []
    for home in SCALES:
        for away in SCALES:
            metric = evaluate_rows(rows, home, away)
            candidates.append((metric["score_log_loss"],
                               abs(home-1.0)+abs(away-1.0), home, away))
    _, _, home, away = min(candidates)
    return home, away

def aggregate(groups, coefficients):
    """Combine metrics weighted by evaluated fixture counts."""
    totals = defaultdict(float)
    for league, rows in groups.items():
        home, away = coefficients.get(league, coefficients["__global__"])
        metric = evaluate_rows(rows, home, away)
        n = metric["evaluated"]
        totals["evaluated"] += n
        totals["exact_hits"] += metric["exact_hits"]
        for key in ("top3_hit_rate","outcome_accuracy","predicted_1_1_share",
                    "score_log_loss","outcome_brier"):
            totals[key] += metric[key] * n
    n = totals["evaluated"]
    if not n:
        return {"evaluated": 0}
    return {"evaluated": int(n), "exact_hits": int(totals["exact_hits"]),
            "exact_hit_rate": totals["exact_hits"]/n,
            **{key: totals[key]/n for key in ("top3_hit_rate","outcome_accuracy",
               "predicted_1_1_share","score_log_loss","outcome_brier")}}

def evaluate(matches, calibration_year=2025, holdout_year=2026, min_history=30):
    ordered = sorted((m for m in matches if finished(m) and _date(m.get("utcDate"))),
                     key=lambda m: _date(m["utcDate"]))
    histories = defaultdict(list)
    calibration, holdout = defaultdict(list), defaultdict(list)
    for fixture in ordered:
        league = (fixture.get("competition") or {}).get("code") or "unknown"
        kickoff = _date(fixture["utcDate"])
        if kickoff.year in (calibration_year, holdout_year):
            past = [m for m in histories[league] if _date(m["utcDate"]) < kickoff]
            if len(past) >= min_history:
                forecast = predict(fixture, past, variant="baseline")
                if forecast:
                    score = fixture["score"]["fullTime"]
                    row = (forecast["home_xg"], forecast["away_xg"],
                           score["home"], score["away"])
                    (calibration if kickoff.year == calibration_year else holdout)[league].append(row)
        histories[league].append(fixture)
    train_rows = [row for rows in calibration.values() for row in rows]
    test_rows = [row for rows in holdout.values() for row in rows]
    if not train_rows or not test_rows:
        return {"error":"Missing calibration or holdout matches",
                "calibration_matches":len(train_rows),"holdout_matches":len(test_rows)}
    global_scales = best_scales(train_rows)
    league_scales = {"__global__": global_scales}
    league_details = {}
    for league, rows in sorted(calibration.items()):
        n = len(rows)
        if n < MIN_LEAGUE_CALIBRATION:
            league_scales[league] = global_scales
            league_details[league] = {"calibration_matches":n,
                "fallback":"global (insufficient league sample)",
                "home_scale":global_scales[0],"away_scale":global_scales[1]}
            continue
        raw_home, raw_away = best_scales(rows)
        weight = n/(n+SHRINKAGE_MATCHES)
        home = global_scales[0] + weight*(raw_home-global_scales[0])
        away = global_scales[1] + weight*(raw_away-global_scales[1])
        league_scales[league] = (home, away)
        league_details[league] = {"calibration_matches":n,
            "raw_home_scale":raw_home,"raw_away_scale":raw_away,
            "shrinkage_weight":weight,"home_scale":home,"away_scale":away}
    baseline = {"__global__":(1.0,1.0)}
    global_only = {"__global__":global_scales}
    return {"calibration_year":calibration_year,"holdout_year":holdout_year,
        "calibration_matches":len(train_rows),"holdout_matches":len(test_rows),
        "global_home_scale":global_scales[0],"global_away_scale":global_scales[1],
        "league_coefficients":league_details,
        "calibration_baseline":aggregate(calibration,baseline),
        "calibration_global":aggregate(calibration,global_only),
        "calibration_league":aggregate(calibration,league_scales),
        "holdout_baseline":aggregate(holdout,baseline),
        "holdout_global":aggregate(holdout,global_only),
        "holdout_league":aggregate(holdout,league_scales),
        "holdout_by_league":{league:{
            "baseline":evaluate_rows(rows,1.0,1.0),
            "global":evaluate_rows(rows,*global_scales),
            "league":evaluate_rows(rows,*league_scales.get(league,global_scales))}
            for league,rows in sorted(holdout.items())},
        "note":"2025 selects global and league adjustments using score log loss; league adjustments shrink toward global with fixed 100-match prior. All 2026 results are reserved for final evaluation. Experimental only; no production changes."}

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("matches_json")
    parser.add_argument("--output",default="league_calibration_results.json")
    args=parser.parse_args()
    with open(args.matches_json,encoding="utf-8") as source:
        matches=json.load(source)
    report=evaluate(matches)
    with open(args.output,"w",encoding="utf-8") as target:
        json.dump(report,target,indent=2)
    print(json.dumps({k:v for k,v in report.items() if k!="holdout_by_league"},indent=2))
    if "error" in report:
        raise SystemExit(report["error"])
