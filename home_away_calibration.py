"""Calibrate home and away goal rates independently, without holdout leakage.

Select multiplicative home/away adjustments on 2025 matches only using
proper exact-score log loss. Evaluate the frozen pair on 2026 matches.
The live prediction engine is not changed.
"""
import argparse
import json
import math
from collections import defaultdict
import numpy as np
from engine import predict, finished, _date, _grid

SCALES = (0.80, 0.90, 0.95, 1.0, 1.05, 1.10, 1.20)
EPS = 1e-12

def evaluate_rows(rows, home_scale, away_scale):
    n = len(rows)
    if not n:
        return {"evaluated": 0}
    exact = top3 = outcome = one_one = 0
    logloss = brier = 0.0
    for home_xg, away_xg, h, a in rows:
        matrix = _grid(max(.25, home_xg * home_scale),
                       max(.25, away_xg * away_scale))
        ranked = sorted(((i, j, float(matrix[i,j])) for i in range(matrix.shape[0])
                         for j in range(matrix.shape[1])), key=lambda item:item[2], reverse=True)
        best = ranked[0]
        exact += int((best[0], best[1]) == (h,a))
        top3 += int(any((i,j)==(h,a) for i,j,_ in ranked[:3]))
        one_one += int((best[0],best[1]) == (1,1))
        logloss -= math.log(max(float(matrix[h,a]) if h < matrix.shape[0] and a < matrix.shape[1] else EPS, EPS))
        probs = (float(np.tril(matrix,-1).sum()), float(np.trace(matrix)),
                 float(np.triu(matrix,1).sum()))
        actual = 0 if h>a else 1 if h==a else 2
        outcome += int(max(range(3),key=lambda i:probs[i]) == actual)
        brier += sum((p-int(i==actual))**2 for i,p in enumerate(probs))
    return {"evaluated":n,"exact_hits":exact,"exact_hit_rate":exact/n,
            "top3_hit_rate":top3/n,"outcome_accuracy":outcome/n,
            "predicted_1_1_share":one_one/n,"score_log_loss":logloss/n,
            "outcome_brier":brier/n}

def evaluate(matches, calibration_year=2025, holdout_year=2026, min_history=30):
    ordered=sorted((m for m in matches if finished(m) and _date(m.get("utcDate"))),
                   key=lambda m:_date(m["utcDate"]))
    histories=defaultdict(list)
    calibration, holdout=[],[]
    for fixture in ordered:
        code=(fixture.get("competition") or {}).get("code") or "unknown"
        kickoff=_date(fixture["utcDate"])
        if kickoff.year in (calibration_year,holdout_year):
            past=[m for m in histories[code] if _date(m["utcDate"])<kickoff]
            if len(past)>=min_history:
                forecast=predict(fixture,past,variant="baseline")
                if forecast:
                    goals=fixture["score"]["fullTime"]
                    row=(forecast["home_xg"],forecast["away_xg"],goals["home"],goals["away"])
                    (calibration if kickoff.year==calibration_year else holdout).append(row)
        histories[code].append(fixture)
    if not calibration or not holdout:
        return {"error":"Insufficient chronological calibration or holdout matches",
                "calibration_matches":len(calibration),"holdout_matches":len(holdout)}
    candidates=[]
    for home_scale in SCALES:
        for away_scale in SCALES:
            m=evaluate_rows(calibration,home_scale,away_scale)
            candidates.append({"home_scale":home_scale,"away_scale":away_scale,**m})
    # Calibration metric fixed in advance: proper exact-score log loss.
    # Equal log loss favours minimal deviation from live baseline.
    chosen=min(candidates,key=lambda x:(x["score_log_loss"],
        abs(x["home_scale"]-1)+abs(x["away_scale"]-1)))
    h,a=chosen["home_scale"],chosen["away_scale"]
    return {"calibration_year":calibration_year,"holdout_year":holdout_year,
            "calibration_matches":len(calibration),"holdout_matches":len(holdout),
            "selected_home_scale":h,"selected_away_scale":a,
            "calibration_baseline":evaluate_rows(calibration,1.0,1.0),
            "calibration_selected":evaluate_rows(calibration,h,a),
            "holdout_baseline":evaluate_rows(holdout,1.0,1.0),
            "holdout_selected":evaluate_rows(holdout,h,a),
            "calibration_grid":candidates,
            "note":"Home/away scales selected only on 2025 score log loss. 2026 holdout not used for selection. No live model change."}

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("matches_json")
    parser.add_argument("--output",default="home_away_calibration_results.json")
    args=parser.parse_args()
    with open(args.matches_json,encoding="utf-8") as source:
        matches=json.load(source)
    report=evaluate(matches)
    with open(args.output,"w",encoding="utf-8") as target:
        json.dump(report,target,indent=2)
    print(json.dumps({k:v for k,v in report.items() if k!="calibration_grid"},indent=2))
    if "error" in report:
        raise SystemExit(report["error"])

