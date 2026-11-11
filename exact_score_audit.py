"""Audit original predictions and compare leakage-safe low-score corrections.

The 2025 calibration selects Dixon-Coles rho; the 2026 holdout is never
consulted for selection. The live model and original predictions are unchanged.
"""
import argparse
import json
import math
from collections import Counter, defaultdict
import numpy as np
from engine import predict, finished, _date, _grid

RHOS = (-0.20, -0.15, -0.10, -0.05, 0.0, 0.05, 0.10)
EPS = 1e-12

def score_metrics(rows, rho):
    n=len(rows)
    if not n:return {"evaluated":0}
    hits=top3=outcomes=one_one=0
    loss=brier=0.0
    predicted=Counter()
    actual=Counter()
    misses=Counter()
    for home_xg,away_xg,h,a in rows:
        matrix=_grid(home_xg,away_xg,rho=rho)
        ranked=sorted(((i,j,float(matrix[i,j])) for i in range(matrix.shape[0])
                       for j in range(matrix.shape[1])),key=lambda x:x[2],reverse=True)
        ph,pa,_=ranked[0]
        hits+=int((ph,pa)==(h,a))
        top3+=int(any((i,j)==(h,a) for i,j,_ in ranked[:3]))
        one_one+=int((ph,pa)==(1,1))
        predicted[f"{ph}-{pa}"]+=1
        actual[f"{h}-{a}"]+=1
        if (ph,pa)!=(h,a):misses[f"{ph}-{pa}"]+=1
        probability=float(matrix[h,a]) if h<matrix.shape[0] and a<matrix.shape[1] else EPS
        loss-=math.log(max(probability,EPS))
        probs=(float(np.tril(matrix,-1).sum()),float(np.trace(matrix)),
               float(np.triu(matrix,1).sum()))
        winner=0 if h>a else 1 if h==a else 2
        outcomes+=int(int(np.argmax(probs))==winner)
        brier+=sum((p-int(i==winner))**2 for i,p in enumerate(probs))
    return {"evaluated":n,"exact_hits":hits,"exact_hit_rate":hits/n,
        "top3_hit_rate":top3/n,"outcome_accuracy":outcomes/n,
        "predicted_1_1_share":one_one/n,"score_log_loss":loss/n,
        "outcome_brier":brier/n,
        "predicted_score_distribution":dict(predicted.most_common(12)),
        "actual_score_distribution":dict(actual.most_common(12)),
        "most_common_missed_predictions":dict(misses.most_common(10))}

def evaluate(matches, calibration_year=2025,holdout_year=2026,min_history=30):
    ordered=sorted((m for m in matches if finished(m) and _date(m.get("utcDate"))),
                   key=lambda m:_date(m["utcDate"]))
    histories=defaultdict(list)
    train=[]
    test=[]
    per_league=defaultdict(list)
    for fixture in ordered:
        league=(fixture.get("competition") or {}).get("code") or "unknown"
        kickoff=_date(fixture["utcDate"])
        if kickoff.year in (calibration_year,holdout_year):
            past=[m for m in histories[league] if _date(m["utcDate"])<kickoff]
            if len(past)>=min_history:
                forecast=predict(fixture,past,variant="baseline")
                if forecast:
                    full=fixture["score"]["fullTime"]
                    row=(forecast["home_xg"],forecast["away_xg"],full["home"],full["away"])
                    if kickoff.year==calibration_year:train.append(row)
                    else:
                        test.append(row)
                        per_league[league].append(row)
        histories[league].append(fixture)
    if not train or not test:
        return {"error":"Insufficient chronological calibration or holdout matches",
                "calibration_matches":len(train),"holdout_matches":len(test)}
    train_metrics={str(rho):score_metrics(train,rho) for rho in RHOS}
    chosen=min(RHOS,key=lambda rho:(train_metrics[str(rho)]["score_log_loss"],abs(rho)))
    return {"calibration_year":calibration_year,"holdout_year":holdout_year,
        "selected_rho":chosen,"calibration_candidates":train_metrics,
        "holdout_baseline":score_metrics(test,0.0),
        "holdout_selected":score_metrics(test,chosen),
        "holdout_by_league":{league:{"baseline":score_metrics(rows,0.0),
            "candidate":score_metrics(rows,chosen)} for league,rows in sorted(per_league.items())},
        "target_50_of_71":{"target_hits":50,"sample_matches":71,
            "required_accuracy":50/71,
            "note":"Aspirational goal only, not a model claim or fitted target."},
        "note":"Only 2025 selects rho by exact-score log loss. 2026 is untouched until evaluation. No post-kickoff features; no production promotion."}

def audit_saved_snapshots():
    from persist_prematch import earliest_snapshots,load_json,RESULTS
    snapshots=earliest_snapshots()
    results=load_json(RESULTS,{})
    rows=[]
    for fid,original in snapshots.items():
        final=results.get(str(fid))
        if not final:continue
        pred=original.get("predicted_score")
        if not isinstance(pred,list) or len(pred)!=2:continue
        actual=[final["home_goals"],final["away_goals"]]
        rows.append({"fixture_id":fid,"league":original.get("competition"),
            "kickoff_utc":original.get("kickoff_utc"),
            "original_prediction":pred,"verified_actual":actual,
            "exact_hit":pred==actual,
            "predicted_1_1":pred==[1,1]})
    return {"verified_original_snapshots":len(rows),
        "exact_hits":sum(x["exact_hit"] for x in rows),
        "predicted_1_1_count":sum(x["predicted_1_1"] for x in rows),
        "rows":rows,
        "note":"Original immutable pre-kickoff predictions only. Local Streamlit SQLite records are not available to GitHub Actions."}

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("matches_json")
    parser.add_argument("--output",default="exact_score_audit_results.json")
    args=parser.parse_args()
    with open(args.matches_json,encoding="utf-8") as source:
        matches=json.load(source)
    report=evaluate(matches)
    report["saved_snapshot_audit"]=audit_saved_snapshots()
    with open(args.output,"w",encoding="utf-8") as target:
        json.dump(report,target,indent=2)
    print(json.dumps({k:v for k,v in report.items() if k not in ("calibration_candidates","holdout_by_league")},indent=2))
    if "error" in report:raise SystemExit(report["error"])
