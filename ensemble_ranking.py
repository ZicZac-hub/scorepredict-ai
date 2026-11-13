"""Chronological mixture-of-score-probabilities experiment.

Choose weights by 2025 score log loss, freeze before testing 2026.
Never alter historical predictions or auto-promote to production.
"""
import argparse,json,math
from collections import defaultdict
import numpy as np
from engine import predict,finished,_date,mix_score_matrices

VARIANTS=("baseline","team_strength","league_opponent")
WEIGHTS=((1.,0.,0.),(.75,.25,0.),(.5,.5,0.),(.75,0.,.25),(.5,0.,.5),
         (.5,.25,.25),(.34,.33,.33),(.25,.5,.25),(.25,.25,.5))
EPS=1e-12

def summarize(rows,weight):
    hits=top3=loss=one_one=0
    n=len(rows)
    for row in rows:
        matrix=mix_score_matrices([row["matrices"][v] for v in VARIANTS],weight)
        ranked=np.argsort(-matrix.ravel(),kind="stable")
        h,a=row["actual"]
        actual=h*matrix.shape[1]+a
        hits+=int(ranked[0]==actual)
        top3+=int(actual in ranked[:3])
        one_one+=int(ranked[0]==matrix.shape[1]+1)
        loss-=math.log(max(float(matrix[h,a]) if h<matrix.shape[0] and a<matrix.shape[1] else EPS,EPS))
    return {"matches":n,"exact_hits":hits,"exact_accuracy":hits/n if n else None,
            "top3_hits":top3,"top3_accuracy":top3/n if n else None,
            "predicted_1_1_share":one_one/n if n else None,
            "score_log_loss":loss/n if n else None}

def evaluate(matches,calibration_year=2025,holdout_year=2026,min_history=30):
    ordered=sorted((m for m in matches if finished(m) and _date(m.get("utcDate"))),
                   key=lambda m:_date(m["utcDate"]))
    history=defaultdict(list)
    years=defaultdict(list)
    for fixture in ordered:
        code=(fixture.get("competition") or {}).get("code") or "unknown"
        kickoff=_date(fixture["utcDate"])
        if kickoff.year in (calibration_year,holdout_year):
            past=[m for m in history[code] if _date(m["utcDate"])<kickoff]
            if len(past)>=min_history:
                matrices={}
                for variant in VARIANTS:
                    prediction=predict(fixture,past,variant=variant)
                    if prediction is not None:matrices[variant]=prediction["matrix"]
                if len(matrices)==len(VARIANTS):
                    actual=fixture["score"]["fullTime"]
                    years[kickoff.year].append({"actual":(actual["home"],actual["away"]),
                                               "matrices":matrices,"league":code})
        history[code].append(fixture)
    train,test=years[calibration_year],years[holdout_year]
    if not train or not test:
        return {"error":"Insufficient calibration or holdout fixtures",
                "calibration_matches":len(train),"holdout_matches":len(test)}
    candidates=[{"weights":dict(zip(VARIANTS,weights)),
                 "calibration":summarize(train,weights)} for weights in WEIGHTS]
    # Proper scoring rule on calibration only, never select on holdout.
    best=min(range(len(WEIGHTS)),key=lambda i:(candidates[i]["calibration"]["score_log_loss"],i))
    baseline=summarize(test,WEIGHTS[0])
    selected=summarize(test,WEIGHTS[best])
    wins=losses=both=neither=0
    for row in test:
        b=row["matrices"]["baseline"]
        m=mix_score_matrices([row["matrices"][v] for v in VARIANTS],WEIGHTS[best])
        actual=row["actual"]
        bh=tuple(np.unravel_index(np.argmax(b),b.shape))==actual
        mh=tuple(np.unravel_index(np.argmax(m),m.shape))==actual
        wins+=int(mh and not bh);losses+=int(bh and not mh)
        both+=int(bh and mh);neither+=int(not bh and not mh)
    d=wins+losses
    p=min(1.,2*sum(math.comb(d,k) for k in range(min(wins,losses)+1))/2**d) if d else 1.
    return {"calibration_year":calibration_year,"holdout_year":holdout_year,
            "calibration_candidates":candidates,
            "selected_weights":dict(zip(VARIANTS,WEIGHTS[best])),
            "holdout_baseline":baseline,"holdout_selected":selected,
            "paired":{"ensemble_only_hits":wins,"baseline_only_hits":losses,
                      "both_hits":both,"neither_hits":neither,
                      "mcnemar_two_sided_p":p},
            "promotion_recommended":False,
            "note":"Weights selected exclusively by 2025 score log loss. 2026 is evaluation only; original predictions and live model remain unchanged. A future independent prospective test is needed."}

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("matches_json")
    p.add_argument("--output",default="ensemble_ranking_results.json")
    args=p.parse_args()
    with open(args.matches_json,encoding="utf-8") as f:matches=json.load(f)
    report=evaluate(matches)
    with open(args.output,"w",encoding="utf-8") as f:json.dump(report,f,indent=2)
    print(json.dumps({k:v for k,v in report.items() if k!="calibration_candidates"},indent=2))
    if "error" in report:raise SystemExit(report["error"])

