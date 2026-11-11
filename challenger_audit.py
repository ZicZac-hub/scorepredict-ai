"""Paired, chronological challenger audit for exact-score predictions.

2025 selects among predeclared candidates by exact-score log loss.
2026 compares the frozen choice against Baseline, including paired wins,
losses and uncertainty. Never auto-promote based on this holdout.
"""
import argparse
import json
import math
from collections import defaultdict, Counter
from engine import predict, finished, _date

CANDIDATES=("baseline","venue_arithmetic","team_strength","venue_contrast")
EPS=1e-12

def metric(rows, variant):
    n=len(rows)
    hits=top3=loss=one_one=0
    for row in rows:
        p=row["predictions"][variant]
        h,a=row["actual"]
        hits+=int(tuple(p["top"])==(h,a))
        top3+=int((h,a) in p["top3"])
        one_one+=int(tuple(p["top"])==(1,1))
        loss-=math.log(max(p["actual_probability"],EPS))
    return {"matches":n,"exact_hits":hits,"exact_hit_rate":hits/n if n else None,
        "top3_hit_rate":top3/n if n else None,
        "predicted_1_1_share":one_one/n if n else None,
        "score_log_loss":loss/n if n else None}

def evaluate(matches, calibration_year=2025, holdout_year=2026,min_history=30):
    ordered=sorted((m for m in matches if finished(m) and _date(m.get("utcDate"))),
                   key=lambda m:_date(m["utcDate"]))
    history=defaultdict(list)
    by_year=defaultdict(list)
    for fixture in ordered:
        code=(fixture.get("competition") or {}).get("code") or "unknown"
        kickoff=_date(fixture["utcDate"])
        if kickoff.year in (calibration_year,holdout_year):
            past=[m for m in history[code] if _date(m["utcDate"])<kickoff]
            if len(past)>=min_history:
                h=fixture["score"]["fullTime"]["home"]
                a=fixture["score"]["fullTime"]["away"]
                forecasts={}
                for variant in CANDIDATES:
                    f=predict(fixture,past,variant=variant)
                    if not f:break
                    matrix=f["matrix"]
                    forecasts[variant]={
                        "top":list(f["scores"][0][:2]),
                        "top3":[(x,y) for x,y,_ in f["scores"][:3]],
                        "actual_probability":float(matrix[h,a]) if h<matrix.shape[0] and a<matrix.shape[1] else EPS}
                if len(forecasts)==len(CANDIDATES):
                    by_year[kickoff.year].append({"competition":code,"actual":(h,a),"predictions":forecasts})
        history[code].append(fixture)
    train=by_year[calibration_year]
    test=by_year[holdout_year]
    if not train or not test:
        return {"error":"Missing chronological calibration or holdout","calibration_matches":len(train),"holdout_matches":len(test)}
    train_metrics={v:metric(train,v) for v in CANDIDATES}
    # Predeclared proper scoring rule; prefer incumbent when effectively tied.
    selected=min(CANDIDATES,key=lambda v:(train_metrics[v]["score_log_loss"],
                                         0 if v=="baseline" else 1))
    test_metrics={v:metric(test,v) for v in CANDIDATES}
    challenger_wins=baseline_wins=both_hits=both_miss=0
    for row in test:
        actual=tuple(row["actual"])
        b=tuple(row["predictions"]["baseline"]["top"])==actual
        c=tuple(row["predictions"][selected]["top"])==actual
        challenger_wins+=int(c and not b)
        baseline_wins+=int(b and not c)
        both_hits+=int(b and c)
        both_miss+=int(not b and not c)
    discordant=challenger_wins+baseline_wins
    # Exact two-sided McNemar binomial p-value, no external dependencies.
    p_value=min(1.0,2*sum(math.comb(discordant,k) for k in range(min(challenger_wins,baseline_wins)+1))/2**discordant) if discordant else 1.0
    return {"calibration_year":calibration_year,"holdout_year":holdout_year,
        "selected_variant":selected,"calibration":train_metrics,"holdout":test_metrics,
        "paired_comparison":{"selected_only_hits":challenger_wins,
            "baseline_only_hits":baseline_wins,"both_hit":both_hits,"neither_hit":both_miss,
            "mcnemar_exact_two_sided_p":p_value},
        "promotion_recommended":False,
        "note":"2025 selected candidate by log loss, 2026 evaluated once. No automatic promotion; independent future pre-kickoff snapshots required. 50/71 remains aspirational."}

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("matches_json")
    parser.add_argument("--output",default="challenger_audit_results.json")
    args=parser.parse_args()
    with open(args.matches_json,encoding="utf-8") as f:matches=json.load(f)
    result=evaluate(matches)
    with open(args.output,"w",encoding="utf-8") as f:json.dump(result,f,indent=2)
    print(json.dumps(result,indent=2))
    if "error" in result:raise SystemExit(result["error"])
