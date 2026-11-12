"""Chronological holdout comparison of live baseline and strength contrast.

Use only matches whose results were not available at prediction time.
"""
import argparse
import json
import math
from collections import defaultdict
from engine import predict, finished, _date

def evaluate(matches, cutoff_year=2026, min_history=30):
    ordered=sorted((m for m in matches if finished(m) and _date(m.get("utcDate"))),
                   key=lambda m:_date(m["utcDate"]))
    leagues=defaultdict(list)
    rows=[]
    for m in ordered:
        code=(m.get("competition") or {}).get("code")
        history=leagues[code]
        if _date(m["utcDate"]).year>=cutoff_year and len(history)>=min_history:
            baseline=predict(m,history,variant="baseline")
            challenger=predict(m,history,variant="strength_contrast_candidate")
            if baseline and challenger:
                actual=m["score"]["fullTime"]
                rows.append((baseline,challenger,actual["home"],actual["away"]))
        # Exclude same-kickoff fixtures from each other's training history.
        history.append(m)
    def metrics(index):
        n=len(rows)
        hits=top3=0
        loss=0.0
        for row in rows:
            forecast=row[index]
            h,a=row[2:]
            scores=forecast["scores"]
            hits+=int((scores[0][0],scores[0][1])==(h,a))
            top3+=int(any((x,y)==(h,a) for x,y,_ in scores[:3]))
            matrix=forecast["matrix"]
            prob=float(matrix[h,a]) if h<matrix.shape[0] and a<matrix.shape[1] else 1e-12
            loss-=math.log(max(1e-12,prob))
        return {"matches":n,"top1_hits":hits,"top1_rate":hits/n if n else None,
                "top3_hits":top3,"top3_rate":top3/n if n else None,
                "score_log_loss":loss/n if n else None}
    return {"cutoff_year":cutoff_year,"baseline":metrics(0),
            "strength_contrast_candidate":metrics(1),
            "note":"No tuning on held-out outcomes. Experimental only; no automatic promotion."}

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("matches_json")
    parser.add_argument("--cutoff-year",type=int,default=2026)
    parser.add_argument("--output",default="strength_contrast_results.json")
    args=parser.parse_args()
    with open(args.matches_json,encoding="utf-8") as f:
        matches=json.load(f)
    report=evaluate(matches,args.cutoff_year)
    with open(args.output,"w",encoding="utf-8") as f:
        json.dump(report,f,indent=2)
    print(json.dumps(report,indent=2))
