"""Read-only chronological v8 versus V12 exact-score calibration comparison.

Usage:
  python v12_calibration_benchmark.py work/verified_history.json
  python v12_calibration_benchmark.py work/verified_history.json --year 2026

Input is a local list of normalized provider/competition qualified FINISHED
matches. Nothing is downloaded or written to the immutable forecast ledger.
No calibration parameter is selected using these evaluation results.
"""
import argparse
import json
import math
from collections import Counter
from datetime import timedelta
from pathlib import Path
import numpy as np

import engine
from research_ensemble_v8 import predict as v8_predict
from score_calibration_v12 import (
    _league_rows, calibrate_matrix, MIN_MATCHES, VERSION)


def measures(matrix, actual):
    grid = np.asarray(matrix, dtype=float)
    h, a = actual
    probability = float(grid[h,a]) if h<grid.shape[0] and a<grid.shape[1] else 0.0
    top = tuple(map(int,np.unravel_index(int(grid.argmax()),grid.shape)))
    indices = np.argsort(-grid.ravel(),kind="stable")[:3]
    three = [tuple(map(int,np.unravel_index(int(i),grid.shape))) for i in indices]
    predicted = [float(np.tril(grid,-1).sum()),float(np.trace(grid)),
                 float(np.triu(grid,1).sum())]
    result = 0 if h>a else 1 if h==a else 2
    return {
        "hit":int(top==(h,a)), "top3_hit":int((h,a) in three),
        "score_logloss":-math.log(max(1e-12,probability)),
        "outcome_brier":sum((p-int(index==result))**2
                            for index,p in enumerate(predicted)),
        "modal_score":f"{top[0]}-{top[1]}",
        "assigned_actual_score_probability":probability,
    }


def _summary(records):
    if not records:
        return {"count":0,"exact_hits":0,"exact_rate":None,
                "top3_rate":None,"mean_score_logloss":None,"mean_outcome_brier":None,
                "modal_1_1_share":None}
    n=len(records)
    return {
        "count":n,
        "exact_hits":sum(x["hit"] for x in records),
        "exact_rate":sum(x["hit"] for x in records)/n,
        "top3_rate":sum(x["top3_hit"] for x in records)/n,
        "mean_score_logloss":sum(x["score_logloss"] for x in records)/n,
        "mean_outcome_brier":sum(x["outcome_brier"] for x in records)/n,
        "modal_1_1_share":sum(x["modal_score"]=="1-1" for x in records)/n,
        "score_modes":dict(Counter(x["modal_score"] for x in records).most_common(8)),
    }


def evaluate(matches, *, year=2026, limit=200):
    """Evaluate a deterministic prefix of eligible future-masked fixtures."""
    verified=[m for m in matches if engine.finished(m) and
              engine._date(m.get("utcDate")) is not None]
    verified.sort(key=lambda m:(engine._date(m["utcDate"]),
                               str(m.get("provider")),str(m.get("id"))))
    comparisons=[]
    checked=0
    for fixture in verified:
        kickoff=engine._date(fixture["utcDate"])
        if kickoff.year!=year:
            continue
        cutoff=kickoff-timedelta(hours=1)
        # A historical record must have enough prior complete league evidence
        # to be comparable. No same-fixture result or future result is included.
        league_prior=_league_rows(fixture,verified,cutoff)
        if len(league_prior)<MIN_MATCHES:
            continue
        checked+=1
        base=v8_predict(fixture,verified,league_history=verified,as_of=cutoff)
        if base is None:
            continue
        candidate=calibrate_matrix(base["matrix"],league_prior,as_of=cutoff)
        if candidate is None:
            continue
        actual=fixture["score"]["fullTime"]
        true=(actual["home"],actual["away"])
        old=measures(base["matrix"],true)
        new=measures(candidate[0],true)
        comparisons.append({
            "fixture_id":str(fixture.get("id")),
            "provider":fixture.get("provider"),
            "competition":(fixture.get("competition") or {}).get("code"),
            "kickoff":kickoff.isoformat(),
            "history_matches":len(league_prior),
            "actual":list(true),
            "v8":old,"v12":new
        })
        if len(comparisons)>=limit:
            break
    old=_summary([r["v8"] for r in comparisons])
    new=_summary([r["v12"] for r in comparisons])
    v12_only=sum(r["v12"]["hit"] and not r["v8"]["hit"] for r in comparisons)
    v8_only=sum(r["v8"]["hit"] and not r["v12"]["hit"] for r in comparisons)
    return {
        "model":"v12 shadow versus same-time v8",
        "challenger_version":VERSION,"holdout_year":year,
        "status":"historical_reconstruction_not_prospective",
        "selection":"first qualifying matches by chronological kickoff, not by outcome",
        "prediction_cutoff":"one hour before each kickoff",
        "result_available_after":"finishedAt if given, otherwise kickoff plus 3 hours",
        "parameters_selected_on_holdout":False,
        "original_predictions_modified":False,
        "eligible_evaluated":len(comparisons),
        "eligible_checked":checked,
        "baseline_v8":old,"challenger_v12":new,
        "paired_v12_only_exact_hits":v12_only,
        "paired_v8_only_exact_hits":v8_only,
        "promotion_approved":False,
        "note":("Same-fixture chronological reconstruction, not prospective "
                "accuracy. More varied modes cannot justify promotion. "
                "Require holdout probability improvement and separately "
                "frozen future predictions before changing production."),
        "records":comparisons,
    }


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input",type=Path)
    parser.add_argument("--year",type=int,default=2026)
    parser.add_argument("--limit",type=int,default=200)
    parser.add_argument("--output",type=Path,default=Path("work/v12_calibration_holdout.json"))
    args=parser.parse_args()
    if args.limit<1:
        parser.error("--limit must be positive")
    data=json.loads(args.input.read_text(encoding="utf-8"))
    matches=data if isinstance(data,list) else data.get("matches")
    if not isinstance(matches,list):
        parser.error("Expected a list of normalized finished match records")
    result=evaluate(matches,year=args.year,limit=args.limit)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({k:v for k,v in result.items() if k!="records"},indent=2))


if __name__=="__main__":
    main()
