"""Experimental outcome-aware five-score-double selection.

Uses five statistical variants (not independent external AI services).
Never replaces live predictions without prospective validation.
"""
from itertools import combinations
from five_model_consensus import consensus_prediction

VERSION = "outcome-aware-five-double-challenger-v1"

def candidate_forecast(fixture, history):
    result = consensus_prediction(fixture, history)
    if not result.get("available"):
        return result
    scores = result["top10_scores"]
    outcome = {"home":result["home_win"],"draw":result["draw"],"away":result["away_win"]}
    label = max(outcome, key=outcome.get)
    confidence = outcome[label]
    return {**result, "outcome_pick":label, "outcome_confidence":confidence,
            "outcome_probabilities":outcome, "scores":scores,
            "model_version":VERSION}

def rank_outcome_aware_pairs(rows, limit=5, min_outcome_confidence=0.50):
    """Rank same-day pairs by five-double coverage, filtering uncertain outcomes.

    Each row must have fixture, kickoff and an experimental candidate forecast
    in 'result'. Joint scores assume independence across the two matches.
    """
    eligible = [r for r in rows if r.get("result",{}).get("available")
                and r["result"].get("outcome_confidence",0)>=min_outcome_confidence
                and r["result"].get("scores")]
    candidates=[]
    for first,second in combinations(eligible,2):
        f1,f2=first["fixture"],second["fixture"]
        if (f1.get("provider"),f1.get("id"))==(f2.get("provider"),f2.get("id")):
            continue
        if str(first.get("kickoff",""))[:10]!=str(second.get("kickoff",""))[:10]:
            continue
        options=[]
        for h,a,p in first["result"]["scores"]:
            for h2,a2,p2 in second["result"]["scores"]:
                if p>0 and p2>0:
                    options.append({"Score 1":f"{h}-{a}","Score 2":f"{h2}-{a2}",
                                    "P1":float(p),"P2":float(p2),
                                    "Joint":float(p)*float(p2)})
        picks=sorted(options,key=lambda x:-x["Joint"])[:5]
        if len(picks)<5:
            continue
        coverage=sum(x["Joint"] for x in picks)
        candidates.append({"first":first,"second":second,"doubles":picks,
                           "coverage":coverage,"model_version":VERSION,
                           "minimum_outcome_confidence":min_outcome_confidence})
    return sorted(candidates,key=lambda x:(-x["coverage"],
        str(x["first"]["fixture"].get("id","")),
        str(x["second"]["fixture"].get("id",""))))[:limit]
