"""Experimental best-two fixture selection for five exact-score doubles.

Ranks only fixtures with real model score distributions. No invented team news,
no guaranteed win claims. All joint probabilities assume independent matches.
"""
from itertools import combinations

def five_doubles(first, second, limit=5):
    """Highest-probability distinct exact-score pairs maximize covered mass."""
    left=first.get("result",{}).get("scores",[])[:10]
    right=second.get("result",{}).get("scores",[])[:10]
    options=[]
    for h,a,p in left:
        for h2,a2,p2 in right:
            if float(p)>0 and float(p2)>0:
                options.append({"Score 1":f"{int(h)}-{int(a)}",
                                "Score 2":f"{int(h2)}-{int(a2)}",
                                "P1":float(p),"P2":float(p2),
                                "Joint":float(p)*float(p2)})
    return sorted(options,key=lambda x:(-x["Joint"],x["Score 1"],x["Score 2"]))[:limit]

def rank_pairs(rows, limit=5):
    """Compare same-day pairs by estimated top-five joint score coverage."""
    eligible=[r for r in rows if r.get("result",{}).get("scores")]
    candidates=[]
    for first,second in combinations(eligible,2):
        if first["fixture"].get("id")==second["fixture"].get("id") and first["fixture"].get("provider")==second["fixture"].get("provider"):
            continue
        if str(first.get("kickoff",""))[:10]!=str(second.get("kickoff",""))[:10]:
            continue
        picks=five_doubles(first,second)
        if len(picks)<5:
            continue
        coverage=sum(x["Joint"] for x in picks)
        candidates.append({"first":first,"second":second,"doubles":picks,
                           "coverage":coverage})
    return sorted(candidates,key=lambda x:(-x["coverage"],
        str(x["first"]["fixture"].get("id","")),
        str(x["second"]["fixture"].get("id",""))))[:limit]
