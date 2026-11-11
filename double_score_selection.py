"""Diversified exact-score double selection.

Five combinations are still only five outcomes; score coverage is not a
guarantee. First pick the highest-probability combination, then prefer
previously uncovered scorelines in either leg with probability as tie-break.
"""
def select_diversified(options,limit=5,score1="Score 1",score2="Score 2",probability="Joint"):
    ordered=sorted(options,key=lambda x:-float(x[probability]))
    if not ordered or limit<=0:return []
    selected=[ordered.pop(0)]
    seen_first={selected[0][score1]}
    seen_second={selected[0][score2]}
    while ordered and len(selected)<limit:
        # Diversity of the two legs is prioritized, without fabricating odds.
        index=max(range(len(ordered)),key=lambda i:(
            int(ordered[i][score1] not in seen_first)+int(ordered[i][score2] not in seen_second),
            float(ordered[i][probability]),-i))
        pick=ordered.pop(index)
        selected.append(pick)
        seen_first.add(pick[score1]);seen_second.add(pick[score2])
    return selected

def score_coverage(rows,score1="Score 1",score2="Score 2"):
    return {"match_1_scores":len({r[score1] for r in rows}),
            "match_2_scores":len({r[score2] for r in rows}),
            "distinct_exact_score_pairs":len({(r[score1],r[score2]) for r in rows})}
