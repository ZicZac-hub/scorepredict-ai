from outcome_double_challenger import rank_outcome_aware_pairs

def row(identifier, day, confidence, scores=None):
    return {"fixture":{"id":identifier,"provider":"test"},
            "kickoff":day+"T12:00:00Z",
            "result":{"available":True,"outcome_confidence":confidence,
                      "scores":scores or [[1,0,.30],[2,0,.20],[1,1,.15]]}}

def test_excludes_uncertain_matches_and_other_days():
    rows=[row(1,"2026-11-20",.8),row(2,"2026-11-20",.7),
          row(3,"2026-11-20",.4),row(4,"2026-11-21",.9)]
    pairs=rank_outcome_aware_pairs(rows)
    assert len(pairs)==1
    assert {pairs[0]["first"]["fixture"]["id"],pairs[0]["second"]["fixture"]["id"]}=={1,2}
    assert len(pairs[0]["doubles"])==5

def test_probability_coverage_is_sum_of_distinct_joint_scores():
    a=row(1,"2026-11-20",.7)
    b=row(2,"2026-11-20",.7)
    pair=rank_outcome_aware_pairs([a,b])[0]
    assert abs(pair["coverage"]-sum(d["Joint"] for d in pair["doubles"]))<1e-12
    assert len({(d["Score 1"],d["Score 2"]) for d in pair["doubles"]})==5
