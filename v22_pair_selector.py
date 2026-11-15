"""V22 manual Studio Auto Select: research ALL pairs, no API spending.

Only previously preserved, genuinely source-matched packets contribute.
If none have sufficient shot-xG history and dated pre-match odds for both
fixtures, the selector abstains rather than choosing an arbitrary pair.
"""
from itertools import combinations

import numpy as np
from ai_five_double_engine import eligible,pair_check,identity
from smart_double_ai import five_doubles
from v22_shadow import shadow
from v10_prospective import utc


def candidates(rows,packet_for,*,now,min_coverage=.055,
               max_pair_disagreement=.20,max_results=5):
    valid=[]
    reasons={"not_eligible":0,"source_not_saved":0,"incomplete_xg_or_market":0,
             "model_disagreement":0,"below_coverage":0,"invalid_pair":0}
    for row in rows or []:
        if not eligible(row,now=now):
            reasons["not_eligible"]+=1
            continue
        fixture=row.get("fixture") or {}
        packet=packet_for(fixture)
        if not packet:
            reasons["source_not_saved"]+=1
            continue
        forecast=shadow(row,packet,as_of=utc(now))
        if not forecast.get("ready"):
            reasons["incomplete_xg_or_market"]+=1
            continue
        base=np.asarray(row["result"]["matrix"],dtype=float)
        adjusted=np.asarray(forecast["matrix"],dtype=float)
        mismatch=float(np.abs(base-adjusted).sum()/2)
        if mismatch>max_pair_disagreement:
            reasons["model_disagreement"]+=1
            continue
        source_model={**row,"result":{**row["result"],
                              "matrix":adjusted,"scores":forecast["scores"],
                              "model_version":forecast["prediction_method"]}}
        valid.append((row,source_model,forecast,mismatch))
    pairs=[]
    for a,b in combinations(valid,2):
        if pair_check(a[0],b[0],now=now) is not None:
            reasons["invalid_pair"]+=1
            continue
        original=five_doubles(a[0],b[0])
        researched=five_doubles(a[1],b[1])
        if len(original)!=5 or len(researched)!=5:
            reasons["invalid_pair"]+=1
            continue
        base_mass=sum(row["Joint"] for row in original)
        new_mass=sum(row["Joint"] for row in researched)
        minimum=min(base_mass,new_mass)
        if minimum<min_coverage:
            reasons["below_coverage"]+=1
            continue
        pairs.append({"first":a[0],"second":b[0],
                      "baseline_coverage_estimate":base_mass,
                      "v22_coverage_estimate":new_mass,
                      "conservative_coverage_estimate":minimum,
                      "marginal_shift":[a[3],b[3]],
                      "evidence_sha256":[a[2]["source_snapshot_digest"],
                                         b[2]["source_snapshot_digest"]],
                      "not_observed_accuracy":True,
                      "method":"v22_source_qualified_research_not_promoted"})
    pairs.sort(key=lambda r:(-r["conservative_coverage_estimate"],
               identity(r["first"]),identity(r["second"])))
    return {"pairs":pairs[:max_results],"total_qualifying_pairs":len(pairs),
            "screened_fixtures":len(rows or []),"source_qualified_fixtures":len(valid),
            "reasons":reasons,"abstained":not bool(pairs),
            "score_coverage_not_a_hit_rate":True,
            "no_external_calls":True}
