"""V21 research-only auto-pair selection for interactive AI Five Doubles.

Rank every supported pair with actual dated goal form on BOTH legs; abstain
instead of borrowing history or selecting the earliest two fixtures.
"""
from itertools import combinations
from ai_five_double_engine import eligible,pair_check,identity
from goal_research_v20 import goal_form_challenger
from smart_double_ai import five_doubles


def _history(row, histories):
    identity_key = identity(row)
    past = row.get("past") or (histories or {}).get(row.get("league"),[]) or []
    return [match for match in past if
            str(match.get("provider") or "football-data.org") == identity_key[0]]

def candidates(rows, histories=None, *, now=None, min_coverage=.055, max_results=5):
    valid=[]
    reasons={"insufficient_dated_goal_evidence":0,"invalid_or_late_fixture":0,
             "ineligible_pair":0,"below_coverage":0}
    for row in rows or []:
        if not eligible(row,now=now):
            reasons["invalid_or_late_fixture"]+=1
            continue
        research=goal_form_challenger(row,_history(row,histories or {}),now)
        if not research.get("available"):
            reasons["insufficient_dated_goal_evidence"]+=1
            continue
        new_row={**row,"result":{**row["result"],
                               "scores":research["scores"],
                               "matrix":research["matrix"],
                               "model_version":research["model"]}}
        valid.append((row,new_row,research))
    options=[]
    for left,right in combinations(valid,2):
        first,research_first,_=left
        second,research_second,_=right
        if pair_check(first,second,now=now) is not None:
            reasons["ineligible_pair"]+=1
            continue
        base=five_doubles(first,second)
        chosen=five_doubles(research_first,research_second)
        if len(base)!=5 or len(chosen)!=5:
            reasons["ineligible_pair"]+=1
            continue
        base_prob=sum(x["Joint"] for x in base)
        new_prob=sum(x["Joint"] for x in chosen)
        conservative=min(base_prob,new_prob)
        if conservative<min_coverage:
            reasons["below_coverage"]+=1
            continue
        options.append({"first":first,"second":second,
                        "model_estimated_research_coverage":new_prob,
                        "model_estimated_original_coverage":base_prob,
                        "conservative_coverage":conservative,
                        "method":"v21_source_gated_goal_form_shadow",
                        "not_observed_win_rate":True,
                        "not_promoted":True})
    options.sort(key=lambda row:(-row["conservative_coverage"],
                identity(row["first"]),identity(row["second"])))
    return {"pairs":options[:max(0,max_results)],"eligible_fixtures":len(valid),
            "pairs_above_threshold":len(options),"screened_fixtures":len(rows or []),
            "abstained":not options,"reasons":reasons,
            "threshold_is_unvalidated_model_coverage":min_coverage,
            "scope":"same_Sydney_day_verified_goal_history_only"}
