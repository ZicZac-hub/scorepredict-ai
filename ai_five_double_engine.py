"""AI-assisted manual five-double selection without altering original models.

Exact-score probability mass ALWAYS comes from the existing ZicZac score
model. Independent AI votes are qualitative opinions, not calibrated
likelihoods, and receive at most a small experimental ranking adjustment.
The immutable prospective ledger is never changed by this module.
"""
from collections import Counter
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from smart_double_ai import _scores, _instant, five_doubles, same_fixture

SYDNEY = ZoneInfo("Australia/Sydney")
MIN_LEAD_SECONDS = 300
AI_RANKING_MAX_BOOST = 0.12


def identity(row):
    fixture = row.get("fixture") or {}
    kickoff = _instant(row.get("kickoff") or fixture.get("utcDate"))
    fixture_id = fixture.get("id")
    if fixture_id is None or kickoff is None:
        return None
    return (str(fixture.get("provider") or "football-data.org"),
            str(fixture_id), kickoff.isoformat())


def teams(row):
    fixture = row.get("fixture") or {}
    return tuple(" ".join(str((fixture.get(side + "Team") or {}).get("name") or
                                  "").casefold().split()) for side in ("home", "away"))


def eligible(row, *, now=None):
    clock = _instant(now or datetime.now(timezone.utc))
    kickoff = _instant(row.get("kickoff") or (row.get("fixture") or {}).get("utcDate"))
    return bool(identity(row) and all(teams(row)) and clock and kickoff and
                (kickoff-clock).total_seconds() > MIN_LEAD_SECONDS and
                len(_scores(row)) >= 5)


def pair_check(first, second, *, now=None):
    if not eligible(first, now=now) or not eligible(second, now=now):
        return "Both matches need valid pre-kickoff fixture IDs and at least five statistical exact scores."
    if same_fixture(first,second) or identity(first)==identity(second):
        return "Choose two different matches, not the same fixture."
    first_day = _instant(first.get("kickoff")).astimezone(SYDNEY).date()
    second_day = _instant(second.get("kickoff")).astimezone(SYDNEY).date()
    if first_day != second_day:
        return "Choose two matches on the same Sydney calendar date."
    names = teams(first)+teams(second)
    if len(set(names)) != 4:
        return "Choose four different teams; one team appears in both matches."
    return None


def _votes(record):
    """Only validated provider opinions from genuinely prior research."""
    if not isinstance(record,dict):
        return {}
    if record.get("status") == "completed_after_kickoff_not_prospective":
        return {}
    if record.get("status") not in ("completed",None):
        return {}
    valid={}
    for name,v in (record.get("votes") or {}).items():
        if not isinstance(v,dict):
            continue
        score=v.get("score")
        if (not isinstance(score,(tuple,list)) or len(score)!=2 or
            any(type(x) is not int or x<0 or x>9 for x in score)):
            continue
        valid[str(name)]=v
    return valid


def endorsements(record):
    opinions=_votes(record)
    counter=Counter(tuple(v["score"]) for v in opinions.values())
    return counter,len(opinions)


def merged_five(first,second,first_research=None,second_research=None,*,now=None):
    error=pair_check(first,second,now=now)
    if error:
        return {"status":"invalid_pair","reason":error,"doubles":[]}
    baseline=five_doubles(first,second)
    if len(baseline)!=5:
        return {"status":"insufficient_scores","reason":"Fewer than five valid model combinations.",
                "doubles":[]}
    count1, n1=endorsements(first_research)
    count2, n2=endorsements(second_research)
    left = _scores(first)[:10]
    right = _scores(second)[:10]
    combined=[]
    for h,a,p in left:
        for j,k,q in right:
            original=p*q
            # Votes only break/rerank closely ranked model candidates:
            # never replace or relabel the statistical joint probability.
            boost1=1+AI_RANKING_MAX_BOOST*count1.get((h,a),0)/n1 if n1 else 1
            boost2=1+AI_RANKING_MAX_BOOST*count2.get((j,k),0)/n2 if n2 else 1
            combined.append({
                "Score 1":f"{h}-{a}", "Score 2":f"{j}-{k}",
                "Model joint":original, "Model P1":p, "Model P2":q,
                "AI endorsements 1":count1.get((h,a),0),
                "AI endorsements 2":count2.get((j,k),0),
                "_rank":original*boost1*boost2
            })
    combined.sort(key=lambda x:(-x["_rank"],-x["Model joint"],x["Score 1"],x["Score 2"]))
    chosen=[]
    for item in combined[:5]:
        chosen.append({k:v for k,v in item.items() if k!="_rank"})
    return {
        "status":"ready","doubles":chosen,
        "baseline":baseline,
        "baseline_coverage":sum(d["Joint"] for d in baseline),
        "selected_model_coverage":sum(d["Model joint"] for d in chosen),
        "model_probability_source":"ZicZac existing model score grid",
        "estimated_coverage_is_not_observed_win_rate":True,
        "ai_rank_influence":"experimental, capped at 12% per leg endorsement share",
        "ai_votes":{str(i):n for i,n in ((1,n1),(2,n2))},
        "fixtures":[identity(first),identity(second)],
        "sydney_date":_instant(first.get("kickoff")).astimezone(SYDNEY).date().isoformat(),
        "original_immutable_predictions_changed":False,
    }
