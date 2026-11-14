"""Optional v8 research-driven five exact-score doubles.

Builds on preserved smart_double_ai.five_doubles. Never edits the historic
ledger or existing production picks. All picks are exploratory until frozen
and verified against final results in an independent comparison.
"""
from datetime import datetime, timezone
from itertools import combinations
from zoneinfo import ZoneInfo
from research_ensemble_v8 import predict as predict_v8
from smart_double_ai import five_doubles, same_fixture

SYDNEY = ZoneInfo("Australia/Sydney")
VERSION = "evidence-v8-five-doubles-challenger"


def _kick(row):
    try:
        dt=datetime.fromisoformat(str(row.get("kickoff", "")).replace("Z","+00:00"))
        return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)
    except (ValueError,TypeError):
        return None


def _team_key(row,side):
    fixture=row["fixture"]
    team=(fixture.get(side+"Team") or {})
    # Club names are a safer cross-provider duplicate check than numeric IDs.
    return " ".join(str(team.get("name") or "").casefold().split())


def research_fixture_rows(rows, histories, max_fixtures=18, as_of=None):
    """Calculate at most 18 time-bounded challenger forecasts on demand."""
    as_of=as_of or datetime.now(timezone.utc)
    available = sorted((row for row in rows if row.get("result") and
                        _kick(row) and _kick(row)>as_of),key=_kick)
    evaluated=[]
    seen=set()
    attempts=0
    for row in available:
        fixture=row["fixture"]
        fingerprint=(_kick(row),_team_key(row,"home"),_team_key(row,"away"))
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        if attempts>=max_fixtures:
            break
        attempts += 1
        before=row.get("past") or histories.get(row.get("league"),[])
        league=histories.get(row.get("league"),before)
        forecast=predict_v8(fixture,before,league_history=league,as_of=as_of)
        if not forecast or len(forecast.get("component_weights",{}))<2:
            # When only the preserved baseline applies, do not relabel it v8.
            continue
        evaluated.append({**row,"result":forecast})
    return evaluated


def rank_research_pairs(rows,limit=5):
    """Same Sydney date, four distinct teams, highest five-double model mass."""
    candidates=[]
    for first,second in combinations(rows,2):
        if same_fixture(first,second):
            continue
        a,b=_kick(first),_kick(second)
        if a is None or b is None or a.astimezone(SYDNEY).date()!=b.astimezone(SYDNEY).date():
            continue
        names1={_team_key(first,"home"),_team_key(first,"away")}
        names2={_team_key(second,"home"),_team_key(second,"away")}
        if "" in names1|names2 or names1&names2:
            continue
        picks=five_doubles(first,second)
        if len(picks)!=5:
            continue
        coverage=sum(x["Joint"] for x in picks)
        candidates.append({"first":first,"second":second,"doubles":picks,
                           "coverage":coverage,"model_version":VERSION,
                           "assumption":"Independent matches; model probabilities uncalibrated"})
    return sorted(candidates,key=lambda item:(-item["coverage"],str(item["first"]["fixture"].get("id")),
                                            str(item["second"]["fixture"].get("id"))))[:limit]


def manually_selected(first,second,histories,as_of=None):
    """Return both original and challenger doubles for the exact user pair."""
    researched=research_fixture_rows([first,second],histories,max_fixtures=2,as_of=as_of)
    if len(researched)!=2:
        return {"available":False,"reason":"One or both matches lack sufficient v8 verified league research."}
    lookup={(str(r["fixture"].get("provider")),str(r["fixture"].get("id"))):r for r in researched}
    a=lookup.get((str(first["fixture"].get("provider")),str(first["fixture"].get("id"))))
    b=lookup.get((str(second["fixture"].get("provider")),str(second["fixture"].get("id"))))
    if a is None or b is None:
        return {"available":False,"reason":"A requested match could not be matched to verified research."}
    doubles=five_doubles(a,b)
    if len(doubles)!=5:
        return {"available":False,"reason":"Insufficient valid score probabilities to form five distinct doubles."}
    return {"available":True,"first":a,"second":b,"doubles":doubles,
            "coverage":sum(x["Joint"] for x in doubles),
            "model_version":VERSION,"assumption":"Independent-match estimate; not measured win rate."}
