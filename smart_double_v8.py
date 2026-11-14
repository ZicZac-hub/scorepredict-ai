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


def select_auto_research_pool(rows,max_fixtures=18,max_days=3,as_of=None):
    """Keep genuine same-Sydney-day pairs in a bounded research batch.

    The old implementation considered only the next 18 kickoffs: during a
    crowded week those may include many one-off isolated fixtures or thin
    histories, leaving no viable two-match research pair. Reserve at least
    two games per represented date, then fill with the strongest observable
    historical support, all without making extra network requests.
    """
    instant=as_of or datetime.now(timezone.utc)
    groups={}
    for row in rows:
        kickoff=_kick(row)
        if not kickoff or kickoff<=instant or not row.get("result"):
            continue
        day=kickoff.astimezone(SYDNEY).date().isoformat()
        groups.setdefault(day,[]).append(row)
    dates=[day for day in sorted(groups) if len(groups[day])>=2][:max_days]
    if not dates or max_fixtures<2:
        return []
    def quality(row):
        result=row.get("result") or {}
        a=len(result.get("home_last10") or [])
        b=len(result.get("away_last10") or [])
        h=len(result.get("home_home10") or [])
        v=len(result.get("away_away10") or [])
        top=(result.get("scores") or [(0,0,0)])[0]
        return (min(a,b)>=5,min(h,v)>=2,min(a,b),min(h,v),
                float(top[2]), -_kick(row).timestamp())
    for day in dates:
        groups[day].sort(key=quality,reverse=True)
    chosen=[]
    identifiers=set()
    def add(row):
        fixture=row["fixture"]
        identity=(str(fixture.get("provider")),str(fixture.get("id")))
        if identity in identifiers:
            return
        identifiers.add(identity)
        chosen.append(row)
    # Ensure no represented day has only one studied match.
    for day in dates:
        for row in groups[day][:2]:
            if len(chosen)<max_fixtures:
                add(row)
    extras=[]
    for day in dates:
        extras.extend((day,row) for row in groups[day][2:])
    extras.sort(key=lambda p:(-quality(p[1])[0],-quality(p[1])[1],
                              -quality(p[1])[2],-quality(p[1])[3],
                              -quality(p[1])[4],p[0],-_kick(p[1]).timestamp()))
    for day,row in extras:
        if len(chosen)>=max_fixtures:
            break
        add(row)
    return chosen


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
