"""Experimental league-relative attack/defence ratings with venue splits.

All input games must precede fixture kickoff. No results or fitted values
from the evaluated fixture are used. Designed as a distinct challenger.
"""
import math
from collections import defaultdict
from datetime import datetime, timezone

def _date(value):
    return datetime.fromisoformat(value.replace("Z","+00:00"))

def expected_goals(fixture, past, prior=8.0, half_life_days=240.0):
    kickoff=_date(fixture["utcDate"])
    home_id=fixture["homeTeam"]["id"]
    away_id=fixture["awayTeam"]["id"]
    games=[m for m in past if _date(m["utcDate"])<kickoff]
    if len(games)<30:return None
    base_h=sum(m["score"]["fullTime"]["home"] for m in games)/len(games)
    base_a=sum(m["score"]["fullTime"]["away"] for m in games)/len(games)
    base_h=max(.65,base_h);base_a=max(.65,base_a)
    # Opponent quality is represented by an iterative regularized
    # league-wide attack/defence system, not just recent raw goals.
    observations=[]
    for m in games:
        h=m["homeTeam"]["id"];a=m["awayTeam"]["id"]
        if h is None or a is None:continue
        d=_date(m["utcDate"])
        age=max(0.0,(kickoff-d).total_seconds()/86400)
        w=2**(-age/half_life_days)
        score=m["score"]["fullTime"]
        observations.append((h,a,min(8,score["home"]),min(8,score["away"]),w))
    if not observations:return None
    teams={x for h,a,_,_,_ in observations for x in (h,a)}
    attack_h={t:1.0 for t in teams};attack_a={t:1.0 for t in teams}
    defence_h={t:1.0 for t in teams};defence_a={t:1.0 for t in teams}
    for _ in range(6):
        numer=defaultdict(float);denom=defaultdict(float)
        for h,a,gh,ga,w in observations:
            numer[("ah",h)]+=w*gh
            denom[("ah",h)]+=w*base_h*defence_a[a]
            numer[("aa",a)]+=w*ga
            denom[("aa",a)]+=w*base_a*defence_h[h]
            numer[("dh",h)]+=w*ga
            denom[("dh",h)]+=w*base_a*attack_a[a]
            numer[("da",a)]+=w*gh
            denom[("da",a)]+=w*base_h*attack_h[h]
        for t in teams:
            attack_h[t]=max(.35,min(2.8,(numer[("ah",t)]+prior)/(denom[("ah",t)]+prior)))
            attack_a[t]=max(.35,min(2.8,(numer[("aa",t)]+prior)/(denom[("aa",t)]+prior)))
            defence_h[t]=max(.35,min(2.8,(numer[("dh",t)]+prior)/(denom[("dh",t)]+prior)))
            defence_a[t]=max(.35,min(2.8,(numer[("da",t)]+prior)/(denom[("da",t)]+prior)))
    if home_id not in teams or away_id not in teams:return None
    return (max(.25,min(4.0,base_h*attack_h[home_id]*defence_a[away_id])),
            max(.25,min(4.0,base_a*attack_a[away_id]*defence_h[home_id])))
