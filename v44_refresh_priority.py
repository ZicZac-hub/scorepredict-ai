"""Bounded priority for historical refresh of already-selected research fixtures.

This module does not select new pairs, generate forecasts, examine results,
or request data. It reorders only an existing eligible refresh queue,
reserving at most two of the existing slots for authenticated pair legs.
"""
from datetime import timedelta
from pathlib import Path

from v10_prospective import utc
import v42_research_trial as sources

MAX_RESERVED=2


def model_key(m):
    return (str(m["provider"]),str(m["fixture_id"]),utc(m["kickoff_utc"]),
            str(m["home_team_id"]),str(m["away_team_id"]))


def fixture_key(f):
    return (str(f["provider"]),str(f["id"]),utc(f["utcDate"]),
            str(f["homeTeam"]["id"]),str(f["awayTeam"]["id"]))


def targets(*,now,pair_root=sources.PAIR_ROOT,source_root=sources.SOURCE_ROOT):
    """Authenticate original pair; prioritize only its upcoming refresh window."""
    now=utc(now)
    out={}
    for path in sorted((Path(pair_root)/"sets").glob("*.json")):
        try:
            _,models=sources._pair(path,source_root=source_root,decision=now)
            earliest=min(utc(m["kickoff_utc"]) for m in models)
            if not timedelta(minutes=45)<earliest-now<=timedelta(hours=24):
                continue
            for m in models:
                key=model_key(m)
                if key[2]-now<=timedelta(hours=48):
                    out[key]=min(out.get(key,earliest),earliest)
        except (ValueError,KeyError,TypeError,OSError,AttributeError,ArithmeticError):
            continue
    return out


def prioritize(awaiting,shadow,*,now,limit,pair_root=sources.PAIR_ROOT,source_root=sources.SOURCE_ROOT):
    """Retain all original eligibility, capacity, and remaining chronological order."""
    ordered=sorted(awaiting,key=lambda r:(r[0],str(r[1]["id"])))
    if limit<=0 or not ordered:
        return ordered,0
    required=targets(now=now,pair_root=pair_root,source_root=source_root)
    candidates=[]
    for i,row in enumerate(ordered):
        try:
            key=fixture_key(row[1])
            # An existing near capture is immutable; no attempt to replace it.
            if key in required and (*key[:2],"v8_near") not in shadow:
                candidates.append((required[key],row[0],key[0],key[1],i))
        except (ValueError,KeyError,TypeError,AttributeError):
            continue
    indices=[r[-1] for r in sorted(candidates)[:min(MAX_RESERVED,limit)]]
    reserved=set(indices)
    return [ordered[i] for i in indices]+[r for i,r in enumerate(ordered) if i not in reserved],len(indices)
