"""Prioritise genuine original five-double fixtures in paid AI research.

Only READ archived write-once pre-kickoff evidence. No API calls, model
promotion, retrospective predictions, or fixture selections after results.
The owner must separately opt in to two-match paid research requests.
"""
import math
import re
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from persist_prematch import load_json
from v10_prospective import utc

SYDNEY = ZoneInfo("Australia/Sydney")
SAFE = re.compile(r"^[A-Za-z0-9_.-]+$")


def original_model(root, leg):
    """Read the actual saved V8 source, not a later latest-per-fixture refresh."""
    parts=[str(leg.get(k) or "") for k in
           ("provider","fixture_id","forecast_id")]
    if any(not SAFE.fullmatch(value) or value in (".","..")
           for value in parts):
        return None
    path=(Path(root)/"forecasts"/parts[0]/parts[1]/(parts[2]+".json"))
    record=load_json(path,{})
    if not isinstance(record,dict):
        return None
    try:
        keys=("provider","fixture_id","forecast_id","home_team_id",
              "away_team_id")
        if any(str(record[k])!=str(leg[k]) for k in keys):
            return None
        if utc(record["kickoff_utc"])!=utc(leg["kickoff_utc"]):
            return None
        if not record.get("input_digest"):
            return None
        if utc(record["captured_at_utc"])>=utc(leg["kickoff_utc"]):
            return None
        if record.get("model_key") not in ("v8","v8_near"):
            return None
        return record
    except (TypeError,ValueError,KeyError):
        return None


def frozen_pair_priority(root, now, already_researched, check_eligible):
    """Return (both original model records, date) or ([], None).

    Search is restricted to known pre-match frozen double sets. Candidate
    preference is strictly their ORIGINAL pre-match model-estimated coverage.
    Scores or verified results never enter this selector.
    """
    clock=utc(now)
    choices=[]
    for path in sorted((Path(root)/"doubles").glob("*.json")):
        pair=load_json(path,{})
        if not isinstance(pair,dict) or (
            pair.get("mode")!="prospective" or
            pair.get("status")!="frozen_pre_kickoff"):
            continue
        try:
            if utc(pair["captured_at_utc"])>clock:
                continue
            a,b=pair["first"],pair["second"]
            kicks=[utc(leg["kickoff_utc"]) for leg in (a,b)]
            if clock>=min(kicks):
                continue
            if kicks[0].astimezone(SYDNEY).date()!=kicks[1].astimezone(SYDNEY).date():
                continue
            if kicks[0].astimezone(SYDNEY).date().isoformat()!=pair["sydney_date"]:
                continue
            identities=[(str(leg["provider"]),str(leg["fixture_id"]))
                        for leg in (a,b)]
            if (identities[0]==identities[1] or
                any(identity in already_researched for identity in identities)):
                continue
            teams=[str(leg[k]) for leg in (a,b) for k in
                   ("home_team_id","away_team_id")]
            if len(set(teams))!=4:
                continue
            models=[original_model(root,leg) for leg in (a,b)]
            if not all(model is not None and check_eligible(model,clock)
                       for model in models):
                continue
            doubles=pair["five_doubles"]
            if len(doubles)!=5 or len({
                (x["score1"],x["score2"]) for x in doubles})!=5:
                continue
            probabilities=[float(row["joint_probability"]) for row in doubles]
            if any(not math.isfinite(p) or p<0 or p>1 for p in probabilities):
                continue
            coverage=sum(probabilities)
            if not 0<coverage<=1:
                continue
            # Prefer original, frozen coverage; never use final results.
            choices.append((-coverage,pair["sydney_date"],path.name,models))
        except (TypeError,ValueError,KeyError):
            continue
    if not choices:
        return [],None
    _,date,_,models=min(choices,key=lambda x:x[:3])
    return models,date
