"""Freeze one genuine ZicZac v8 Prediction of the Day in the scheduled capture.

This runs with the existing verified as-of evidence, makes ZERO new provider
requests and never mutates previous live prediction snapshots. A daily pick is
write-once; a later pass will not silently replace it after kickoff.
"""
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo
import json
import os
from uuid import uuid4

from daily_pick_ui import build_snapshot, _when

SYDNEY = ZoneInfo("Australia/Sydney")
ROOT = Path("data/daily_picks")
POLICY = "daily-prekickoff-v8-frozen-v1"


def candidate(fixture, evidence, observed, code):
    """Use ONLY the verified, pre-match historical packet already collected."""
    asof = _when(observed)
    kickoff = _when(fixture.get("utcDate"))
    if asof is None or kickoff is None or asof>=kickoff:
        return None
    sydney_day = asof.astimezone(SYDNEY).date()
    match_day = kickoff.astimezone(SYDNEY).date()
    if match_day not in (sydney_day,sydney_day+timedelta(days=1)):
        return None
    if not evidence or not evidence.get("input_digest"):
        return None
    from research_ensemble_v8 import predict as v8_predict
    estimate = v8_predict(fixture,evidence.get("history") or [],
                          league_history=evidence.get("league_history") or [],
                          as_of=asof)
    if not estimate or len(estimate.get("component_weights") or {})<2:
        return None
    if min(len(estimate.get("home_last10") or []),
           len(estimate.get("away_last10") or [])) < 5:
        return None
    if min(len(estimate.get("home_home10") or []),
           len(estimate.get("away_away10") or [])) < 3:
        return None
    row={"fixture":fixture,"kickoff":fixture["utcDate"],"league":str(code),
         "result":estimate}
    saved=build_snapshot(row,asof,estimate)
    if not saved:
        return None
    saved["status"]="frozen_in_capture_workflow"
    saved["policy_version"]=POLICY
    saved["evidence_policy_version"]=evidence.get("policy_version")
    saved["evidence_input_digest"]=evidence["input_digest"]
    saved["selection_basis"]="Verified completed goals, venue form, opponent ratings, independent Elo and Dixon-Coles model. No bookmaker odds or unverified lineups inserted."
    return saved


def rank(snapshot):
    """Model-data coverage first, then 1X2 confidence and exact modal mass."""
    try:
        return (min(snapshot["home_recent"],snapshot["away_recent"]),
                min(snapshot["home_venue"],snapshot["away_venue"]),
                snapshot["component_count"],
                max(snapshot["outcomes"].values()),
                snapshot["probability"])
    except (KeyError,TypeError,ValueError):
        return (0,0,0,0,0)


def save_for_dates(candidates,root=ROOT,now=None):
    """Atomic create-only freeze. Returns newly saved days; never rewrites."""
    instant=_when(now or datetime.now(timezone.utc))
    root=Path(root)
    root.mkdir(parents=True,exist_ok=True)
    today=instant.astimezone(SYDNEY).date()
    written=[]
    for day in (today,today+timedelta(days=1)):
        path=root/(day.isoformat()+".json")
        if path.exists():
            continue
        valid=[c for c in candidates if c.get("prediction_date")==day.isoformat()
               and _when(c.get("captured_at_utc"))
               and _when(c.get("kickoff_utc"))
               and _when(c["captured_at_utc"]) < _when(c["kickoff_utc"])]
        if not valid:
            continue
        best=max(valid,key=rank)
        # Frozen outcome must be known before kickoff, even if a job was slow.
        # A past kickoff at saving time is allowed: the prediction itself was
        # generated before kickoff. This file records that original timestamp.
        best=dict(best)
        best["freeze_recorded_at_utc"]=instant.isoformat()
        blob=json.dumps(best,indent=2,sort_keys=True)+"\n"
        temp=path.with_name(path.name+"."+uuid4().hex+".tmp")
        try:
            temp.write_text(blob,encoding="utf-8")
            # open exclusively: unlike os.replace, this cannot overwrite a
            # same-date file written by another capture process.
            try:
                with path.open("x",encoding="utf-8") as dest:
                    dest.write(blob)
                written.append(day.isoformat())
            except FileExistsError:
                pass
        finally:
            temp.unlink(missing_ok=True)
    return written
