"""Independent verified full-time settlement for original v10 research forecasts.

Legacy v6/v7 capture may be absent. This fetches only real completed
API-Football fixtures referenced by frozen v8/v9/v10 forecasts, verifies
provider/team/kickoff identity and 90-minute goals, and saves results without
changing any original predictions. The request budget is deliberately small.
"""
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import requests
from api_football import ROOT as API_ROOT, _convert
from persist_prematch import (RESULTS as LEGACY_RESULTS, load_json,
                              matches_snapshot, verified_result, write_json)
from reconcile_capture_conflicts import merge_verified
from v10_prospective import ROOT, read_latest, utc

RESULTS=ROOT/"verified_results.json"
STATUS=ROOT/"settlement_status.json"


def _reference_rows(forecasts):
    unique={}
    for (provider,identifier,model),row in forecasts.items():
        if provider!="API-Football":
            continue
        key=str(identifier)
        if key not in unique or (model=="v8" and unique[key]["model_key"]!="v8"):
            unique[key]=row
    return unique


def _provider_match(key,fixture_id,reference,request=None):
    get=request or requests.get
    response=get(API_ROOT+"/fixtures",
                 params={"id":str(fixture_id)},
                 headers={"x-apisports-key":key},timeout=25)
    response.raise_for_status()
    data=response.json()
    if data.get("errors"):
        raise RuntimeError("Provider refused the result verification request")
    raw=next((r for r in data.get("response") or []
              if str((r.get("fixture") or {}).get("id"))==str(fixture_id)),None)
    if raw is None:
        return None
    match=_convert(raw,reference.get("competition") or "WORLD")
    if not matches_snapshot(match,reference,fixture_id):
        return None
    if match.get("status")!="FINISHED" or match.get("provider_status") not in ("FT","AET","PEN"):
        return None
    goals=(match.get("score") or {}).get("fullTime") or {}
    if any(type(goals.get(side)) is not int or goals[side]<0 for side in ("home","away")):
        return None
    now=datetime.now(timezone.utc)
    if utc(match["utcDate"])>=now:
        return None
    return {
        "fixture_id":int(fixture_id),"provider":"API-Football",
        "kickoff_utc":match["utcDate"],
        "home_team_id":match["homeTeam"]["id"],"away_team_id":match["awayTeam"]["id"],
        "home_team":match["homeTeam"]["name"],"away_team":match["awayTeam"]["name"],
        "status":match["provider_status"],"score_basis":"regulation",
        "evidence_version":2,"verified_at_utc":now.isoformat(),
        "home_goals":goals["home"],"away_goals":goals["away"]
    }


def collect(key, *, now=None, limit=20, root=ROOT, request=None,
            legacy_results=None):
    """Verify up to limit completed, previously frozen, distinct fixtures."""
    if not key:
        raise ValueError("Result verification requires the API_FOOTBALL_KEY secret")
    if limit<1:
        raise ValueError("Verification request limit must be positive")
    clock=utc(now or datetime.now(timezone.utc))
    forecasts=read_latest(root)
    references=_reference_rows(forecasts)
    own_file=Path(root)/"verified_results.json"
    own=load_json(own_file,{})
    old=legacy_results if legacy_results is not None else load_json(LEGACY_RESULTS,{})
    status={"eligible_frozen_fixtures":len(references),
            "already_verified":0,"checked":0,"newly_verified":0,
            "still_pending":0,"rejected_or_unavailable":0,
            "source_errors":[]}
    merged=merge_verified(old,own)
    for fixture_id,reference in sorted(
            references.items(),key=lambda p:utc(p[1]["kickoff_utc"])):
        # Never verify a fixture before its full-time result could be final.
        if utc(reference["kickoff_utc"])>=clock:
            status["still_pending"]+=1
            continue
        known=merged.get(str(fixture_id))
        if known is not None:
            if not verified_result(reference,known,fixture_id):
                raise ValueError("Previously verified result conflicts with frozen fixture "+fixture_id)
            status["already_verified"]+=1
            # Reuse existing authoritative historical score when present.
            if fixture_id not in own:
                own[fixture_id]=known
            continue
        if status["checked"]>=limit:
            status["still_pending"]+=1
            continue
        status["checked"]+=1
        try:
            row=_provider_match(key,fixture_id,reference,request=request)
            if row and verified_result(reference,row,fixture_id):
                own[fixture_id]=row
                merged[fixture_id]=row
                status["newly_verified"]+=1
            else:
                status["rejected_or_unavailable"]+=1
        except (requests.RequestException,ValueError,TypeError,RuntimeError,KeyError) as exc:
            status["source_errors"].append({"fixture_id":fixture_id,
                                             "error_type":type(exc).__name__})
    # The destination contains verified results ONLY; a re-run never changes
    # an already published result's score or timestamp.
    own_file.parent.mkdir(parents=True,exist_ok=True)
    write_json(own_file,own)
    write_json(Path(root)/"settlement_status.json",status)
    return status


def main():
    key=os.environ.get("API_FOOTBALL_KEY","")
    limit=max(1,min(50,int(os.environ.get("V10_RESULT_LIMIT","20"))))
    print(json.dumps(collect(key,limit=limit),indent=2))


if __name__=="__main__":
    main()
