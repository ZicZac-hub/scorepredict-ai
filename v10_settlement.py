"""Independent verified full-time settlement for original v10 research forecasts.

Legacy v6/v7 capture may be absent. This fetches only real completed
API-Football fixtures referenced by frozen V8-family or V37 research forecasts, verifies
provider/team/kickoff identity and 90-minute goals, and saves results without
changing any original predictions. The request budget is deliberately small.
"""
import json
import os
from datetime import datetime, timezone, timedelta
from pathlib import Path

import requests
from api_football import (ROOT as API_ROOT, _convert, ProviderQuotaError,
                          ProviderAccessError, ProviderRejectedError)
from persist_prematch import (RESULTS as LEGACY_RESULTS, load_json,
                              matches_snapshot, verified_result, write_json)
from reconcile_capture_conflicts import merge_verified
from v10_prospective import ROOT, read_latest, utc
from forecast_ledger import read_file_records
from v37_sparse_history_research import frozen_arm_digest

RESULTS=ROOT/"verified_results.json"
STATUS=ROOT/"settlement_status.json"
ATTEMPTS=ROOT/"result_attempts.json"
# Football is generally 90 minutes plus stoppage; avoid querying the
# provider before its regulation-time final could plausibly exist.
MIN_FINAL_LAG=timedelta(minutes=105)
# An unfinished or temporarily unavailable result must not trigger the
# same billable API request on every 15-minute collection pass.
RETRY_WAIT=timedelta(minutes=60)



def _reference_rows(forecasts):
    unique={}
    for (provider,identifier,model),row in forecasts.items():
        if provider!="API-Football":
            continue
        key=str(identifier)
        if key not in unique or (model=="v8" and unique[key]["model_key"]!="v8"):
            unique[key]=row
    return unique



def _verified_v37_references(root, known, clock):
    """Find original pre-kickoff V37 tickets for independent final settlement.

    This is ONLY a result-lookup source: it never qualifies V37 for V8 or
    five-double selection. Reject duplicate, altered or future-issued tickets.
    Existing V8/V9/V10 source references always take precedence.
    """
    grouped={}
    for row in read_file_records(root,kind="forecasts"):
        if not isinstance(row,dict) or row.get("model_key")!="v37_sparse":
            continue
        if row.get("provider")!="API-Football":
            continue
        grouped.setdefault(str(row.get("fixture_id")),[]).append(row)
    added, rejected=0,0
    for fixture_id, rows in sorted(grouped.items()):
        if fixture_id in known:
            continue
        if len(rows)!=1:
            rejected+=1
            continue
        row=rows[0]
        try:
            if (row.get("v37_research_only") is not True or
                    row.get("v37_no_live_model_change") is not True or
                    row.get("v37_not_eligible_for_v28") is not True or
                    row.get("competition") not in ("CL","EL") or
                    row["v37_arm_digest_sha256"] != frozen_arm_digest(row)):
                rejected+=1
                continue
            evidence_time=utc(row["evidence_as_of_utc"])
            saved=utc(row["captured_at_utc"])
            kickoff=utc(row["kickoff_utc"])
            if not evidence_time<=saved<kickoff or saved>clock:
                rejected+=1
                continue
            known[fixture_id]=row
            added+=1
        except (TypeError,KeyError,ValueError,OverflowError):
            rejected+=1
    return {"v37_only_references_added":added,
            "v37_invalid_originals_excluded":rejected}


def _verification_order(references, root, clock):
    """Verify fully due frozen five-double legs before unrelated old scores.

    This only changes RESULT LOOKUP PRIORITY. It never creates or changes any
    pre-match score, pairing, capture timestamp or result.
    """
    prioritized=[]
    for path in sorted((Path(root)/"doubles").glob("*.json")):
        try:
            packet=load_json(path,{})
            if packet.get("mode")!="prospective" or packet.get("status")!="frozen_pre_kickoff":
                continue
            first,second=packet["first"],packet["second"]
            saved=utc(packet["captured_at_utc"])
            legs=[first,second]
            if saved>=min(utc(leg["kickoff_utc"]) for leg in legs):
                continue
            if not all(utc(leg["kickoff_utc"])+MIN_FINAL_LAG<=clock
                       for leg in legs):
                continue
            matching=[]
            for leg in legs:
                fixture_id=str(leg["fixture_id"])
                row=references.get(fixture_id)
                if (row is None or row.get("provider")!=leg.get("provider") or
                    str(row.get("fixture_id"))!=fixture_id or
                    utc(row["kickoff_utc"])!=utc(leg["kickoff_utc"]) or
                    str(row.get("home_team_id"))!=str(leg.get("home_team_id")) or
                    str(row.get("away_team_id"))!=str(leg.get("away_team_id"))):
                    break
                matching.append((fixture_id,row))
            if len(matching)==2:
                prioritized.extend(matching)
        except (ValueError,TypeError,KeyError,OSError):
            continue
    # Once frozen five-double legs are scheduled, do not starve the
    # newly collected research-only V37 finals behind an unrelated backlog.
    # These V37 lookups STILL use the same overall provider-call limit and
    # the same completed-match lag and 60-minute retry cooldown.
    sparse_due=sorted(
        ((fid,row) for fid,row in references.items()
         if row.get("model_key")=="v37_sparse"
         and utc(row["kickoff_utc"])+MIN_FINAL_LAG<=clock),
        key=lambda entry:utc(entry[1]["kickoff_utc"]),reverse=True)
    # Other singles remain newest-first. V8 doubles keep first priority.
    rest=sorted(references.items(),key=lambda entry:utc(entry[1]["kickoff_utc"]),
                reverse=True)
    seen=set()
    ordered=[]
    for fid,row in prioritized+sparse_due+rest:
        if fid in seen:
            continue
        seen.add(fid)
        ordered.append((fid,row))
    return ordered

def _provider_match(key,fixture_id,reference,request=None):
    get=request or requests.get
    response=get(API_ROOT+"/fixtures",
                 params={"id":str(fixture_id)},
                 headers={"x-apisports-key":key},timeout=25)
    status=getattr(response,"status_code",None)
    if status==429:
        raise ProviderQuotaError("Result provider quota or rate limit")
    if status in (401,403):
        raise ProviderAccessError("Result provider credentials or plan unavailable")
    response.raise_for_status()
    data=response.json()
    if not isinstance(data,dict):
        raise ValueError("Invalid result provider envelope")
    if data.get("errors"):
        # Classify locally; never store or log raw vendor bodies or keys.
        hint=str(data["errors"]).casefold()[:3000]
        if any(word in hint for word in ("rate limit","request limit","quota",
                "too many","reached the limit","limit of requests","limit of calls")):
            raise ProviderQuotaError("Result provider quota or rate limit")
        if any(word in hint for word in ("api key","token","subscription",
                "plan access","not authorized","not authorised")):
            raise ProviderAccessError("Result provider credentials or plan unavailable")
        raise ProviderRejectedError("Result provider rejected request")
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
    sparse_report=_verified_v37_references(root,references,clock)
    own_file=Path(root)/"verified_results.json"
    own=load_json(own_file,{})
    old=legacy_results if legacy_results is not None else load_json(LEGACY_RESULTS,{})
    attempts_file=Path(root)/"result_attempts.json"
    attempts=load_json(attempts_file,{})
    status={"eligible_frozen_fixtures":len(references),
            **sparse_report,
            "already_verified":0,"checked":0,"newly_verified":0,
            "still_pending":0,"rejected_or_unavailable":0,
            "waiting_for_full_time":0,"retry_cooldown":0,
            "source_errors":[],"provider_blocker":None,"deferred_provider_block":0}
    merged=merge_verified(old,own)
    for fixture_id,reference in _verification_order(references,root,clock):
        known=merged.get(str(fixture_id))
        if known is not None:
            if not verified_result(reference,known,fixture_id):
                raise ValueError("Previously verified result conflicts with frozen fixture "+fixture_id)
            status["already_verified"]+=1
            # Reuse existing authoritative historical score when present.
            if fixture_id not in own:
                own[fixture_id]=known
            continue
        # A frozen forecast alone does not justify a premature provider call.
        if utc(reference["kickoff_utc"])+MIN_FINAL_LAG>clock:
            status["waiting_for_full_time"]+=1
            continue
        try:
            last_check=utc(attempts[fixture_id])
        except (KeyError,ValueError,TypeError):
            last_check=None
        if last_check is not None and clock-last_check<RETRY_WAIT:
            status["retry_cooldown"]+=1
            continue
        if status["provider_blocker"] is not None:
            status["deferred_provider_block"]+=1
            continue
        if status["checked"]>=limit:
            status["still_pending"]+=1
            continue
        status["checked"]+=1
        # Persist the attempt even when no verified final is available.
        # This records a retry window, never an inferred match result.
        attempts[fixture_id]=clock.isoformat()
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
            if isinstance(exc,(ProviderQuotaError,ProviderAccessError)):
                # Stop further network requests, but still reconcile known verified
                # results below in the same pass. Untouched fixtures remain pending.
                status["provider_blocker"]={
                    "fixture_id":fixture_id,
                    "reason":"provider_quota" if isinstance(exc,ProviderQuotaError)
                             else "provider_credentials_or_plan"
                }
    # All not-yet-verified references remain pending, including early
    # kickoffs, missing provider scores and cooldowns: never mislabel losses.
    status["still_pending"]=max(0,len(references)-status["already_verified"]-
                                status["newly_verified"])
    # The destination contains verified results ONLY; a re-run never changes
    # an already published result's score or timestamp.
    own_file.parent.mkdir(parents=True,exist_ok=True)
    write_json(own_file,own)
    write_json(attempts_file,attempts)
    write_json(Path(root)/"settlement_status.json",status)
    return status


def main():
    key=os.environ.get("API_FOOTBALL_KEY","")
    limit=max(1,min(50,int(os.environ.get("V10_RESULT_LIMIT","20"))))
    print(json.dumps(collect(key,limit=limit),indent=2))


if __name__=="__main__":
    main()
