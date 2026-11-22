"""Bounded, identity-checked API-Football final-score backfill for frozen V8 fixtures."""
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from api_football import _request, ProviderQuotaError, ProviderAccessError, ProviderRejectedError
from v10_prospective import ROOT, read_latest, utc
from v16_pair_research import original_model
from v31_slate_settlement import enrolled_due

RESULTS = Path("data/prematch/verified_results.json")
STATUS = Path("data/diagnostics/result_backfill.json")
MAX_CALLS = 12

def _accept_verified(item,fid,frozen,existing,status,clock):
    fixture=item.get("fixture") or {}
    teams=item.get("teams") or {}
    home=teams.get("home") or {}
    away=teams.get("away") or {}
    goals=item.get("goals") or {}
    if (fixture.get("status") or {}).get("short")!="FT":
        return
    if (str(fixture.get("id"))!=fid or
        str(home.get("id"))!=str(frozen["home_team_id"]) or
        str(away.get("id"))!=str(frozen["away_team_id"]) or
        utc(fixture["date"])!=utc(frozen["kickoff_utc"])):
        status["skipped_identity"]+=1
        return
    hg,ag=goals.get("home"),goals.get("away")
    if type(hg)!=int or type(ag)!=int or hg<0 or ag<0:
        return
    existing[fid]={"provider":"API-Football","fixture_id":int(fid),
        "home_team_id":int(home["id"]),"away_team_id":int(away["id"]),
        "home_team":home.get("name"),"away_team":away.get("name"),
        "kickoff_utc":utc(fixture["date"]).isoformat(),
        "home_goals":hg,"away_goals":ag,"score_basis":"regulation",
        "status":"FT","verified_at_utc":clock.isoformat(),
        "evidence_version":2}
    status["new_verified"].append(fid)

def run(request=_request, now=None):
    clock = utc(now) if now else datetime.now(timezone.utc)
    key = os.environ.get("API_FOOTBALL_KEY", "")
    status = {"at_utc":clock.isoformat(),"requests":0,"new_verified":[],
              "skipped_identity":0,"eligible_unverified":0,"state":"complete"}
    if not key:
        status["state"]="missing_provider_key"
        return save(status)
    try:
        existing=json.loads(RESULTS.read_text()) if RESULTS.exists() else {}
        if not isinstance(existing,dict):
            status["state"]="invalid_existing_results"
            return save(status)
        # Original source receipts from ALL frozen V28 candidate slates are
        # enrollment evidence even after a latest forecast pointer changes.
        # Refuse tampered slates before issuing ANY API request.
        priority, enrolled = enrolled_due(ROOT, existing, clock)
        status["original_slate_settlement"] = enrolled
        candidates=dict(priority)
        # Newest forecasts remain eligible for the legacy result pipeline,
        # but cannot displace original pre-kickoff V28 source identities.
        try:
            originals=read_latest(ROOT)
        except (OSError,ValueError,TypeError,KeyError):
            originals={}
            status["latest_forecast_index_unavailable"]=True
        latest_ids=set()
        for row in originals.values() if isinstance(originals,dict) else originals:
            try:
                if row.get("provider")!="API-Football" or row.get("model_key") not in ("v8","v8_near"):
                    continue
                kickoff=utc(row["kickoff_utc"])
                if kickoff+timedelta(hours=3)>clock:
                    continue
                fid=str(int(row["fixture_id"]))
                latest_ids.add(fid)
                if fid in existing or fid in candidates:
                    continue
                frozen=original_model(ROOT,row)
                if not frozen or frozen.get("input_digest")!=row.get("input_digest"):
                    status["skipped_identity"]+=1
                    continue
                if not all(frozen.get(k) is not None for k in ("home_team_id","away_team_id","kickoff_utc","forecast_id")):
                    status["skipped_identity"]+=1
                    continue
                candidates.setdefault(fid,frozen)
            except (KeyError,ValueError,TypeError,OverflowError):
                status["skipped_identity"]+=1
        status["eligible_unverified"]=len(candidates)
        status["original_slate_settlement"]["due_frozen_not_in_latest_forecast_index"] = sum(
            fid not in latest_ids for fid in priority)
        # Frozen research fixtures ALWAYS take precedence over legacy rows.
        # Both arms share the original provider-call budget (max 12*20).
        priority_order=sorted(priority.items(),key=lambda x:utc(x[1]["kickoff_utc"]))
        other_order=sorted(
            ((fid,source) for fid,source in candidates.items() if fid not in priority),
            key=lambda x:utc(x[1]["kickoff_utc"]))
        ordered=priority_order+other_order
        selected=ordered[:MAX_CALLS*20]
        selected_ids={fid for fid,_ in selected}
        status["original_slate_settlement"]["due_frozen_ids_prioritized"] = sum(
            fid in selected_ids for fid in priority)
        status["original_slate_settlement"]["due_frozen_ids_deferred_by_budget"] = sum(
            fid not in selected_ids for fid in priority)
        status["requested_fixture_ids"]=len(selected)
        for offset in range(0,len(selected),20):
            group=selected[offset:offset+20]
            status["requests"]+=1
            try:
                result=request(key,"/fixtures",{"ids":"-".join(fid for fid,_ in group)},with_metadata=True)
                if (result.get("metadata") or {}).get("stale"):
                    continue
                items=result.get("data") or []
                if not isinstance(items,list):
                    continue
                status["provider_rows_returned"]=status.get("provider_rows_returned",0)+len(items)
                for item in items:
                    fixture=item.get("fixture") or {}
                    fid=str(fixture.get("id"))
                    frozen=candidates.get(fid)
                    if frozen is None or fid not in dict(group):
                        status["skipped_identity"]+=1
                        continue
                    try:
                        _accept_verified(item,fid,frozen,existing,status,clock)
                    except (KeyError,TypeError,ValueError,OverflowError):
                        # One malformed provider fixture must not prevent the
                        # remaining independently identified IDs in this batch
                        # from being settled.
                        status["skipped_identity"]+=1
                        continue
            except (ProviderQuotaError,ProviderAccessError,ProviderRejectedError) as exc:
                status["state"]=type(exc).__name__
                break
            except (KeyError,TypeError,ValueError,OverflowError):
                status["skipped_identity"]+=1
        if status["new_verified"]:
            RESULTS.parent.mkdir(parents=True,exist_ok=True)
            RESULTS.write_text(json.dumps(existing,indent=2,sort_keys=True)+"\n")
    except (OSError,ValueError,TypeError,KeyError) as exc:
        status["state"]=type(exc).__name__
    return save(status)

def save(status):
    STATUS.parent.mkdir(parents=True,exist_ok=True)
    STATUS.write_text(json.dumps(status,indent=2,sort_keys=True)+"\n")
    print(json.dumps(status,sort_keys=True))
    return status

if __name__=="__main__":
    run()
