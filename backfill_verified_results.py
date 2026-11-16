"""Bounded, identity-checked API-Football final-score backfill for frozen V8 fixtures."""
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from api_football import _request, ProviderQuotaError, ProviderAccessError, ProviderRejectedError
from v10_prospective import ROOT, read_latest, utc
from v16_pair_research import original_model

RESULTS = Path("data/prematch/verified_results.json")
STATUS = Path("data/diagnostics/result_backfill.json")
MAX_CALLS = 12

def run(request=_request, now=None):
    clock = utc(now) if now else datetime.now(timezone.utc)
    key = os.environ.get("API_FOOTBALL_KEY", "")
    status = {"at_utc":clock.isoformat(),"requests":0,"new_verified":[],
              "skipped_identity":0,"eligible_unverified":0,"state":"complete"}
    if not key:
        status["state"]="missing_provider_key"
        return save(status)
    try:
        originals=read_latest(ROOT)
        existing=json.loads(RESULTS.read_text()) if RESULTS.exists() else {}
        if not isinstance(existing,dict):
            status["state"]="invalid_existing_results"
            return save(status)
        candidates={}
        for row in originals.values() if isinstance(originals,dict) else originals:
            try:
                if row.get("provider")!="API-Football" or row.get("model_key") not in ("v8","v8_near"):
                    continue
                kickoff=utc(row["kickoff_utc"])
                if kickoff+timedelta(hours=3)>clock:
                    continue
                fid=str(int(row["fixture_id"]))
                if fid in existing:
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
        for fid,frozen in sorted(candidates.items(),key=lambda x:x[1]["kickoff_utc"])[:MAX_CALLS]:
            status["requests"]+=1
            try:
                result=request(key,"/fixtures",{"id":int(fid)},with_metadata=True)
                if (result.get("metadata") or {}).get("stale"):
                    continue
                items=result.get("data") or []
                if len(items)!=1:
                    continue
                item=items[0]
                fixture=item.get("fixture") or {}
                teams=item.get("teams") or {}
                home=teams.get("home") or {}
                away=teams.get("away") or {}
                goals=item.get("goals") or {}
                state=(fixture.get("status") or {}).get("short")
                # FT alone is a verified regulation-time score. Avoid AET/PEN
                # unless a separate, independently supplied 90-minute score exists.
                if state!="FT":
                    continue
                if (str(fixture.get("id"))!=fid or
                    str(home.get("id"))!=str(frozen["home_team_id"]) or
                    str(away.get("id"))!=str(frozen["away_team_id"]) or
                    utc(fixture["date"])!=utc(frozen["kickoff_utc"])):
                    status["skipped_identity"]+=1
                    continue
                hg,ag=goals.get("home"),goals.get("away")
                if type(hg)!=int or type(ag)!=int or hg<0 or ag<0:
                    continue
                existing[fid]={"provider":"API-Football","fixture_id":int(fid),
                    "home_team_id":int(home["id"]),"away_team_id":int(away["id"]),
                    "home_team":home.get("name"),"away_team":away.get("name"),
                    "kickoff_utc":utc(fixture["date"]).isoformat(),
                    "home_goals":hg,"away_goals":ag,"score_basis":"regulation",
                    "status":"FT","verified_at_utc":clock.isoformat(),
                    "evidence_version":2}
                status["new_verified"].append(fid)
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
