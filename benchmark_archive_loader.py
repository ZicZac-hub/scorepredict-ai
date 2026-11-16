"""Read-only conversion of frozen ZicZac forecasts and verified final scores.

Only original prospective snapshots with a recorded pre-kickoff capture and
independently verified regulation-time FT result are eligible. Retrospective
rescores never count as prospective predictions.
"""
import json
from pathlib import Path

ROOTS=("data/v10_prospective","data/prematch")
MAX_FILES=500
MAX_BYTES=4_000_000

def _read(path):
    if path.stat().st_size>MAX_BYTES:
        raise ValueError("oversized")
    return json.loads(path.read_text(encoding="utf-8"))

def _score_rows(matrix):
    if not isinstance(matrix,list) or not matrix:
        return None
    rows=[]
    try:
        for h,line in enumerate(matrix):
            if not isinstance(line,list):
                return None
            for a,prob in enumerate(line):
                rows.append([h,a,float(prob)])
    except (TypeError,ValueError):
        return None
    return rows

def load_archive(root=".", limit=MAX_FILES):
    base=Path(root).resolve()
    diagnostics={"files_checked":0,"compatible_records":0,"skipped_records":0,
                 "unreadable_files":0,"truncated":False,
                 "verified_fixtures":0,"frozen_forecasts":0,
                 "missing_forecasts":0}
    records={}
    verified_file=base/"data/v10_prospective/verified_results.json"
    try:
        verified=_read(verified_file)
        if not isinstance(verified,dict):
            verified={}
    except (OSError,ValueError,TypeError,UnicodeError):
        verified={}
    valid={}
    for fid,result in verified.items():
        if not isinstance(result,dict):
            continue
        if str(result.get("status","")).upper() not in ("FT","AET","PEN"):
            continue
        if result.get("score_basis")!="regulation":
            continue
        h,a=result.get("home_goals"),result.get("away_goals")
        if type(h) is not int or type(a) is not int or min(h,a)<0:
            continue
        key=str(result.get("fixture_id",fid))
        valid[key]=result
    diagnostics["verified_fixtures"]=len(valid)
    max_count=min(MAX_FILES,max(0,int(limit)))
    for fid,result in sorted(valid.items()):
        provider=str(result.get("provider") or "API-Football")
        location=base/"data/v10_prospective/forecasts"/provider/fid
        if not location.is_dir():
            diagnostics["missing_forecasts"]+=1
            continue
        for path in sorted(location.glob("*.json")):
            if diagnostics["files_checked"]>=max_count:
                diagnostics["truncated"]=True
                break
            diagnostics["files_checked"]+=1
            try:
                forecast=_read(path)
                if not isinstance(forecast,dict):
                    diagnostics["skipped_records"]+=1
                    continue
                if str(forecast.get("fixture_id"))!=fid or forecast.get("mode")!="prospective":
                    diagnostics["skipped_records"]+=1
                    continue
                scores=_score_rows(forecast.get("score_matrix"))
                if not scores:
                    diagnostics["skipped_records"]+=1
                    continue
                from model_comparison_audit import _time
                stamp=_time(forecast.get("captured_at_utc"))
                kickoff=_time(forecast.get("kickoff_utc"))
                if stamp is None or kickoff is None or stamp>=kickoff:
                    diagnostics["skipped_records"]+=1
                    continue
                if _time(result.get("kickoff_utc"))!=kickoff:
                    diagnostics["skipped_records"]+=1
                    continue
                if str(forecast.get("provider"))!=provider:
                    diagnostics["skipped_records"]+=1
                    continue
                model=str(forecast.get("model_version") or forecast.get("model_key") or "unknown")
                row=records.setdefault(fid,{"fixture_id":provider+":"+fid,
                    "kickoff":forecast["kickoff_utc"],
                    "actual":[result["home_goals"],result["away_goals"]],
                    "forecasts":{}})
                candidate={"created_at":forecast["captured_at_utc"],"scores":scores}
                previous=row["forecasts"].get(model)
                # Keep the earliest valid frozen forecast per model and fixture.
                if previous is None or stamp<_time(previous["created_at"]):
                    row["forecasts"][model]=candidate
                diagnostics["frozen_forecasts"]+=1
            except (OSError,ValueError,TypeError,UnicodeError):
                diagnostics["unreadable_files"]+=1
        if diagnostics["truncated"]:
            break
    rows=list(records.values())
    diagnostics["compatible_records"]=len(rows)
    return rows,diagnostics
