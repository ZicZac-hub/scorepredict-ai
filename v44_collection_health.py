"""Read-only operational evidence monitor; not a prediction model.

Published report timestamps cannot prove whether a scheduled job failed,
was delayed, or is still running. Never infer a live provider outage from
an old repository report. No network, secrets, or model changes.
"""
from datetime import datetime,timedelta,timezone
from pathlib import Path
import json
import argparse

ROOT=Path("data")
CAPTURE_GRACE=timedelta(minutes=45)
HEAVY_GRACE=timedelta(hours=6,minutes=45)

def utc(value):
    if isinstance(value,str):
        value=datetime.fromisoformat(value.replace("Z","+00:00"))
    if not isinstance(value,datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timezone_required")
    return value.astimezone(timezone.utc)

def read(path):
    try:
        v=json.loads(path.read_text(encoding="utf-8"))
        return v if isinstance(v,dict) else None
    except (OSError,ValueError,TypeError):
        return None

def timestamp(report,field,now,grace):
    if report is None or field not in report:
        return {"status":"missing_or_unreadable","timestamp_utc":None,"age_seconds":None}
    try:
        at=utc(report[field])
        age=now-at
        status="future_timestamp" if age<timedelta(0) else "overdue" if age>grace else "within_grace"
        return {"status":status,"timestamp_utc":at.isoformat(),
                "age_seconds":round(age.total_seconds(),3),
                "grace_seconds":int(grace.total_seconds())}
    except (ValueError,TypeError,OverflowError):
        return {"status":"invalid_timestamp","timestamp_utc":None,"age_seconds":None}

def evaluate(*,root=ROOT,now=None):
    root=Path(root)
    now=utc(now or datetime.now(timezone.utc))
    coverage=read(root/"prospective_ledger"/"capture_coverage.json")
    cadence=read(root/"prospective_ledger"/"cadence.json")
    protocol=read(root/"v44_fresh_history_study"/"protocol.json")
    readiness=read(root/"v44_fresh_history_study"/"readiness.json")
    capture=timestamp(coverage,"collector_started_at_utc",now,CAPTURE_GRACE)
    heavy=timestamp(cadence,"last_heavy_at_utc",now,HEAVY_GRACE)
    registration=timestamp(protocol,"registered_at_utc",now,timedelta.max)
    # A published run started before registration is not evidence that
    # post-registration collection has run.
    post_registration=None
    if (capture["status"] in ("within_grace","overdue")
            and registration["status"]=="within_grace"):
        post_registration=utc(capture["timestamp_utc"])>=utc(registration["timestamp_utc"])
    fields=("v44_pair_refresh_reserved","v44_recovery_reserved")
    diagnostics={field:None for field in fields}
    for field in fields:
        value=(coverage or {}).get(field)
        if type(value) is int and value>=0:
            diagnostics[field]=value
    statuses=[]
    if capture["status"]!="within_grace":
        statuses.append("inspect_capture_workflow_and_publication")
    if heavy["status"]!="within_grace":
        statuses.append("inspect_heavy_refresh_completion")
    if post_registration is False:
        statuses.append("no_published_capture_started_after_registration")
    if diagnostics["v44_recovery_reserved"] is None:
        statuses.append("recovery_execution_not_confirmed_by_report")
    if readiness is None:
        statuses.append("readiness_report_missing")
    return {"report_type":"v44_collection_health_v1","checked_at_utc":now.isoformat(),
            "capture":capture,"heavy_completion":heavy,"study_registration":registration,
            "published_capture_started_after_registration":post_registration,
            "published_diagnostics":diagnostics,
            "coverage_complete":(coverage or {}).get("coverage_complete") if type((coverage or {}).get("coverage_complete")) is bool else None,
            "actions_to_investigate":statuses,"external_requests":0,
            "limitations":["Repository timestamps are publication evidence, not job status.",
                           "Missing diagnostics do not prove a disabled flag or failed job.",
                           "Grace thresholds are operational defaults, not provider guarantees.",
                           "No forecasts are changed or generated."]}

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=ROOT)
    args=parser.parse_args()
    print(json.dumps(evaluate(root=args.root),indent=2,allow_nan=False))
