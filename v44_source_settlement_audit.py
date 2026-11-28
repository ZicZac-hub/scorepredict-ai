"""Read-only source timing and V37 pending-fixture audit. No provider calls."""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path

from v10_prospective import ROOT, read_latest, utc
from professional_analyst_shadow_v25 import _eligible
from forecast_ledger import read_file_records
from v37_sparse_history_research import frozen_arm_digest
from v10_evaluate import _qualified
from verified_result_index import index_verified_results
from v10_settlement import MIN_FINAL_LAG

def time_reason(row, clock):
    try:
        e,c,k=(utc(row[x]) for x in ("evidence_as_of_utc","captured_at_utc","kickoff_utc"))
    except (KeyError,ValueError,TypeError,OverflowError):
        return "unparseable_time"
    if e>c:
        return "evidence_after_capture"
    if c>=k:
        return "capture_at_or_after_kickoff"
    if c>clock:
        return "capture_after_cutoff"
    if clock>=k:
        return "fixture_already_started"
    return "time_valid"

def timing(rows, clock):
    counts=Counter()
    for row in rows:
        _, reason=_eligible(row,clock)
        if reason=="invalid_source_time":
            counts[time_reason(row,clock)]+=1
    return {"invalid_source_time_total":sum(counts.values()),"breakdown":dict(counts)}

def pending(rows, finals, conflicts, clock, attempts, error_ids):
    groups=defaultdict(list)
    for row in rows:
        if row.get("model_key")=="v37_sparse":
            groups[(str(row.get("provider")),str(row.get("fixture_id")))].append(row)
    out=[]
    for key, group in sorted(groups.items()):
        row=group[0]
        try:
            if (len(group)!=1 or row.get("v37_research_only") is not True
                or row.get("v37_no_live_model_change") is not True
                or row.get("v37_not_eligible_for_v28") is not True
                or row["v37_arm_digest_sha256"]!=frozen_arm_digest(row)
                or not utc(row["evidence_as_of_utc"])<=utc(row["captured_at_utc"])<utc(row["kickoff_utc"])):
                raise ValueError("invalid_original")
            kickoff=utc(row["kickoff_utc"])
            final=finals.get(key)
            if key in conflicts:
                reason="conflicting_verified_finals"
            elif final is not None and _qualified(row,final):
                reason="verified"
            elif final is not None:
                reason="verified_final_identity_mismatch"
            elif kickoff+MIN_FINAL_LAG>clock:
                reason="awaiting_final_lag"
            elif key[0]=="API-Football" and key[1] in error_ids:
                reason="provider_error_in_saved_status"
            elif key[0]=="API-Football" and key[1] in attempts:
                reason="due_previously_attempted"
            else:
                reason="due_without_recorded_attempt"
        except (ValueError,KeyError,TypeError,OverflowError):
            reason="invalid_original"
        out.append({"provider":key[0],"fixture_id":key[1],"kickoff_utc":row.get("kickoff_utc"),"reason":reason})
    return {"unique_fixtures":len(out),"counts":dict(Counter(x["reason"] for x in out)),"rows":out}

def evaluate(root=ROOT,prematch=Path("data/prematch"),now=None):
    root=Path(root)
    clock=utc(now or datetime.now(timezone.utc))
    def load(name):
        p=root/name
        return json.loads(p.read_text()) if p.exists() else {}
    finals,conflicts,rejected=index_verified_results(root,prematch,clock)
    attempts=load("result_attempts.json")
    status=load("settlement_status.json")
    errors={str(x["fixture_id"]) for x in status.get("source_errors",[]) if "fixture_id" in x}
    return {"as_of_utc":clock.isoformat(),"source_timing":timing(read_latest(root).values(),clock),
            "v37":pending(read_file_records(root,kind="forecasts"),finals,conflicts,clock,attempts,errors),
            "final_rejections":dict(rejected),"external_requests":0,"automatic_promotion":False,
            "limitations":["Timing counts are evaluated at this report cutoff, not necessarily the earlier bot cutoff.",
                           "Saved provider errors describe the last stored settlement report, not current provider availability."]}

if __name__=="__main__":
    print(json.dumps(evaluate(),indent=2,allow_nan=False))
