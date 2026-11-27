"""Opt-in V44 collector adapter; reuse unchanged provider validation.

The caller must reserve the existing collector's attempt and adapter-call
budget before calling. No scheduler, new API allowance or archive write is
introduced here. The existing V21/V42 frozen implementation is not edited.
"""
from collections import Counter
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path

from source_capture_clock import capture_clock
from v10_prospective import utc
from v21_provider_evidence import capture, valid_snapshot
import v44_venue_sampling_research as sampling
import verified_xg_evidence_pipeline as archive


def prepare(fixture, history, cutoff, *, limit=6):
    """Select history without mutating the full source input or timestamps."""
    cutoff=utc(cutoff)
    report=sampling.plan(fixture,history,cutoff,limit)
    desired=report["venue_aware"]["fixture_ids"]
    _,_,eligible=sampling._candidates(fixture,history,cutoff)
    qualified={r[1]:r for r in eligible}
    originals={}
    for row in history:
        try:
            fid=str(row["id"])
            signature=(utc(row["utcDate"]),fid,str(row["homeTeam"]["id"]),str(row["awayTeam"]["id"]))
            if (qualified.get(fid)==signature and row.get("provider")=="API-Football"
                    and row.get("status")=="FINISHED"):
                originals.setdefault(fid,row)
        except (KeyError,TypeError,ValueError,AttributeError):
            continue
    def subset(ids):
        return [originals[fid] for fid in ids]
    selected=subset(desired)
    replay=sampling.plan(fixture,selected,cutoff,limit)["recent"]
    fallback=set(replay["fixture_ids"])!=set(desired)
    if fallback:
        # The unchanged provider capture still enforces its per-team selection
        # rules; never promise that a plan it cannot execute will be captured.
        selected=subset(report["recent"]["fixture_ids"])
        replay=sampling.plan(fixture,selected,cutoff,limit)["recent"]
    if replay["requests"]>report["recent"]["requests"]:
        raise ValueError("historical_request_budget_exceeded")
    return selected,{"policy":"venue_aware_v44_opt_in",
                     "fallback_to_recent":fallback,"original":report["recent"],
                     "executable":replay,"additional_historical_requests":0}


def capture_venue_aware(fixture,history,api_key,*,enabled=False,reserved_adapter_calls=0,
                        now=None,clock=None,request=None,limit=6,
                        include_provider_prediction=True):
    """Execute the candidate in a caller-authorized, already reserved slot.

    Reservation bounds adapter invocations, not underlying HTTP retry counts.
    Keep the original provider helper so certified odds metadata still works.
    Return diagnostics separately from the immutable provider snapshot.
    """
    if enabled is not True:
        return {"status":"abstained","reason":"research_collection_disabled"}
    if not api_key:
        return {"status":"abstained","reason":"missing_provider_key"}
    started,observed=capture_clock(now,clock)
    if not started < utc(fixture["utcDate"])-timedelta(minutes=5):
        return {"status":"abstained","reason":"outside_pre_match_window"}
    selected,report=prepare(fixture,history,started,limit=limit)
    maximum=3+int(bool(include_provider_prediction))+report["executable"]["requests"]
    if type(reserved_adapter_calls) is not int or reserved_adapter_calls<maximum:
        return {"status":"abstained","reason":"insufficient_reserved_adapter_calls",
                "required_adapter_slots":maximum,"sampling":report}
    # Preserve the initial cutoff while the observation clock advances.
    packet=capture(fixture,selected,api_key,now=started,clock=observed,
                   request=request,max_history_per_team=limit,
                   include_provider_prediction=include_provider_prediction)
    calls=(packet.get("diagnostics") or {}).get("queries",0) if isinstance(packet,dict) else 0
    if not valid_snapshot(packet,fixture,now=observed()):
        return {"status":"abstained","reason":"unqualified_provider_snapshot",
                "sampling":report,"adapter_calls":calls}
    actual=(packet.get("diagnostics") or {}).get("historical_stats_fixtures_queried")
    if type(actual) is not int or actual!=report["executable"]["requests"]:
        return {"status":"abstained","reason":"unexpected_historical_request_count",
                "sampling":report,"adapter_calls":calls}
    return {"status":"captured","provider_snapshot":packet,"sampling":report,
            "reserved_adapter_calls":reserved_adapter_calls,
            "maximum_adapter_calls":maximum,"automatic_promotion":False}


def evaluate(*,root=archive.ROOT,provider_root=archive.PROVIDER_ROOT,now=None):
    clock=utc(now or datetime.now(timezone.utc))
    counts,rejections=Counter(),Counter()
    for path in sorted((Path(root)/"history").glob("*.json")):
        try:
            record=archive._authenticated_record(path,provider_root)
            if record is None or utc(record["capture"])>clock:
                rejections["invalid_record"]+=1
                continue
            raw=json.loads(path.read_text(encoding="utf-8"))
            limit=record["snapshot"]["source_limits"]["max_historical_fixtures_per_team"]
            _,report=prepare(record["fixture"],record["history"],raw["as_of_utc"],limit=limit)
            counts["authenticated_records"]+=1
            counts["original_ready"]+=int(report["original"]["potentially_ready"])
            counts["executable_ready"]+=int(report["executable"]["potentially_ready"])
            counts["fallbacks"]+=int(report["fallback_to_recent"])
            counts["original_request_slots"]+=report["original"]["requests"]
            counts["executable_request_slots"]+=report["executable"]["requests"]
        except (KeyError,TypeError,ValueError,OSError,AttributeError,ArithmeticError):
            rejections["unqualified_adapter_plan"]+=1
    return {"study_type":"venue_aware_collector_adapter_dry_run","counts":dict(counts),
            "rejections":dict(rejections),"external_requests":0,
            "live_collector_changed":False,"automatic_promotion":False,
            "limitations":["Dry run verifies executable sampling only.",
                           "Caller must reserve real collector quota and serialize writes.",
                           "No genuine new provider data or accuracy claim."]}


if __name__=="__main__":
    print(json.dumps(evaluate(),indent=2,allow_nan=False))
