"""Read-only operational diagnostics for newly merged V44 research.

Distinguish absence, chronology, source identity and model readiness.
Do not generate forecasts, select pairs, retrieve external data, or modify
any original forecast or registered study implementation.
"""
from collections import Counter
from datetime import datetime,timedelta,timezone
import json
from pathlib import Path

from v10_prospective import utc
from v44_early_history_audit import qualifies
import v44_early_history as early
import v44_score_driven_research as ratings
import v44_ratings_prospective as trial
import v42_research_trial as sources


def history_status(model,cutoff,root):
    event=[str(model["provider"]),str(model["fixture_id"]),utc(model["kickoff_utc"]).isoformat(),
           str(model["home_team_id"]),str(model["away_team_id"])]
    keys=(early.receipt_key(event,model["input_digest"]),early.digest(event))
    failures=set()
    for key in keys:
        path=Path(root)/"receipts"/(key+".json")
        if not path.is_file():
            continue
        try:
            raw=early._read(path)
            if utc(raw["captured_at_utc"])>cutoff:
                failures.add("receipt_after_pair_cutoff")
                continue
            record=early.load(path,root=root,now=cutoff)
            if not qualifies(record,model,cutoff):
                failures.add("receipt_does_not_match_original_forecast")
                continue
            result=ratings.predict(record["fixture"],record["history"],record["as_of_utc"])
            return "ready" if result["status"]=="ready" else "insufficient_team_or_league_history"
        except (ValueError,KeyError,TypeError,OSError,AttributeError,ArithmeticError):
            failures.add("invalid_history_receipt")
    for status in ("invalid_history_receipt","receipt_does_not_match_original_forecast","receipt_after_pair_cutoff"):
        if status in failures:
            return status
    return "missing_exact_history_receipt"


def evaluate(*,root=trial.ROOT,history_root=early.ROOT,pair_root=sources.PAIR_ROOT,
             source_root=sources.SOURCE_ROOT,now=None):
    clock=utc(now or datetime.now(timezone.utc))
    root=Path(root)
    registered=None
    if (root/"protocol.json").is_file():
        registered=utc(trial._protocol(root,clock)["registered_at_utc"])
    counts=Counter()
    statuses=Counter()
    for path in sorted((Path(pair_root)/"sets").glob("*.json")):
        try:
            pair,models=sources._pair(path,source_root=source_root,decision=clock)
            original=utc(pair["predicted_at_utc"])
            kickoff=min(utc(m["kickoff_utc"]) for m in models)
            closes=kickoff-timedelta(minutes=5)
            opens=max(original,kickoff-timedelta(hours=3),registered or clock)
            if registered is None:
                counts["study_not_registered"]+=1
                continue
            if closes<=opens:
                counts["window_closed_before_registration_or_source"]+=1
                continue
            if opens>clock:
                counts["future_window"]+=1
                continue
            stage="open" if clock<closes else "closed"
            issued=(root/"trials"/path.name).is_file()
            counts[stage+"_pair_opportunities"]+=1
            counts[stage+"_with_trial_file"]+=int(issued)
            # Use original decision for historical availability; later
            # receipts never repair a missing original source retrospectively.
            legs=[history_status(m,original,history_root) for m in models]
            for status in legs:
                statuses[stage+":"+status]+=1
            counts[stage+"_both_histories_ready"]+=int(legs==["ready","ready"])
        except (ValueError,KeyError,TypeError,OSError,AttributeError,ArithmeticError):
            counts["invalid_original_pair"]+=1
    return {"study_type":"v44_operational_evidence_diagnostics",
            "registered":registered is not None,"counts":dict(counts),
            "leg_statuses":dict(statuses),"external_requests":0,
            "automatic_promotion":False,
            "limitations":["Counts diagnose data readiness, not accuracy.",
                           "A trial file must still pass the registered trial integrity evaluation.",
                           "Original history must be available before its forecast and original pair decision."]}


if __name__=="__main__":
    print(json.dumps(evaluate(),indent=2,allow_nan=False))
