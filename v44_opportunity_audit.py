"""Read-only V44 research evaluation: retain the opportunity denominator.

Fixed first 100 eligible original pair opportunities, not first 100
successful predictions or first 100 completed results. Missing/abstained
closed windows count as no successful forecast, not as a model score loss.
Inspired by selective prediction risk/coverage and proper scoring:
https://proceedings.mlr.press/v97/geifman19a.html
https://doi.org/10.1146/annurev-statistics-042424-050626
No changes to pair selection, score probabilities, or production model.
"""
from collections import Counter
from datetime import datetime,timedelta,timezone
import json,math
from pathlib import Path

from v10_prospective import utc
from five_double_accuracy import wilson
import v44_ratings_prospective as trial
import v44_early_history as early
import v42_research_trial as sources

COHORT_SIZE=100
TARGET=.70
MIN_LEAD=timedelta(minutes=5)
MAX_LEAD=timedelta(hours=3)


def summarize(opportunities,settled_rows,now,*,integrity_ok=True):
    """Input must be authenticated by evaluate; no result-dependent cohort choice."""
    now=utc(now)
    ordered=sorted(opportunities,key=lambda r:(utc(r["window_opens_utc"]),r["day"]))
    if len({r["day"] for r in ordered})!=len(ordered):
        raise ValueError("duplicate_opportunity")
    cohort=ordered[:COHORT_SIZE]
    if len(settled_rows)!=len({r["day"] for r in settled_rows}):
        raise ValueError("duplicate_settlement")
    settled={r["day"]:r["metrics"] for r in settled_rows}
    seen=set()
    seen_events=set()
    closed=issued=verified=missed=pending=open_windows=0
    measures={name:[] for name in trial.METHODS}
    days=[]
    for row in cohort:
        ids={tuple(i) for i in row["fixture_ids"]}
        events={tuple(i) for i in row["events"]}
        if len(ids)!=2 or len(events)!=2 or seen&ids or seen_events&events:
            integrity_ok=False
        seen.update(ids);seen_events.update(events)
        day=row["day"]
        is_closed=utc(row["window_closes_utc"])<=now
        is_issued=row["issued"] is True
        metrics=settled.get(day)
        if metrics is not None and not is_issued:
            integrity_ok=False
        closed+=int(is_closed)
        issued+=int(is_issued)
        if metrics is not None and is_issued:
            verified+=1
            for name in trial.METHODS:
                measures[name].append(metrics[name])
            state="verified"
        elif is_issued:
            pending+=1
            state="issued_pending"
        elif is_closed:
            missed+=1
            state="closed_without_trial"
        else:
            open_windows+=1
            state="window_not_closed"
        days.append({"day":day,"status":state})
    complete=(len(cohort)==COHORT_SIZE and closed==COHORT_SIZE and
              pending==0 and verified+missed==COHORT_SIZE and integrity_ok)
    by_method={}
    for name,metrics in measures.items():
        hits=sum(int(m["hit"]) for m in metrics)
        if len(metrics)!=verified:
            raise ValueError("inconsistent_comparative_cohort")
        probs=[m["estimated_coverage"] for m in metrics]
        if any(type(p) not in (int,float) or not math.isfinite(p) or not 0<=p<=1 for p in probs):
            raise ValueError("invalid_model_probability")
        interval=wilson(hits,COHORT_SIZE) if complete else None
        by_method[name]={
            "hits":hits,"verified_issued_hit_rate":hits/verified if verified else None,
            "verified_issued_wilson_95":wilson(hits,verified),
            "mean_model_estimated_five_score_mass":math.fsum(probs)/verified if verified else None,
            "mean_set_brier":math.fsum((p-int(m["hit"]))**2 for p,m in zip(probs,metrics))/verified if verified else None,
            "mean_per_leg_log_loss":math.fsum(m["score_log_loss"] for m in metrics)/verified if verified else None,
            "fixed_cohort_success_rate":hits/COHORT_SIZE if complete else None,
            "fixed_cohort_wilson_95":interval,
            "target_met":bool(complete and interval and interval[0]>=TARGET)}
    return {"cohort_size":COHORT_SIZE,"target":TARGET,"opportunities_in_cohort":len(cohort),
            "closed_windows":closed,"issued":issued,"verified_issued":verified,
            "closed_without_trial":missed,"issued_pending":pending,"open_or_future_windows":open_windows,
            "closed_window_issuance_rate":(sum(d["status"] in ("verified","issued_pending")
                and utc(row["window_closes_utc"])<=now for d,row in zip(days,cohort))/closed if closed else None),
            "integrity_ok":integrity_ok,"fixed_review_complete":complete,
            "methods":by_method,"days":days}


def evaluate(*,root=trial.ROOT,history_root=early.ROOT,pair_root=sources.PAIR_ROOT,
             source_root=sources.SOURCE_ROOT,prematch_root=sources.PREMATCH_ROOT,now=None,report=None):
    root=Path(root)
    now=utc(now or datetime.now(timezone.utc))
    if report is None:
        report=trial.evaluate(root=root,history_root=history_root,pair_root=pair_root,
                              source_root=source_root,prematch_root=prematch_root,now=now)
    if not (root/"protocol.json").exists():
        summary=summarize([],[],now,integrity_ok=not list((root/"trials").glob("*.json")))
        return {"study_type":"v44_fixed_opportunity_audit","status":"not_registered",
                "summary":summary,"external_requests":0,"automatic_promotion":False}
    protocol=trial._protocol(root,now)
    registered=utc(protocol["registered_at_utc"])
    rejections=Counter()
    opportunities=[]
    trial_days={p.stem for p in (root/"trials").glob("*.json")}
    for path in sorted((Path(pair_root)/"sets").glob("*.json")):
        try:
            pair,models=sources._pair(path,source_root=source_root,decision=now)
            first=min(utc(m["kickoff_utc"]) for m in models)
            opens=max(registered,utc(pair["predicted_at_utc"]),first-MAX_LEAD)
            closes=first-MIN_LEAD
            # A pre-registration closed window is never counted against V44.
            if opens>=closes or opens>now:
                continue
            opportunities.append({"day":path.stem,"window_opens_utc":opens.isoformat(),
                "window_closes_utc":closes.isoformat(),"issued":path.stem in trial_days,
                "fixture_ids":[[str(m["provider"]),str(m["fixture_id"])] for m in models],
                "events":[[utc(m["kickoff_utc"]).isoformat(),
                          *(" ".join(m[k].casefold().split()) for k in ("home_team","away_team"))] for m in models]})
        except (ValueError,KeyError,TypeError,OSError,AttributeError,ArithmeticError) as exc:
            rejections[str(exc) if isinstance(exc,ValueError) else type(exc).__name__]+=1
    opportunity_days={r["day"] for r in opportunities}
    integrity_ok=(not rejections and not report["counts"].get("rejected")
                  and not report.get("result_conflicts")
                  and trial_days<=opportunity_days
                  and report["counts"].get("authenticated_issued",0)==len(trial_days))
    summary=summarize(opportunities,report["rows"],now,integrity_ok=integrity_ok)
    return {"study_type":"v44_fixed_opportunity_audit","status":"registered",
            "registered_at_utc":registered.isoformat(),"all_eligible_opportunities":len(opportunities),
            "summary":summary,"original_pair_rejections":dict(rejections),
            "external_requests":0,"automatic_promotion":False,
            "limitations":["First 100 eligible original pair windows are fixed by chronology, not outcomes.",
                           "Closed windows without an issued trial count as no successful forecast, not a probability loss.",
                           "Wilson intervals assume independent observations; shared league or team effects may violate that assumption.",
                           "No estimate establishes a guaranteed future hit rate."]}


if __name__=="__main__":
    print(json.dumps(evaluate(),indent=2,allow_nan=False))
