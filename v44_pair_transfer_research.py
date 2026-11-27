"""Transfer chronological single-fixture calibration to original two-match trials.

The original fixed pair and its pre-match cutoff are never changed. Training
uses only independently matched single-fixture finals known strictly before
that cutoff. No fabricated pair, API calls, or live promotion.
"""
from collections import Counter
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import numpy as np

from forecast_ledger import _forecast, LedgerValidationError
from five_double_accuracy import _final, _result_index, wilson
from professional_double_shadow import _validated
from professional_double_optimizer import optimal_five
from v10_prospective import utc
import v42_research_trial as sources
import v44_fixture_evidence as inventory
from v44_single_calibration import _single_groups, _prob, MIN_TRAIN_FIXTURES
from v44_family_pool_research import _grid, _weights
from v43_prediction import combine_grids


def _identity(model):
    return str(model["provider"]),str(model["fixture_id"])


def load_training(*,source_root,prematch_root,provider_root,clock,family_fn=_single_groups):
    report=inventory.evaluate(source_root=source_root,prematch_root=prematch_root,
                              provider_root=provider_root,now=clock)
    history, rejects=[],Counter()
    for row in report["rows"]:
        try:
            path=Path(source_root)/"forecasts"/row["provider"]/row["fixture_id"]/(row["forecast_id"]+".json")
            model=_forecast(json.loads(path.read_text(encoding="utf-8")))
            if (model["forecast_id"]!=row["forecast_id"]
                    or _identity(model)!=(row["provider"],row["fixture_id"])
                    or list(inventory._event(model))!=row["event"]
                    or utc(model["captured_at_utc"])!=utc(row["decision_utc"])
                    or utc(model["kickoff_utc"])!=utc(row["kickoff_utc"])):
                raise ValueError("training_source_changed")
            decision,kickoff,known=map(utc,(row["decision_utc"],row["kickoff_utc"],row["final_known_utc"]))
            if not decision<kickoff<known<=clock:
                raise ValueError("training_time_order_invalid")
            actual=row["actual"]
            if len(actual)!=2 or any(type(v) is not int or v<0 for v in actual):
                raise ValueError("invalid_training_final")
            base=_grid(model["score_matrix"])
            groups=family_fn(base)
            history.append({"provider_id":_identity(model),"event":tuple(row["event"]),
                            "known":known,"kickoff":kickoff,"decision":decision,
                            "base_grid":base,"groups":groups,"actual":actual,"roster":tuple(groups),
                            "likelihood":np.array([[_prob(g,actual) for g in groups.values()]])})
        except (KeyError,ValueError,TypeError,OSError,AttributeError,ArithmeticError,LedgerValidationError) as exc:
            rejects[str(exc) if isinstance(exc,ValueError) else type(exc).__name__]+=1
    return history,rejects


def prior_training(history,models,decision,roster):
    ids={_identity(m) for m in models}
    events={inventory._event(m) for m in models}
    eligible=[]
    for row in history:
        if (row["known"]<decision and row["kickoff"]<row["known"]
                and row["provider_id"] not in ids and row["event"] not in events
                and row["roster"]==roster):
            eligible.append(row)
    return [r["likelihood"] for r in eligible]


def _arm(grids):
    choices=optimal_five(*grids)
    if len(choices)!=5 or len({(c["score1"],c["score2"]) for c in choices})!=5:
        raise ValueError("five_distinct_positive_combinations_required")
    return {"full_score_grids":grids,"five_doubles":choices,
            "estimated_coverage":math.fsum(c["joint_probability"] for c in choices)}


def _summaries(rows):
    result={}
    for name in ("original_v8","v43_statistical","v44_equal","v44_transfer"):
        m=[r["metrics"][name] for r in rows]
        n=len(m)
        hits=sum(v["hit"] for v in m)
        result[name]={"pairs":n,"hits":hits,"hit_rate":hits/n if n else None,
                      "wilson_95":wilson(hits,n),
                      **{k:math.fsum(v[k] for v in m)/n if n else None
                         for k in ("score_log_loss","estimated_coverage","set_brier_descriptive")}}
    return result


def evaluate(*,pair_root=sources.PAIR_ROOT,source_root=sources.SOURCE_ROOT,
             prematch_root=sources.PREMATCH_ROOT,provider_root=sources.PROVIDER_ROOT,now=None,
             family_fn=_single_groups):
    clock=utc(now or datetime.now(timezone.utc))
    training,training_rejects=load_training(source_root=source_root,prematch_root=prematch_root,
                                          provider_root=provider_root,clock=clock,family_fn=family_fn)
    finals,conflicts,result_rejects=_result_index(source_root,prematch_root)
    paths=sorted((Path(pair_root)/"sets").glob("*.json"))
    candidates,rejects=[],Counter()
    for path in paths:
        try:
            original=json.loads(path.read_text(encoding="utf-8"))
            models,choices,coverage=_validated(original,source_root=source_root,now=clock)
            if path.stem!=original["sydney_date"]:
                raise ValueError("filename_day_mismatch")
            candidates.append((utc(original["predicted_at_utc"]),path.stem,models,choices,coverage))
        except (ValueError,KeyError,TypeError,OSError,AttributeError,ArithmeticError) as exc:
            rejects[str(exc) if isinstance(exc,ValueError) else type(exc).__name__]+=1
    rows,seen_ids,seen_events=[],set(),set()
    pending=warmup=0
    for decision,day,models,choices,coverage in sorted(candidates,key=lambda r:r[:2]):
        try:
            ids={_identity(m) for m in models}
            events={inventory._event(m) for m in models}
            if len(ids)!=2 or len(events)!=2 or seen_ids&ids or seen_events&events:
                raise ValueError("repeated_fixture")
            seen_ids.update(ids)
            seen_events.update(events)
            actual=[_final(m,finals.get(_identity(m)),clock) for m in models]
            if any(s is None for s in actual):
                pending+=1
                continue
            base=[_grid(m["score_matrix"]) for m in models]
            groups=[family_fn(g) for g in base]
            roster=tuple(groups[0])
            if tuple(groups[1])!=roster:
                raise ValueError("inconsistent_family_roster")
            prior=prior_training(training,models,decision,roster)
            if len(prior)<MIN_TRAIN_FIXTURES:
                warmup+=1
                continue
            equal=np.full(len(roster),1/len(roster))
            learned=_weights(prior,len(roster))
            stacks=[np.stack(list(g.values())) for g in groups]
            # The transfer model is fixed in advance, not selected on this pair's result.
            equal_grids=[np.tensordot(equal,g,axes=1) for g in stacks]
            learned_grids=[np.tensordot(learned,g,axes=1) for g in stacks]
            control=combine_grids(base,[{},{}])
            if control["status"]!="issued":
                raise ValueError("v43_control_abstained")
            arms={"original_v8":{"full_score_grids":base,"five_doubles":choices,
                                 "estimated_coverage":coverage},
                  "v43_statistical":control,"v44_equal":_arm(equal_grids),
                  "v44_transfer":_arm(learned_grids)}
            metrics={name:sources._metric(arm,actual) for name,arm in arms.items()}
            rows.append({"sydney_date":day,"decision_utc":decision.isoformat(),
                         "fixture_ids":[list(_identity(m)) for m in models],
                         "training_fixtures":len(prior),
                         "weights":dict(zip(roster,map(float,learned))),
                         "metrics":metrics})
        except (ValueError,KeyError,TypeError,OSError,AttributeError,ArithmeticError) as exc:
            rejects[str(exc) if isinstance(exc,ValueError) else type(exc).__name__]+=1
    return {"study_type":"single_to_original_pair_weight_transfer_replay",
            "evaluated_at_utc":clock.isoformat(),"archived_pairs":len(paths),
            "verified_single_training_inventory":len(training),"minimum_prior_fixtures":MIN_TRAIN_FIXTURES,
            "eligible_comparison_pairs":len(rows),"pending_pairs":pending,
            "warmup_pairs":warmup,"comparisons":_summaries(rows),"source_rejections":dict(rejects),
            "training_rejections":dict(training_rejects),"result_rejections":dict(result_rejects),
            "conflicting_result_identities":len(conflicts),"rows":rows,
            "external_requests":0,"new_pairs_created":0,"automatic_promotion":False,
            "limitations":["All transformed models share the V8 foundation.",
                           "Joint score probabilities assume independent matches.",
                           "V43 control is statistical only, not a claim about its evidence-assisted live model.",
                           "Retrospective transfer results are not prospective V44 hit rates.",
                           "No accuracy improvement can be concluded from a small matched cohort."]}


if __name__=="__main__":
    print(json.dumps(evaluate(),indent=2,allow_nan=False))
