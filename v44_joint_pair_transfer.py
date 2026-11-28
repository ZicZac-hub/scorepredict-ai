"""Read-only chronological transfer audit for joint-score rate calibration.

Evaluate only immutable original two-match sets, on identical eligible
fixtures. No fresh selections are published. Both target fixtures and
their aliases are excluded from all training; both finals must be verified.
This replay is exploratory, not an independent prospective accuracy trial.
"""
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path

from v10_prospective import utc
from five_double_accuracy import _final,_result_index
from professional_double_shadow import _validated
from v44_family_pool_research import _grid
from v44_pair_transfer_research import _arm
import v44_market_ablation as archive
import v44_bayesian_rate_calibration as bayes
import v44_joint_rate_tilt as joint
import v44_fixture_evidence as inventory
import v44_v9_pair_replay as replay
import v42_research_trial as sources

ARMS=("original_v8","time_matched_v8","stored_v9","joint_half","joint_full")

def training_records(matches,finals,clock):
    records=[]
    for a,b in matches:
        identity=inventory._identity(a)
        final=finals.get(identity)
        actual=_final(a,final,clock)
        if actual is None or _final(b,final,clock)!=actual:
            continue
        known,decision,kickoff=map(utc,(final["verified_at_utc"],a["captured_at_utc"],a["kickoff_utc"]))
        league=str(a.get("competition") or "").strip()
        if not decision<kickoff<known<=clock or not league:
            continue
        records.append({"identity":identity,"event":inventory._event(a),"league":league,
                        "known":known,"decision":decision,"kickoff":kickoff,"actual":actual,
                        "means":bayes.means(_grid(b["score_matrix"]))})
    return records

def corrections(training,models,legs,decision):
    decision=utc(decision)
    if len(models)!=2 or len(legs)!=2:
        raise ValueError("two_legs_required")
    prior=replay.earlier_training(training,models,decision)
    grids=[];sizes=[]
    for model,leg in zip(models,legs):
        matched=replay.match_leg(model,leg,decision)
        if matched is None or not decision<utc(model["kickoff_utc"]):
            raise ValueError("missing_predecision_market_match")
        a,b=matched
        league=str(a.get("competition") or "").strip()
        if not league:
            raise ValueError("missing_competition")
        g=_grid(b["score_matrix"])
        target={"identity":inventory._identity(a),"event":inventory._event(a),
                "decision":decision,"league":league,"means":bayes.means(g)}
        fitted,n,local=bayes.fit(prior,target)
        if fitted is None:
            raise ValueError("insufficient_prior_fixtures")
        scales=fitted[2]
        grids.append((joint.tilt(g,scales,.5),joint.tilt(g,scales,1.)))
        sizes.append({"prior_fixtures":n,"league_fixtures":local})
    return [g[0] for g in grids],[g[1] for g in grids],sizes

def evaluate(*,source_root=sources.SOURCE_ROOT,prematch_root=sources.PREMATCH_ROOT,
             pair_root=sources.PAIR_ROOT,now=None):
    clock=utc(now or datetime.now(timezone.utc))
    matches,_,source_rejects=archive.match_archive(source_root,clock)
    finals,conflicts,result_rejects=_result_index(source_root,prematch_root)
    training=training_records(matches,finals,clock)
    lookup={inventory._identity(a):(a,b) for a,b in matches}
    candidates=[];counts=Counter();rejects=Counter()
    for path in sorted((Path(pair_root)/"sets").glob("*.json")):
        try:
            raw=json.loads(path.read_text(encoding="utf-8"))
            models,choices,coverage=_validated(raw,source_root=source_root,now=clock)
            if path.stem!=raw["sydney_date"]:
                raise ValueError("filename_mismatch")
            candidates.append((utc(raw["predicted_at_utc"]),path.stem,models,choices,coverage))
        except (ValueError,KeyError,TypeError,OSError,AttributeError,ArithmeticError) as exc:
            rejects[str(exc) if isinstance(exc,ValueError) else type(exc).__name__]+=1
    rows=[];seen=set();events=set()
    for decision,day,models,choices,coverage in sorted(candidates,key=lambda r:r[:2]):
        try:
            ids={inventory._identity(m) for m in models}
            physical={inventory._event(m) for m in models}
            if len(ids)!=2 or len(physical)!=2 or seen&ids or events&physical:
                raise ValueError("repeated_original_fixture")
            seen.update(ids);events.update(physical)
            legs=[replay.match_leg(m,lookup.get(inventory._identity(m)),decision) for m in models]
            counts["pairs_with_"+str(sum(x is not None for x in legs))+"_matched_legs"]+=1
            if any(x is None for x in legs):
                continue
            half,full,sizes=corrections(training,models,legs,decision)
            counts["chronologically_qualified_pairs"]+=1
            actual=[_final(m,finals.get(inventory._identity(m)),clock) for m in models]
            if any(x is None for x in actual):
                counts["qualified_pending_final"]+=1
                continue
            if any(_final(x,finals.get(inventory._identity(x)),clock)!=score
                   for leg,score in zip(legs,actual) for x in leg):
                raise ValueError("matched_final_disagrees")
            arms={"original_v8":{"full_score_grids":[_grid(m["score_matrix"]) for m in models],
                                 "five_doubles":choices,"estimated_coverage":coverage},
                  "time_matched_v8":_arm([_grid(a["score_matrix"]) for a,b in legs]),
                  "stored_v9":_arm([_grid(b["score_matrix"]) for a,b in legs]),
                  "joint_half":_arm(half),"joint_full":_arm(full)}
            rows.append({"day":day,"training":sizes,
                         "metrics":{name:sources._metric(arm,actual) for name,arm in arms.items()}})
        except (ValueError,KeyError,TypeError,OSError,AttributeError,ArithmeticError) as exc:
            rejects[str(exc) if isinstance(exc,ValueError) else type(exc).__name__]+=1
    return {"study_type":"original_pair_joint_rate_transfer_replay_v1",
            "verified_training_fixtures":len(training),"counts":dict(counts),
            "matched_pair_comparisons":replay.summary(rows,ARMS),
            "pair_rejections":dict(rejects),"source_rejections":dict(source_rejects),
            "result_rejections":dict(result_rejects),"result_conflicts":len(conflicts),
            "external_requests":0,"new_pairs_created":0,"automatic_promotion":False,
            "rows":rows,"limitations":["Exploratory chronological replay; no new prospective forecasts.",
                "Single-match log loss improvements need not transfer to two-match exact-score coverage.",
                "Sparse qualifying pairs cannot establish accuracy or superiority.",
                "Same-time market sources must predate the original pair decision."]}

if __name__=="__main__":
    print(json.dumps(evaluate(),indent=2,allow_nan=False))
