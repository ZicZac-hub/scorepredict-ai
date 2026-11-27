"""Compare V9 probability models on original frozen pairs, never new pairs.

Archived same-time controls isolate the market adjustment; original V8
remains an additional benchmark. Use only sources available by the original
pair decision. Adaptive weight fitting uses earlier independently verified
finals, excluding BOTH target fixtures. No output of future selections.
"""
from collections import Counter
from datetime import datetime,timedelta,timezone
import json,math
from pathlib import Path

from v10_prospective import utc
from five_double_accuracy import _final,_result_index,wilson
from professional_double_shadow import _validated
from v44_family_pool_research import _grid
from v44_pair_transfer_research import _arm
from v44_single_calibration import _prob
import v44_market_ablation as archive
import v44_market_pool_calibration as calibration
import v44_v9_reliability as reliable
import v44_fixture_evidence as inventory
import v42_research_trial as sources

FIXED=("original_v8","time_matched_v8","stored_v9","checked_v9")
ADAPTIVE=(*FIXED,"adaptive_v9")


def match_leg(model,matched,cutoff):
    """Require the matched fixture and original timing, not just team names."""
    cutoff=utc(cutoff)
    if matched is None:
        return None
    a,b=matched
    if (archive.match_key(a)!=archive.match_key(b)
            or any(inventory._identity(m)!=inventory._identity(model)
                   or inventory._event(m)!=inventory._event(model)
                   or str(m["home_team_id"])!=str(model["home_team_id"])
                   or str(m["away_team_id"])!=str(model["away_team_id"])
                   or utc(m["kickoff_utc"])!=utc(model["kickoff_utc"])
                   or utc(m["captured_at_utc"])>cutoff for m in (a,b))):
        return None
    market=archive.qualify_market(b)
    if utc(market["snapshot_utc"])>cutoff:
        return None
    return a,b


def earlier_training(records,models,cutoff):
    ids={inventory._identity(m) for m in models}
    events={inventory._event(m) for m in models}
    return [r for r in records if r["known"]<cutoff and r["identity"] not in ids
            and r["event"] not in events]


def summary(rows,names):
    out={}
    for name in names:
        m=[r["metrics"][name] for r in rows]
        n=len(m); hits=sum(v["hit"] for v in m)
        out[name]={"pairs":n,"hits":hits,"hit_rate":hits/n if n else None,
                   "wilson_95":wilson(hits,n),
                   **{k:math.fsum(v[k] for v in m)/n if n else None
                      for k in ("score_log_loss","outcome_brier","estimated_coverage","set_brier_descriptive")}}
    return out


def evaluate(*,source_root=sources.SOURCE_ROOT,prematch_root=sources.PREMATCH_ROOT,
             pair_root=sources.PAIR_ROOT,now=None):
    clock=utc(now or datetime.now(timezone.utc))
    matches,_,source_rejects=archive.match_archive(source_root,clock)
    finals,conflicts,result_rejects=_result_index(source_root,prematch_root)
    lookup={inventory._identity(a):(a,b) for a,b in matches}
    training=[]
    for a,b in matches:
        identity=inventory._identity(a)
        final=finals.get(identity)
        actual=_final(a,final,clock)
        if actual is None or _final(b,final,clock)!=actual:
            continue
        training.append({"identity":identity,"event":inventory._event(a),
                         "known":utc(final["verified_at_utc"]),
                         "p0":_prob(_grid(a["score_matrix"]),actual),
                         "p1":_prob(_grid(b["score_matrix"]),actual)})
    counts=Counter();rejects=Counter();rows=[];adaptive=[];fresh=[]
    seen=set();events=set()
    for path in sorted((Path(pair_root)/"sets").glob("*.json")):
        try:
            raw=json.loads(path.read_text(encoding="utf-8"))
            models,choices,coverage=_validated(raw,source_root=source_root,now=clock)
            if path.stem!=raw["sydney_date"]:
                raise ValueError("filename_mismatch")
            ids={inventory._identity(m) for m in models}
            physical={inventory._event(m) for m in models}
            if len(ids)!=2 or len(physical)!=2 or seen&ids or events&physical:
                raise ValueError("repeated_original_fixture")
            seen.update(ids);events.update(physical)
            cutoff=utc(raw["predicted_at_utc"])
            legs=[match_leg(m,lookup.get(inventory._identity(m)),cutoff) for m in models]
            counts["pairs_with_"+str(sum(v is not None for v in legs))+"_matched_legs"]+=1
            if any(v is None for v in legs):
                continue
            # Freeze inclusion based on pre-match evidence, before reading results.
            age=[(cutoff-utc(b["market_snapshot"]["snapshot_utc"])).total_seconds()/3600
                 for a,b in legs]
            is_fresh=all(0<=v<=12 for v in age)
            counts["both_market_snapshots_within_12h"]+=int(is_fresh)
            actual=[_final(m,finals.get(inventory._identity(m)),clock) for m in models]
            if any(v is None for v in actual):
                counts["matched_pairs_pending_final"]+=1
                continue
            if any(_final(a,finals.get(inventory._identity(a)),clock)!=score
                   or _final(b,finals.get(inventory._identity(b)),clock)!=score
                   for (a,b),score in zip(legs,actual)):
                raise ValueError("matched_final_disagrees")
            base=[_grid(a["score_matrix"]) for a,b in legs]
            stored=[_grid(b["score_matrix"]) for a,b in legs]
            checked=[reliable.checked_projection(g,archive.qualify_market(b))[0]
                     for g,(a,b) in zip(base,legs)]
            original={"full_score_grids":[_grid(m["score_matrix"]) for m in models],
                      "five_doubles":choices,"estimated_coverage":coverage}
            arms={"original_v8":original,"time_matched_v8":_arm(base),
                  "stored_v9":_arm(stored),"checked_v9":_arm(checked)}
            measured={k:sources._metric(a,actual) for k,a in arms.items()}
            row={"day":path.stem,"market_age_hours":age,"metrics":measured}
            rows.append(row)
            if is_fresh:
                fresh.append(row)
            prior=earlier_training(training,models,cutoff)
            # Fit operates on a prefiltered cohort; no target event can re-enter.
            w,n=calibration.fit(prior,cutoff,inventory._identity(models[0]),inventory._event(models[0]))
            if w is None:
                counts["matched_pairs_insufficient_prior_training"]+=1
                continue
            pooled=_arm([calibration.pool(a,b,w) for a,b in zip(base,stored)])
            adaptive.append({**row,"weight":w,"training_fixtures":n,
                             "metrics":{**measured,"adaptive_v9":sources._metric(pooled,actual)}})
        except (ValueError,KeyError,TypeError,OSError,AttributeError,ArithmeticError) as exc:
            rejects[str(exc) if isinstance(exc,ValueError) else type(exc).__name__]+=1
    return {"study_type":"original_pair_v9_matched_replay",
            "verified_single_training_fixtures":len(training),
            "counts":dict(counts),"all_matched_pair_comparisons":summary(rows,FIXED),
            "fresh_snapshot_subset_comparisons":summary(fresh,FIXED),
            "adaptive_matched_subset_comparisons":summary(adaptive,ADAPTIVE),
            "pair_rejections":dict(rejects),"source_rejections":dict(source_rejects),
            "result_conflicts":len(conflicts),"result_rejections":dict(result_rejects),
            "external_requests":0,"new_pairs_created":0,"automatic_promotion":False,
            "rows":{"fixed":rows,"adaptive":adaptive},
            "limitations":["Retrospective replay of original pairs, not live or prospective V9 accuracy.",
                           "Archived snapshots may predate the original decision by many hours; fresh subset is reported separately.",
                           "Time-matched V8 and V9 use the same underlying inputs; original V8 may use a different capture.",
                           "Joint score probabilities assume independent matches; small matched samples cannot establish superiority."]}

if __name__=="__main__":
    print(json.dumps(evaluate(),indent=2,allow_nan=False))
