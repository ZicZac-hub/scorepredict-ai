"""Independent same-evidence 2026 model tournament: read-only historical replay.

Every candidate uses one identical, provider/competition-verified pre-match
history packet. Results as-of kickoff-minus-one-hour exclude same-fixture and
unavailable finished scores. No fitting to the 2026 holdout and no writing to
production forecasts. Missing historical market odds and AI votes mean those
methods are NOT TESTABLE (never substituted with fake market inputs).
"""
import argparse
import json
import math
from collections import defaultdict
from copy import deepcopy
from datetime import timedelta
from itertools import combinations
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import engine
from history_pipeline import build_evidence, POLICY_VERSION
from forecast_model import predict as versioned_predict
from research_ensemble_v8 import predict as v8_predict
from score_calibration_v12 import predict as v12_predict
from hybrid_five_v13 import predict as v13_predict
from smart_double_ai import five_doubles
from double_score_selection import select_balanced, select_diversified

SYDNEY=ZoneInfo("Australia/Sydney")
VARIANTS={
    "v6_baseline":"baseline",
    "v6_regularized":"regularized",
    "v6_venue_arithmetic":"venue_arithmetic",
    "v6_team_strength":"team_strength",
    "v6_opponent_adjusted":"opponent_adjusted",
    "v6_venue_contrast":"venue_contrast",
    "v6_league_opponent":"league_opponent",
    "v6_strength_contrast":"strength_contrast_candidate",
    "v6_dixon_coles":"dixon_coles",
    "v6_temperature":"temperature_candidate",
}
CORE=("v6_baseline","v7_fitted","v8_ensemble","v12_calibrated","v13_hybrid")
NOT_COMPARABLE={
    "v9_market_odds":"No timestamped, authentic pre-match bookmaker market quotes for the 2026 holdout.",
    "v10_multi_market":"No three timestamped pre-match quotes from independent bookmakers for the 2026 holdout.",
    "external_ai_5_10":"No complete, frozen before-kickoff votes from the configured external AI services on the same fixtures.",
    "sportmonks_external":"No independently matched and timestamped external correct-score probability records on the same fixtures.",
}


def masked_fixture(game):
    result=deepcopy(game)
    result["status"]="SCHEDULED"
    result["score"]={"fullTime":{"home":None,"away":None}}
    result.pop("provider_status",None)
    return result


def score_metrics(model,actual):
    matrix=np.asarray(model["matrix"],dtype=float)
    assert matrix.ndim==2 and np.isfinite(matrix).all() and (matrix>=0).all()
    assert abs(float(matrix.sum())-1)<1e-5
    home,away=actual
    top=tuple(map(int,np.unravel_index(int(matrix.argmax()),matrix.shape)))
    flat=np.argsort(-matrix.ravel(),kind="stable")[:3]
    top3={tuple(map(int,np.unravel_index(int(i),matrix.shape))) for i in flat}
    p=(float(matrix[home,away])
       if home<matrix.shape[0] and away<matrix.shape[1] else 0.0)
    result=0 if home>away else 1 if home==away else 2
    outcomes=(float(np.tril(matrix,-1).sum()),
              float(np.trace(matrix)),float(np.triu(matrix,1).sum()))
    return {"exact":int(top==actual),"top3":int(actual in top3),
            "log_loss":-math.log(max(p,1e-12)),
            "brier":sum((val-int(i==result))**2 for i,val in enumerate(outcomes)),
            "top_score":str(top[0])+"-"+str(top[1]),
            "top_probability":float(matrix[top])}


def summarize(rows):
    if not rows:
        return {"n":0,"exact_hits":0,"exact_pct":None,"top3_pct":None,
                "log_loss":None,"brier":None,"modal_1_1_pct":None}
    count=len(rows)
    avg=lambda key: sum(r[key] for r in rows)/count
    return {
        "n":count,"exact_hits":sum(r["exact"] for r in rows),
        "exact_pct":100*avg("exact"),"top3_pct":100*avg("top3"),
        "log_loss":avg("log_loss"),"brier":avg("brier"),
        "modal_1_1_pct":100*sum(r["top_score"]=="1-1" for r in rows)/count,
    }


def _predict_models(fixture, history, league, as_of, mode_keys):
    found={};errors={}
    base=None;v8=None
    for label,variant in VARIANTS.items():
        if label not in mode_keys:
            continue
        try:
            value=engine.predict(fixture,history,variant=variant,league_history=league)
            if value is not None:
                found[label]=value
                if label=="v6_baseline":base=value
        except (ValueError,TypeError,KeyError,ZeroDivisionError,OverflowError,
                ArithmeticError,ImportError) as exc:
            errors[label]=type(exc).__name__
    if "v7_fitted" in mode_keys:
        try:
            value=versioned_predict(fixture,history,variant="reference_v7",league_history=league)
            if value is not None:
                found["v7_fitted"]=value
        except (ValueError,TypeError,KeyError,ZeroDivisionError,OverflowError,
                ArithmeticError,ImportError) as exc:
            errors["v7_fitted"]=type(exc).__name__
    if "v8_ensemble" in mode_keys or "v12_calibrated" in mode_keys:
        try:
            v8=v8_predict(fixture,history,league_history=league,as_of=as_of)
            if v8 is not None and "v8_ensemble" in mode_keys:
                found["v8_ensemble"]=v8
        except (ValueError,TypeError,KeyError,ZeroDivisionError,OverflowError,
                ArithmeticError,ImportError) as exc:
            errors["v8_ensemble"]=type(exc).__name__
    if "v12_calibrated" in mode_keys and v8 is not None:
        try:
            candidate=v12_predict(fixture,history,league_history=league,
                                  as_of=as_of,reference_forecast=v8)
            if candidate is not None:
                found["v12_calibrated"]=candidate
        except (ValueError,TypeError,KeyError,ZeroDivisionError,OverflowError,
                ArithmeticError,ImportError) as exc:
            errors["v12_calibrated"]=type(exc).__name__
    if "v13_hybrid" in mode_keys and v8 is not None:
        try:
            candidate=v13_predict(
                fixture,history,league_history=league,as_of=as_of,
                reference_v8=v8,reference_v6=found.get("v6_baseline"),
                reference_v7=found.get("v7_fitted"),
                reference_v12=found.get("v12_calibrated"))
            if candidate is not None:
                found["v13_hybrid"]=candidate
        except (ValueError,TypeError,KeyError,ZeroDivisionError,OverflowError,
                ArithmeticError,ImportError) as exc:
            errors["v13_hybrid"]=type(exc).__name__
    return found,errors


def _score(actual):
    return str(actual["home"])+"-"+str(actual["away"])


def _hits(options,a,b):
    return int(any(row["Score 1"]==a and row["Score 2"]==b for row in options))


def evaluate(matches,*,year=2026,max_fixtures=220,max_pair_days=24):
    ordered=sorted((m for m in matches if engine.finished(m) and
                    engine._date(m.get("utcDate")) is not None),
                   key=lambda m:(engine._date(m["utcDate"]),str(m.get("id"))))
    target=[m for m in ordered if engine._date(m["utcDate"]).year==year][-max_fixtures:]
    stats=defaultdict(list);by_fixture={};errors=defaultdict(int);eligible=[]
    for index,game in enumerate(target,1):
        kickoff=engine._date(game["utcDate"])
        as_of=kickoff-timedelta(hours=1)
        evidence=build_evidence(game,ordered,as_of)
        past=evidence["history"]
        fixture=masked_fixture(evidence["fixture"])
        models,failures=_predict_models(
            fixture,past,past,as_of,set(VARIANTS)|set(CORE))
        for key,reason in failures.items():
            errors[key+":"+reason]+=1
        goals=game["score"]["fullTime"]
        actual=(goals["home"],goals["away"])
        metrics={}
        for key,model in models.items():
            try:
                current=score_metrics(model,actual)
                metrics[key]=current
                stats[key].append(current)
            except (ValueError,TypeError,AssertionError,IndexError) as exc:
                errors[key+":metrics_"+type(exc).__name__]+=1
        if metrics:
            eligible.append(game)
            by_fixture[(game.get("provider"),str(game["id"]))]=metrics
        if index%50==0:
            print("Evaluated",index,"/",len(target),"fixtures",flush=True)
    # Report an identical COMMON core cohort in addition to each model's
    # own qualified population. A model with fewer available forecasts
    # must not get an artificially inflated-looking unpaired accuracy rank.
    common=[d for d in by_fixture.values() if all(key in d for key in CORE)]
    common_report={name:summarize([r[name] for r in common]) for name in CORE}
    all_report={name:summarize(stats[name]) for name in list(VARIANTS)+list(CORE)[1:]}
    # Same-day double test: same selected fixture pair for all methods, one
    # snapshot taken BEFORE the first kickoff (not separate in-play data).
    days=defaultdict(list)
    for game in target:
        day=engine._date(game["utcDate"]).astimezone(SYDNEY).date().isoformat()
        days[day].append(game)
    double_records=[]
    pair_days=[(day,group) for day,group in sorted(days.items())
               if len(group)>=2][-max_pair_days:] if max_pair_days else []
    for day,group in pair_days:
        group=sorted(group,key=lambda item:engine._date(item["utcDate"]))[:5]
        cutoff=engine._date(group[0]["utcDate"])-timedelta(hours=1)
        rows=[];actuals={}
        for game in group:
            evidence=build_evidence(game,ordered,cutoff)
            fixture=masked_fixture(evidence["fixture"])
            models,_=_predict_models(fixture,evidence["history"],
                                     evidence["league_history"],cutoff,set(CORE))
            if any(name not in models for name in CORE):
                continue
            rows.append({"fixture":fixture,"kickoff":game["utcDate"],
                         "models":models})
            actuals[str(game["id"])]=_score(game["score"]["fullTime"])
        if len(rows)<2:
            continue
        # Select the highest V8 five-double coverage pair on that day.
        choices=[]
        for first,second in combinations(rows,2):
            left={"fixture":first["fixture"],"kickoff":first["kickoff"],
                  "result":first["models"]["v8_ensemble"]}
            right={"fixture":second["fixture"],"kickoff":second["kickoff"],
                   "result":second["models"]["v8_ensemble"]}
            pick=five_doubles(left,right)
            if len(pick)==5:
                choices.append((sum(x["Joint"] for x in pick),first,second))
        if not choices:
            continue
        _,a,b=max(choices,key=lambda x:x[0])
        outcomes={}
        for name in CORE:
            left={"fixture":a["fixture"],"result":a["models"][name]}
            right={"fixture":b["fixture"],"result":b["models"][name]}
            model_picks=five_doubles(left,right)
            if len(model_picks)<5:
                continue
            # Test original highest-five, fully diverse, and the experimental
            # balanced selector on IDENTICAL underlying scores and matches.
            options=[{"Score 1":f"{x}-{y}","Score 2":f"{h}-{k}","Joint":p*q}
                     for x,y,p in a["models"][name]["scores"][:10]
                     for h,k,q in b["models"][name]["scores"][:10]]
            balanced=select_balanced(options,5,diversity_weight=.18)
            diverse=select_diversified(options,5)
            actual_a=actuals[str(a["fixture"]["id"])]
            actual_b=actuals[str(b["fixture"]["id"])]
            outcomes[name]={
                "top5_hit":_hits(model_picks,actual_a,actual_b),
                "balanced_hit":_hits(balanced,actual_a,actual_b),
                "diversified_hit":_hits(diverse,actual_a,actual_b),
                "top5_coverage":sum(x["Joint"] for x in model_picks),
                "balanced_coverage":sum(x["Joint"] for x in balanced),
                "diversified_coverage":sum(x["Joint"] for x in diverse),
            }
        if len(outcomes)==len(CORE):
            double_records.append({"sydney_day":day,
                "pair_ids":[str(a["fixture"]["id"]),str(b["fixture"]["id"])],
                "actual":[actuals[str(a["fixture"]["id"])],actuals[str(b["fixture"]["id"])]],
                "models":outcomes})
    double_summary={}
    for name in CORE:
        matching=[r["models"][name] for r in double_records]
        n=len(matching)
        double_summary[name]={"pairs":n}
        for variant in ("top5","balanced","diversified"):
            double_summary[name][variant+"_hits"]=sum(r[variant+"_hit"] for r in matching)
            double_summary[name][variant+"_mean_coverage"]=(
                sum(r[variant+"_coverage"] for r in matching)/n if n else None)
    report={
        "type":"Independent read-only ZicZac 2026 historical tournament",
        "policy":POLICY_VERSION,
        "source_results":len(ordered),
        "target_year":year,
        "target_latest_fixtures":len(target),
        "asof":"Each fixture kickoff minus 60 minutes. Prior results available by finishedAt or kickoff plus three hours. Competition and provider matched.",
        "testing_risks":"Historical replay, not prospective; V12 and V13 choices were authored after observing earlier 2026 aggregate results. Treat 2026 as development evidence, NOT a clean untouched confirmation of V13 superiority.",
        "core_common_fixtures":len(common),
        "common_core_models":common_report,
        "by_model_own_eligible":all_report,
        "abstentions_and_errors":dict(errors),
        "not_testable_on_identical_historical_inputs":NOT_COMPARABLE,
        "double_protocol":"At most five matches per Sydney day; same earliest pre-kickoff snapshot. Pair selected by V8 on that day; every model tested on same two games.",
        "double_same_pair_days":len(double_records),
        "double_models":double_summary,
        "double_records":double_records,
        "v13_weights_frozen_before_this_run":True,
        "promote_any_model":False,
        "no_production_forecast_mutated":True,
    }
    return report


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("matches")
    ap.add_argument("--year",type=int,default=2026)
    ap.add_argument("--max-fixtures",type=int,default=220)
    ap.add_argument("--max-pair-days",type=int,default=24)
    ap.add_argument("--output",type=Path,default=Path("work/all_methods_tournament.json"))
    args=ap.parse_args()
    if args.max_fixtures<1 or args.max_pair_days<0:
        ap.error("Invalid test horizon")
    raw=json.loads(Path(args.matches).read_text(encoding="utf-8"))
    matches=raw if isinstance(raw,list) else raw.get("matches")
    if not isinstance(matches,list):
        ap.error("Expected a JSON list of normalized verified results")
    report=evaluate(matches,year=args.year,max_fixtures=args.max_fixtures,
                    max_pair_days=args.max_pair_days)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    compact={k:v for k,v in report.items() if k!="double_records"}
    print("===== ZICZAC INDEPENDENT MODEL TOURNAMENT =====",flush=True)
    print(json.dumps(compact,indent=2),flush=True)


if __name__=="__main__":
    main()
