"""V21 read-only chronological historical replay on archived real fixtures.

No network requests. All models see the exact same time-safe result history,
one hour BEFORE each kickoff. The test outcomes are intentionally hidden from
the forecast function. This is retrospective simulation, NOT a future-verified
success rate and never changes any published predictions.
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

from history_pipeline import build_evidence
from research_ensemble_v8 import predict as predict_v8
from goal_research_v20 import goal_form_challenger
from v10_prospective import utc
from v21_goal_model_comparison import variants,top_scores,inspect_prediction

SYDNEY=ZoneInfo("Australia/Sydney")
NAMES=("baseline_v8","negative_binomial_r3","bivariate_poisson_c012",
       "poisson_marginal","v20_goal_form")


def _metrics(observations):
    if not observations:
        return {"evaluated":0,"exact_hits":0,"exact_hit_rate":None,
                "top5_hits":0,"top5_rate":None,"score_log_loss":None,
                "outcome_brier":None,"over_2_5_brier":None,
                "goal_mae":None,"actual_high_score_rate":None,
                "predicted_high_score_mass":None}
    n=len(observations)
    mean=lambda k: sum(a[k] for a in observations)/n
    return {"evaluated":n,"exact_hits":sum(a["exact_hit"] for a in observations),
            "exact_hit_rate":mean("exact_hit"),
            "top5_hits":sum(a["top5_hit"] for a in observations),
            "top5_rate":mean("top5_hit"),
            "score_log_loss":mean("exact_log_loss"),
            "outcome_brier":mean("outcome_brier"),
            "over_2_5_brier":mean("over_2_5_brier"),
            "goal_mae":mean("goal_mae"),
            "actual_high_score_rate":mean("actual_high_score"),
            "predicted_high_score_mass":mean("predicted_high_score_mass"),
            "high_score_underprediction":mean("actual_high_score")-mean("predicted_high_score_mass"),
            "actual_over_2_5_rate":mean("observed_over_2_5"),
            "predicted_over_2_5_rate":mean("predicted_over_2_5")}


def _five(a,b,name):
    if name not in a["models"] or name not in b["models"]:
        return []
    left=top_scores(a["models"][name],5)
    right=top_scores(b["models"][name],5)
    return sorted([(h,j,k,l,float(p*q)) for h,j,p in left for k,l,q in right],
                  key=lambda x:(-x[4],x[:4]))[:5]


def _joint(row):
    return sum(x[4] for x in row)


def replay(matches,*,year=2026,max_eval=500,max_per_day=6,
           min_coverage=.055):
    rows=[]
    for match in matches:
        try:
            at=utc(match["utcDate"])
            s=(match.get("score") or {}).get("fullTime") or {}
            if (match.get("status")!="FINISHED" or
                match.get("provider")!="API-Football" or
                type(s.get("home")) is not int or type(s.get("away")) is not int or
                min(s["home"],s["away"])<0):
                continue
            rows.append((at,match))
        except (TypeError,ValueError,KeyError):
            continue
    rows.sort(key=lambda p:(p[0],str(p[1].get("id"))))
    targets=[(at,row) for at,row in rows if at.year==year][-max_eval:]
    observed={name:[] for name in NAMES}
    by_day=defaultdict(list)
    excluded=defaultdict(int)
    processed=0
    for at,match in targets:
        as_of=at-timedelta(hours=1)
        # All API/results have a 3h conservative settled-data delay.
        previously=[game for past,game in rows if past+timedelta(hours=3)<=as_of]
        masked=deepcopy(match)
        masked["status"]="SCHEDULED"
        masked["score"]={"fullTime":{"home":None,"away":None}}
        try:
            evidence=build_evidence(masked,previously,as_of)
            fixture=evidence["fixture"]
            forecast=predict_v8(fixture,evidence["history"],
                                league_history=evidence["league_history"],as_of=as_of)
            if not forecast or len(forecast.get("component_weights") or {})<2:
                excluded["original_research_history_insufficient"]+=1
                continue
            models=variants(forecast["matrix"])
            form=goal_form_challenger({"fixture":fixture,"result":forecast},
                                      evidence["history"],as_of)
            if form.get("available"):
                models["v20_goal_form"]=form["matrix"]
            else:
                excluded["v20_goal_form_insufficient"]+=1
            goals=match["score"]["fullTime"]
            for name,matrix in models.items():
                observed[name].append(inspect_prediction(matrix,goals["home"],goals["away"]))
            day=at.astimezone(SYDNEY).date().isoformat()
            by_day[day].append({"fixture_id":str(match["id"]),
                                "home_id":str(fixture["homeTeam"]["id"]),
                                "away_id":str(fixture["awayTeam"]["id"]),
                                "provider":str(fixture["provider"]),
                                "models":models,
                                "actual":(goals["home"],goals["away"]),
                                "as_of_utc":as_of.isoformat()})
            processed+=1
        except (ValueError,TypeError,KeyError,ArithmeticError):
            excluded["prediction_or_history_unavailable"]+=1
    # Only common fixtures, and the same chosen match pair for all modes.
    pair_observed={name:{"wins":0,"issued":0} for name in NAMES}
    valid_pair_days=0
    abstained=0
    for day,fixtures in sorted(by_day.items()):
        fixtures=fixtures[:max_per_day]
        choices=[]
        for a,b in combinations(fixtures,2):
            identities=[(r["provider"],r[k]) for r in (a,b)
                        for k in ("home_id","away_id")]
            if len(set(identities))!=4:
                continue
            base=_five(a,b,"baseline_v8")
            nb=_five(a,b,"negative_binomial_r3")
            biv=_five(a,b,"bivariate_poisson_c012")
            if min(len(base),len(nb),len(biv))!=5:
                continue
            lower=min(_joint(base),_joint(nb),_joint(biv))
            if lower>=min_coverage:
                choices.append((-lower,a["fixture_id"],b["fixture_id"],a,b))
        if not choices:
            abstained+=1
            continue
        _,_,_,a,b=min(choices,key=lambda v:v[:3])
        valid_pair_days+=1
        actual=(*a["actual"],*b["actual"])
        for name in NAMES:
            pick=_five(a,b,name)
            if len(pick)!=5:
                continue
            pair_observed[name]["issued"]+=1
            pair_observed[name]["wins"]+=int(any(x[:4]==actual for x in pick))
    for stats in pair_observed.values():
        stats["rate_per_issued"]=stats["wins"]/stats["issued"] if stats["issued"] else None
        stats["rate_per_eligible_day"]=stats["wins"]/valid_pair_days if valid_pair_days else None
    return {"kind":"retrospective_walk_forward_replay_not_genuine_future_accuracy",
            "year":year,"archived_verified_results":len(rows),
            "target_fixtures":len(targets),
            "matched_fixtures":processed,
            "one_hour_pre_kickoff_evidence":True,
            "three_hour_prior_result_delay":True,
            "same_fixture_original_vs_challenger":True,
            "single_match_metrics":{n:_metrics(observed[n]) for n in NAMES},
            "same_pair_exact_five":{"eligible_pair_days":valid_pair_days,
                                   "days_abstained":abstained,
                                   "methods":pair_observed,
                                   "selection_min_estimated_coverage":min_coverage},
            "excluded_counts":dict(excluded),
            "model_promoted":False,
            "note":"Retrospective and same dataset may have informed prior V8 model choices. Independent future verification is still required."}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--input",required=True)
    parser.add_argument("--output",required=True)
    parser.add_argument("--max-eval",type=int,default=500)
    args=parser.parse_args()
    history=json.loads(Path(args.input).read_text(encoding="utf-8"))
    if not isinstance(history,list):
        raise ValueError("Expected archived verified match list")
    report=replay(history,max_eval=max(10,min(args.max_eval,2500)))
    output=Path(args.output)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,sort_keys=True,indent=2,allow_nan=False)+"\n",
                      encoding="utf-8")
    print(json.dumps(report,indent=2))


if __name__=="__main__":
    main()
