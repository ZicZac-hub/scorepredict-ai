"""V14 pre-match walk-forward test of ROBUST FIVE-DOUBLE PAIR SELECTION.

Compare on the SAME Sydney calendar days and exactly the same accessible
fixtures:
  1. Original v8 highest-estimated-coverage match pair and five guesses.
  2. V7 fully diversified five guesses on that same v8-selected pair.
  3. V14 five conservative envelope guesses on that same pair.
  4. V14 *independently selected* strongest agreement pair.

Use full same-provider/league history available one hour before the EARLIEST
kickoff each day. No future results, market odds or AI votes. A changed match
pair can change the actual result: success on one pair is not success on the
other. Count every baseline-eligible Sydney date in the denominator,
including dates where V14 abstains. Never promote based on this old data.
"""
import argparse
import json
from collections import defaultdict
from copy import deepcopy
from datetime import timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import engine
from history_pipeline import build_evidence
from hybrid_five_v13 import predict as v13_predict
from robust_double_v14 import rank_robust_pairs, robust_pair, VERSION
from smart_double_v8 import rank_research_pairs
from smart_double_ai import five_doubles
from double_score_selection import select_diversified

SYDNEY=ZoneInfo("Australia/Sydney")


def _hide_result(fixture):
    copy=deepcopy(fixture)
    copy["status"]="SCHEDULED"
    copy["score"]={"fullTime":{"home":None,"away":None}}
    copy.pop("provider_status",None)
    return copy


def _predictions(fixture,history,asof):
    packet=build_evidence(fixture,history,asof)
    masked=_hide_result(packet["fixture"])
    model=v13_predict(masked,packet["history"],
                      league_history=packet["league_history"],as_of=asof)
    if model is None:
        return None
    models=model.get("component_score_distributions") or {}
    if not all(models.get(m) and model.get("component_score_matrices",{}).get(m)
               is not None for m in ("v7","v8")):
        return None
    return {"fixture":masked,"result":model,"kickoff":fixture["utcDate"],
            "league":str((fixture.get("competition") or {}).get("code") or "?"),
            "digest":packet["input_digest"]}


def _key(row):
    f=row["fixture"]
    return (str(f.get("provider") or "football-data.org"),str(f.get("id")))


def _actual(game):
    g=game["score"]["fullTime"]
    return f"{g['home']}-{g['away']}"


def _hit(picks,first,second,actuals):
    return int(any(item["Score 1"]==actuals[_key(first)] and
                   item["Score 2"]==actuals[_key(second)]
                   for item in picks))


def _v8_row(hybrid):
    return {**hybrid,"result":{
        "scores":hybrid["result"]["component_score_distributions"]["v8"],
        "model_version":hybrid["result"]["component_versions"]["v8"]}}


def _v7_row(hybrid):
    return {**hybrid,"result":{
        "scores":hybrid["result"]["component_score_distributions"]["v7"],
        "model_version":hybrid["result"]["component_versions"]["v7"]}}


def _result(records,field):
    total=len(records)
    issued=sum(r.get(field) is not None for r in records)
    hits=sum(int(r.get(field) or 0) for r in records)
    return {
        "baseline_eligible_days":total,
        "selections_issued":issued,
        "abstained_days":total-issued,
        "verified_five_double_hits":hits,
        "per_eligible_day_hit_rate":hits/total if total else None,
        "per_selection_hit_rate":hits/issued if issued else None,
    }


def evaluate(matches,*,year=2026,max_fixtures=700,max_pair_days=100,
             min_robust_envelope=0.04,max_per_day=6):
    """Read-only strictly chronological same-calendar-day retrospective test."""
    if max_fixtures<2 or max_pair_days<1 or max_per_day<2:
        raise ValueError("A nonempty fixed historical test population is required")
    ordered=sorted((x for x in matches if engine.finished(x)
                    and engine._date(x.get("utcDate")) is not None),
                   key=lambda x:(engine._date(x["utcDate"]),
                                 str(x.get("provider")),str(x.get("id"))))
    targets=[x for x in ordered if engine._date(x["utcDate"]).year==year][-max_fixtures:]
    days=defaultdict(list)
    for match in targets:
        day=engine._date(match["utcDate"]).astimezone(SYDNEY).date().isoformat()
        days[day].append(match)
    # Chronological latest dates from a predeclared horizon, never choose
    # the best-looking historical dates after observing hit/miss outcomes.
    grouped=[(d,sorted(items,key=lambda x:engine._date(x["utcDate"]))[:max_per_day])
             for d,items in sorted(days.items()) if len(items)>=2][-max_pair_days:]
    rows=[]
    diagnostics=defaultdict(int)
    for index,(day,games) in enumerate(grouped,1):
        as_of=engine._date(games[0]["utcDate"])-timedelta(hours=1)
        candidates=[]
        actuals={}
        for game in games:
            result=_predictions(game,ordered,as_of)
            if result is None:
                diagnostics["no_full_v7_v8_evidence"]+=1
                continue
            candidates.append(result)
            actuals[_key(result)]=_actual(game)
        if len(candidates)<2:
            diagnostics["no_two_qualified_forecasts"]+=1
            continue
        baseline=rank_research_pairs([_v8_row(x) for x in candidates],limit=1)
        if not baseline:
            diagnostics["no_valid_v8_pair"]+=1
            continue
        choice=baseline[0]
        source={_key(x):x for x in candidates}
        original_a=source[_key(choice["first"])]
        original_b=source[_key(choice["second"])]
        baseline_hit=_hit(choice["doubles"],choice["first"],choice["second"],actuals)
        # Same two fixtures, only the five score combinations change.
        v7left,v7right=_v7_row(original_a),_v7_row(original_b)
        possible=[{"Score 1":f"{h}-{a}","Score 2":f"{j}-{k}",
                   "Joint":float(p)*float(q)}
                  for h,a,p in v7left["result"]["scores"][:10]
                  for j,k,q in v7right["result"]["scores"][:10]]
        v7_picks=select_diversified(possible,5)
        robust_same=robust_pair(original_a,original_b,now=as_of)
        if robust_same is None:
            diagnostics["no_same_pair_robust"]+=1
        best_robust=rank_robust_pairs(
            candidates,now=as_of,limit=1,min_coverage=min_robust_envelope)
        selected=best_robust[0] if best_robust else None
        record={
            "sydney_day":day,
            "cutoff_utc":as_of.isoformat(),
            "eligible_fixtures":len(candidates),
            "v8_pair":[_key(choice["first"]),_key(choice["second"])],
            "v8_model_coverage":float(choice["coverage"]),
            "v8_top5_hit":baseline_hit,
            "v7_diversified_same_pair_hit":_hit(v7_picks,choice["first"],
                                               choice["second"],actuals),
            "v14_robust_same_pair_hit":(
                _hit(robust_same["doubles"],original_a,original_b,actuals)
                if robust_same is not None else None),
            "v14_auto_pair_hit":(
                _hit(selected["doubles"],selected["first"],selected["second"],actuals)
                if selected is not None else None),
            "v14_auto_envelope_score":(
                float(selected["estimated_joint_coverage"])
                if selected is not None else None),
            "v14_auto_pair":[_key(selected["first"]),_key(selected["second"])]
                if selected is not None else None,
            "v14_same_as_v8_pair":(
                {_key(selected["first"]),_key(selected["second"])}==
                {_key(original_a),_key(original_b)}
                if selected is not None else None),
        }
        rows.append(record)
        if index%15==0:
            print(f"Backtested {index}/{len(grouped)} Sydney days; "
                  f"{len(rows)} eligible pairs",flush=True)
    result={
        "title":"V14 robust two-match five-double chronological historical challenge",
        "policy":"All games and original results masked until their final-score evaluation",
        "as_of":"One hour before first kickoff of the Sydney day",
        "source_result_count":len(ordered),
        "year":year,
        "latest_year_fixtures":len(targets),
        "candidate_days":len(grouped),
        "max_per_sydney_day":max_per_day,
        "baseline_eligible_pair_days":len(rows),
        "min_v14_envelope_for_auto_selection":min_robust_envelope,
        "v8_original_top5":_result(rows,"v8_top5_hit"),
        "v7_diversified_on_v8_pair":_result(rows,"v7_diversified_same_pair_hit"),
        "v14_robust_on_same_v8_pair":_result(rows,"v14_robust_same_pair_hit"),
        "v14_independently_selected_pair":_result(rows,"v14_auto_pair_hit"),
        "auto_v14_selected_the_same_two_matches_as_v8":sum(
            r["v14_same_as_v8_pair"] is True for r in rows),
        "diagnostics":dict(diagnostics),
        "status":"EXPERIMENTAL historical reconstruction, 2026 already explored",
        "not_independent_holdout":True,
        "not_a_prospective_verified_hit_rate":True,
        "automatic_production_changed":False,
        "promotion_approved":False,
        "limitations":("Model probabilities are uncalibrated and V7/V8 share "
                      "input evidence. Full-grid robust score is NOT a "
                      "statistically verified confidence interval. "
                      "Historical snapshot cannot reconstruct real-time "
                      "injuries, odds, or missing future fixtures."),
        "records":rows,
    }
    return result


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("matches_json",type=Path)
    ap.add_argument("--year",type=int,default=2026)
    ap.add_argument("--max-fixtures",type=int,default=700)
    ap.add_argument("--max-pair-days",type=int,default=100)
    ap.add_argument("--min-robust-envelope",type=float,default=0.04)
    ap.add_argument("--output",type=Path,default=Path("work/v14_robust_backtest.json"))
    args=ap.parse_args()
    raw=json.loads(args.matches_json.read_text(encoding="utf-8"))
    matches=raw if isinstance(raw,list) else raw.get("matches")
    if not isinstance(matches,list):
        ap.error("Expected normalized verified matches list")
    result=evaluate(matches,year=args.year,max_fixtures=args.max_fixtures,
                    max_pair_days=args.max_pair_days,
                    min_robust_envelope=args.min_robust_envelope)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print("===== V14 ROBUST FIVE DOUBLE HISTORICAL COMPARISON =====",flush=True)
    print(json.dumps({k:v for k,v in result.items() if k!="records"},indent=2),
          flush=True)


if __name__=="__main__":
    main()
