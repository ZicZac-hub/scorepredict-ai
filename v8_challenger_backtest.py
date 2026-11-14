"""Historical pre-kickoff replay of preserved v6 versus experimental v8.

Never changes live forecasts or the immutable prospective record. Replay
uses the same provider-qualified 730-day results-as-of cutoff for both models.
Weights were fixed before this script was written: do NOT tune on holdout.
"""
import argparse
import json
from collections import defaultdict
from copy import deepcopy
from datetime import timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import engine
from history_pipeline import build_evidence, POLICY_VERSION
from reference_model_benchmark import metric, summarize
from research_ensemble_v8 import predict as v8_predict, MODEL_VERSION
from smart_double_ai import five_doubles
from smart_double_v8 import rank_research_pairs

SYDNEY=ZoneInfo("Australia/Sydney")


def _clean_target(fixture):
    target=deepcopy(fixture)
    target["status"]="SCHEDULED"
    target["score"]={"fullTime":{"home":None,"away":None}}
    target.pop("provider_status",None)
    return target


def _doubles_hit(picks, first, second):
    a=first["score"]["fullTime"]
    b=second["score"]["fullTime"]
    one=f"{a['home']}-{a['away']}"
    two=f"{b['home']}-{b['away']}"
    return any(x["Score 1"]==one and x["Score 2"]==two for x in picks)


def evaluate(matches, holdout_year=2026, max_fixtures=180,
             max_pair_days=12, lead_minutes=60):
    if max_fixtures<1 or max_pair_days<0 or lead_minutes<1:
        raise ValueError("Positive pre-match sample and lead required")
    ordered=sorted((m for m in matches if engine.finished(m) and engine._date(m.get("utcDate"))),
                   key=lambda row:engine._date(row["utcDate"]))
    target=[m for m in ordered if engine._date(m["utcDate"]).year==holdout_year][-max_fixtures:]
    baseline_stats=[]
    v8_stats=[]
    paired=[]
    fallbacks=0
    source_counts=defaultdict(int)
    for game in target:
        kickoff=engine._date(game["utcDate"])
        asof=kickoff-timedelta(minutes=lead_minutes)
        packet=build_evidence(game,ordered,asof)
        fixture=_clean_target(packet["fixture"])
        past=packet["history"]
        old=engine.predict(fixture,past,league_history=packet["league_history"])
        new=v8_predict(fixture,past,league_history=packet["league_history"],as_of=asof)
        if not old or not new:
            continue
        old_row=metric(old["matrix"],game)
        new_row=metric(new["matrix"],game)
        baseline_stats.append(old_row)
        v8_stats.append(new_row)
        fallbacks+=int(len(new["component_weights"])==1)
        source_counts[str((game.get("competition") or {}).get("code") or "?")]+=1
        paired.append({"fixture":game,"old":old,"new":new,
                       "old_exact":old_row["exact"],"new_exact":new_row["exact"]})
    results={"protocol":POLICY_VERSION,"model":MODEL_VERSION,
             "input_matches":len(ordered),"year":holdout_year,
             "latest_chronological_fixture_sample":len(target),
             "paired_matches":len(paired),"v8_baseline_only_fallbacks":fallbacks,
             "v6":summarize(baseline_stats) if baseline_stats else None,
             "v8":summarize(v8_stats) if v8_stats else None,
             "challenger_only_exact_hits":sum(p["new_exact"] and not p["old_exact"] for p in paired),
             "baseline_only_exact_hits":sum(p["old_exact"] and not p["new_exact"] for p in paired),
             "leagues":dict(source_counts),
             "forecast_lead_minutes":lead_minutes,
             "policy":"Pre-match replay; evaluated fixture result excluded; results must be available before kickoff minus the lead. Fixed v8 weights; independent holdout, no tuning or production promotion.",
             "limitations":"Historical reconstruction, not a prospective live hit rate. Unknown injury/lineup/xG numbers were not included."}
    # Recreate a common snapshot BEFORE the earliest match of each day,
    # rather than accidentally using another match's in-play result.
    days=defaultdict(list)
    for m in target:
        date=engine._date(m["utcDate"]).astimezone(SYDNEY).date()
        days[date].append(m)
    pairs=[]
    eligible_days=[day for day,games in days.items() if len(games)>=2][-max_pair_days:] if max_pair_days else []
    for day in eligible_days:
        games=sorted(days[day],key=lambda row:engine._date(row["utcDate"]))[:8]
        common_asof=engine._date(games[0]["utcDate"])-timedelta(minutes=lead_minutes)
        candidates=[]
        baselines=[]
        actual_by_id={}
        for game in games:
            packet=build_evidence(game,ordered,common_asof)
            fixture=_clean_target(packet["fixture"])
            old=engine.predict(fixture,packet["history"],league_history=packet["league_history"])
            new=v8_predict(fixture,packet["history"],league_history=packet["league_history"],as_of=common_asof)
            if not old or not new or len(new["component_weights"])<2:
                continue
            row={"fixture":fixture,"kickoff":game["utcDate"],
                 "result":new,"league":str((game.get("competition") or {}).get("code") or "?")}
            candidates.append(row)
            baselines.append({**row,"result":old})
            actual_by_id[(str(fixture.get("provider")),str(fixture.get("id")))]=game
        ranked=rank_research_pairs(candidates,limit=1)
        if not ranked:
            continue
        best=ranked[0]
        first=best["first"];second=best["second"]
        def original_for(row):
            return next((b for b in baselines
                         if (str(b["fixture"].get("provider")),str(b["fixture"].get("id"))) ==
                         (str(row["fixture"].get("provider")),str(row["fixture"].get("id")))),None)
        old_first,old_second=original_for(first),original_for(second)
        if old_first is None or old_second is None:
            continue
        x=actual_by_id[(str(first["fixture"].get("provider")),str(first["fixture"].get("id")))]
        y=actual_by_id[(str(second["fixture"].get("provider")),str(second["fixture"].get("id")))]
        original=five_doubles(old_first,old_second)
        if len(original)!=5:
            continue
        pairs.append({"day":day.isoformat(),"v8_hit":_doubles_hit(best["doubles"],x,y),
                      "same_pair_v6_hit":_doubles_hit(original,x,y),
                      "v8_estimated_coverage":best["coverage"],
                      "v6_estimated_coverage":sum(o["Joint"] for o in original)})
    n=len(pairs)
    results["five_double_replay"]={
        "same_day_eligible_snapshots":n,
        "v8_five_double_hits":sum(p["v8_hit"] for p in pairs),
        "preserved_v6_five_double_hits_for_same_pairs":sum(p["same_pair_v6_hit"] for p in pairs),
        "v8_rate":sum(p["v8_hit"] for p in pairs)/n if n else None,
        "preserved_v6_rate":sum(p["same_pair_v6_hit"] for p in pairs)/n if n else None,
        "notes":"Both algorithms evaluated on exactly the two matches selected by v8, using a COMMON PRE-MATCH snapshot before the first kickoff. Sample is limited; no auto-promotion.",
        "days":pairs}
    return results


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("matches_json",type=Path)
    parser.add_argument("--output",type=Path,default=Path("work/v8_challenger_evaluation.json"))
    parser.add_argument("--holdout-year",type=int,default=2026)
    parser.add_argument("--max-fixtures",type=int,default=180)
    parser.add_argument("--max-pair-days",type=int,default=12)
    args=parser.parse_args()
    raw=json.loads(args.matches_json.read_text(encoding="utf-8"))
    report=evaluate(raw,holdout_year=args.holdout_year,max_fixtures=args.max_fixtures,
                    max_pair_days=args.max_pair_days)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({k:v for k,v in report.items() if k!="five_double_replay"} |
                     {"five_double_replay":{k:v for k,v in report["five_double_replay"].items() if k!="days"}},indent=2))

if __name__=="__main__":
    main()
