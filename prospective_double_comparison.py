"""Immutable prospective comparison of five exact-score double strategies.

Pairs are frozen from the same capture batch, strictly before both kickoffs.
Older snapshots without top-ten score probabilities are not backfilled.
"""
import json
from collections import Counter
from pathlib import Path
from persist_prematch import parse_time,load_json,write_json,RESULTS
from double_score_selection import select_diversified,select_balanced,score_coverage

PAIR_DIR=Path("data/prematch/double_pairs")
REPORT=Path("data/prematch/prospective_double_comparison.json")
BALANCED_WEIGHT=0.15  # Frozen from the 2025 calibration; never tune on prospective results.

def candidate_pairs(records):
    eligible=sorted((r for r in records if len(r.get("double_top10_scores",[]))>=5),
                    key=lambda r:(r["kickoff_utc"],str(r["fixture_id"])))
    # Pair consecutive fixtures within a competition; do not reuse a fixture in a batch.
    leagues={}
    for r in eligible:leagues.setdefault(r["competition"],[]).append(r)
    for league,rows in sorted(leagues.items()):
        for i in range(0,len(rows)-1,2):
            a,b=rows[i:i+2]
            if a["fixture_id"]==b["fixture_id"]:continue
            if parse_time(a["captured_at_utc"])>=min(parse_time(a["kickoff_utc"]),parse_time(b["kickoff_utc"])):continue
            if parse_time(b["captured_at_utc"])>=min(parse_time(a["kickoff_utc"]),parse_time(b["kickoff_utc"])):continue
            yield league,a,b

def freeze():
    PAIR_DIR.mkdir(parents=True,exist_ok=True)
    created=0
    for path in sorted(Path("prematch_snapshots").glob("prematch_*.json")):
        payload=load_json(path,{})
        for league,a,b in candidate_pairs(payload.get("records",[])):
            ids=sorted([int(a["fixture_id"]),int(b["fixture_id"])])
            key=f"{league}_{ids[0]}_{ids[1]}"
            destination=PAIR_DIR/(key+".json")
            if destination.exists():continue
            options=[{"Score 1":f"{int(h)}-{int(v)}","Score 2":f"{int(x)}-{int(y)}",
                      "Joint":float(pa)*float(pb)}
                     for h,v,pa in a["double_top10_scores"][:10]
                     for x,y,pb in b["double_top10_scores"][:10]]
            original=sorted(options,key=lambda x:-x["Joint"])[:5]
            diverse=select_diversified(options,5)
            balanced=select_balanced(options,5,diversity_weight=BALANCED_WEIGHT)
            write_json(destination,{"pair_key":key,"league":league,
                "captured_at_utc":a["captured_at_utc"],
                "first_fixture_id":int(a["fixture_id"]),
                "second_fixture_id":int(b["fixture_id"]),
                "first_match":a["home_team"]+" vs "+a["away_team"],
                "second_match":b["home_team"]+" vs "+b["away_team"],
                "first_kickoff_utc":a["kickoff_utc"],"second_kickoff_utc":b["kickoff_utc"],
                "balanced_weight":BALANCED_WEIGHT,
                "original":original,"diversified":diverse,"balanced":balanced})
            created+=1
    return created

def evaluate():
    results=load_json(RESULTS,{})
    rows=[];excluded=Counter()
    for path in sorted(PAIR_DIR.glob("*.json")):
        pair=load_json(path,{})
        first=results.get(str(pair["first_fixture_id"]))
        second=results.get(str(pair["second_fixture_id"]))
        if not first or not second:
            excluded["awaiting_both_verified_results"]+=1
            continue
        actual=(f"{first['home_goals']}-{first['away_goals']}",
                f"{second['home_goals']}-{second['away_goals']}")
        row={"pair_key":pair["pair_key"],"league":pair["league"],
             "first_match":pair["first_match"],"second_match":pair["second_match"],
             "actual_scores":list(actual)}
        for method in ("original","diversified","balanced"):
            choices=pair[method]
            row[method+"_hit"]=any((o["Score 1"],o["Score 2"])==actual for o in choices)
            row[method+"_probability_coverage"]=sum(o["Joint"] for o in choices)
            row[method+"_coverage"]=score_coverage(choices)
        rows.append(row)
    n=len(rows)
    summary={}
    for method in ("original","diversified","balanced"):
        wins=sum(r[method+"_hit"] for r in rows)
        summary[method]={"wins":wins,"win_rate":wins/n if n else None,
                         "mean_estimated_probability_coverage":
                             sum(r[method+"_probability_coverage"] for r in rows)/n if n else None}
    return {"frozen_balanced_weight":BALANCED_WEIGHT,
            "pairs_frozen":len(list(PAIR_DIR.glob("*.json"))),
            "pairs_verified":n,"methods":summary,
            "balanced_only_vs_original":sum(r["balanced_hit"] and not r["original_hit"] for r in rows),
            "original_only_vs_balanced":sum(r["original_hit"] and not r["balanced_hit"] for r in rows),
            "diversified_only_vs_original":sum(r["diversified_hit"] and not r["original_hit"] for r in rows),
            "original_only_vs_diversified":sum(r["original_hit"] and not r["diversified_hit"] for r in rows),
            "exclusions":dict(excluded),"rows":rows,
            "note":"Only genuine frozen pre-kickoff five-combination sets are evaluated. Both actual scores must match within one same row. Old snapshots never backfilled. No automatic promotion."}

if __name__=="__main__":
    created=freeze()
    report=evaluate()
    write_json(REPORT,report)
    print("New immutable prospective double pairs:",created)
    print(json.dumps({k:v for k,v in report.items() if k!="rows"},indent=2))
