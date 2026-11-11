"""Evaluate only immutable pre-kickoff paired predictions and verified results."""
import json
from collections import Counter
from pathlib import Path
from persist_prematch import earliest_snapshots,load_json,RESULTS,write_json

OUTPUT=Path("data/prematch/prospective_comparison.json")

def evaluate(snapshots,results):
    pairs=[]
    exclusions=Counter()
    for fid,row in snapshots.items():
        result=results.get(str(fid))
        if not result:
            exclusions["not_verified"]+=1
            continue
        challenger=row.get("prospective_challenger")
        if not isinstance(challenger,dict) or not challenger.get("available"):
            exclusions["no_pre_kickoff_challenger"]+=1
            continue
        baseline=row.get("predicted_score")
        alternative=challenger.get("predicted_score")
        if not (isinstance(baseline,list) and len(baseline)==2 and isinstance(alternative,list) and len(alternative)==2):
            exclusions["invalid_prediction"]+=1
            continue
        actual=[result["home_goals"],result["away_goals"]]
        b=baseline==actual;c=alternative==actual
        pairs.append({"fixture_id":fid,"kickoff_utc":row["kickoff_utc"],
            "captured_at_utc":row["captured_at_utc"],
            "original_baseline":baseline,"frozen_challenger":alternative,
            "actual":actual,"baseline_hit":b,"challenger_hit":c,
            "challenger_top3_hit":actual in challenger.get("top3_scores",[])})
    n=len(pairs)
    b=sum(r["baseline_hit"] for r in pairs)
    c=sum(r["challenger_hit"] for r in pairs)
    return {"paired_verified_matches":n,"baseline_exact_hits":b,
            "challenger_exact_hits":c,
            "baseline_exact_accuracy":b/n if n else None,
            "challenger_exact_accuracy":c/n if n else None,
            "challenger_top3_hits":sum(r["challenger_top3_hit"] for r in pairs),
            "challenger_only_hits":sum(r["challenger_hit"] and not r["baseline_hit"] for r in pairs),
            "baseline_only_hits":sum(r["baseline_hit"] and not r["challenger_hit"] for r in pairs),
            "exclusions":dict(exclusions),"rows":pairs,
            "note":"Earliest pre-kickoff snapshot only. No retroactive challenger predictions, no changed originals, no automatic model promotion."}

if __name__=="__main__":
    report=evaluate(earliest_snapshots(),load_json(RESULTS,{}))
    write_json(OUTPUT,report)
    print(json.dumps({k:v for k,v in report.items() if k!="rows"},indent=2))
