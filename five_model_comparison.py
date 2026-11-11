"""Prospective comparison of frozen five-model consensus versus baseline."""
import json
from collections import Counter
from pathlib import Path
from persist_prematch import earliest_snapshots,load_json,RESULTS,write_json

OUTPUT=Path("data/prematch/five_model_comparison.json")

def evaluate(snapshots,results):
    rows=[];exclusions=Counter()
    for fid,row in snapshots.items():
        actual_result=results.get(str(fid))
        if not actual_result:
            exclusions["not_verified"]+=1
            continue
        challenger=row.get("five_model_consensus")
        if not isinstance(challenger,dict) or not challenger.get("available"):
            exclusions["missing_frozen_consensus"]+=1
            continue
        baseline=row.get("predicted_score")
        predicted=challenger.get("predicted_score")
        if not (isinstance(baseline,list) and len(baseline)==2 and
                isinstance(predicted,list) and len(predicted)==2):
            exclusions["invalid_score"]+=1
            continue
        actual=[actual_result["home_goals"],actual_result["away_goals"]]
        rows.append({"fixture_id":fid,"kickoff_utc":row["kickoff_utc"],
                     "captured_at_utc":row["captured_at_utc"],
                     "baseline":baseline,"consensus":predicted,"actual":actual,
                     "baseline_hit":baseline==actual,"consensus_hit":predicted==actual,
                     "consensus_top3_hit":actual in challenger.get("top3_scores",[])})
    n=len(rows)
    b=sum(r["baseline_hit"] for r in rows)
    c=sum(r["consensus_hit"] for r in rows)
    return {"paired_verified_matches":n,"baseline_exact_hits":b,
            "consensus_exact_hits":c,
            "baseline_exact_accuracy":b/n if n else None,
            "consensus_exact_accuracy":c/n if n else None,
            "consensus_top3_hits":sum(r["consensus_top3_hit"] for r in rows),
            "consensus_only_hits":sum(r["consensus_hit"] and not r["baseline_hit"] for r in rows),
            "baseline_only_hits":sum(r["baseline_hit"] and not r["consensus_hit"] for r in rows),
            "exclusions":dict(exclusions),"rows":rows,
            "note":"Earliest immutable pre-kickoff snapshots only; no retroactive challenger creation or production promotion."}

if __name__=="__main__":
    report=evaluate(earliest_snapshots(),load_json(RESULTS,{}))
    write_json(OUTPUT,report)
    print(json.dumps({k:v for k,v in report.items() if k!="rows"},indent=2))
