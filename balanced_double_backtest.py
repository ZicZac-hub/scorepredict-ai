"""Choose balanced five-double weight using 2025 only, freeze for 2026.

The 2026 holdout is reported, never used to choose a weight.
"""
import argparse,json
from double_score_backtest import evaluate

CANDIDATES=(0.0,0.05,0.10,0.15,0.25,0.40)

def run(matches):
    calibration=[]
    for weight in CANDIDATES:
        report=evaluate(matches,year=2025,balanced_weight=weight)
        calibration.append({"weight":weight,"pairs":report["pairs_evaluated"],
                            "balanced_wins":report["balanced_top5_winning_pairs"],
                            "win_rate":report["balanced_win_rate"],
                            "joint_probability_coverage":report["balanced_mean_joint_probability_coverage"]})
    # Select only using 2025 wins; ties prefer higher estimated coverage,
    # then lower diversity weight to avoid unnecessary complexity.
    if not any(x["pairs"] for x in calibration):
        return {"status":"insufficient_2025_calibration","calibration":calibration,
                "promotion_recommended":False}
    chosen=max(calibration,key=lambda x:(x["balanced_wins"],
                                         x["joint_probability_coverage"] or 0,-x["weight"]))
    holdout=evaluate(matches,year=2026,balanced_weight=chosen["weight"])
    b_only=holdout["balanced_only_vs_original"]
    o_only=holdout["original_only_vs_balanced"]
    return {"status":"evaluated","selection_year":2025,"holdout_year":2026,
            "calibration":calibration,"frozen_selected_weight":chosen["weight"],
            "holdout":holdout,"paired_discordant_balanced_only":b_only,
            "paired_discordant_original_only":o_only,
            "promotion_recommended":False,
            "note":"Weight selected on 2025 only. Compare 2026 without tuning to holdout; production remains unchanged."}

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("matches_json")
    p.add_argument("--output",default="balanced_double_backtest_results.json")
    args=p.parse_args()
    with open(args.matches_json,encoding="utf-8") as f:matches=json.load(f)
    result=run(matches)
    with open(args.output,"w",encoding="utf-8") as f:json.dump(result,f,indent=2)
    print(json.dumps({k:v for k,v in result.items() if k!="holdout"},indent=2))
    if "holdout" in result:
        print(json.dumps({k:v for k,v in result["holdout"].items() if k!="records"},indent=2))
