"""Walk-forward five-double exact-score evaluation.

Pairs consecutive fixtures within each league, no fixture reused; forecasts
use only results strictly before BOTH kickoffs (earlier kickoff). Compares
original top-five joint-probability ranking against diversified five picks.
"""
import argparse,json,math
from collections import defaultdict
from engine import predict,finished,_date
from double_score_selection import select_diversified,score_coverage

def evaluate(matches,year=2026,min_history=30):
    ordered=sorted((m for m in matches if finished(m) and _date(m.get("utcDate"))),
                   key=lambda m:_date(m["utcDate"]))
    leagues=defaultdict(list)
    for m in ordered:leagues[(m.get("competition") or {}).get("code","unknown")].append(m)
    records=[]
    for league,fixtures in leagues.items():
        targets=[m for m in fixtures if _date(m["utcDate"]).year==year]
        for i in range(0,len(targets)-1,2):
            a,b=targets[i:i+2]
            cutoff=min(_date(a["utcDate"]),_date(b["utcDate"]))
            past=[m for m in fixtures if _date(m["utcDate"])<cutoff]
            if len(past)<min_history:continue
            p=predict(a,past,variant="baseline")
            q=predict(b,past,variant="baseline")
            if p is None or q is None:continue
            # Joint exact-score outcomes for a single fixed two-match pair.
            options=[{"Score 1":f"{h}-{v}","Score 2":f"{x}-{y}","Joint":float(pa)*float(pb)}
                     for h,v,pa in p["scores"][:10] for x,y,pb in q["scores"][:10]]
            original=sorted(options,key=lambda o:-o["Joint"])[:5]
            diverse=select_diversified(options,5)
            actual=(f"{a['score']['fullTime']['home']}-{a['score']['fullTime']['away']}",
                    f"{b['score']['fullTime']['home']}-{b['score']['fullTime']['away']}")
            def hit(picks):return any((o["Score 1"],o["Score 2"])==actual for o in picks)
            records.append({"league":league,"first_fixture_id":a["id"],"second_fixture_id":b["id"],
                            "first_kickoff":a["utcDate"],"second_kickoff":b["utcDate"],
                            "actual_scores":list(actual),"top5_hit":hit(original),
                            "diversified_hit":hit(diverse),
                            "top5_joint_probability_sum":sum(o["Joint"] for o in original),
                            "diversified_joint_probability_sum":sum(o["Joint"] for o in diverse),
                            "top5_coverage":score_coverage(original),
                            "diversified_coverage":score_coverage(diverse)})
    n=len(records)
    old=sum(r["top5_hit"] for r in records)
    new=sum(r["diversified_hit"] for r in records)
    win=sum(r["diversified_hit"] and not r["top5_hit"] for r in records)
    lose=sum(r["top5_hit"] and not r["diversified_hit"] for r in records)
    d=win+lose
    pval=min(1.,2*sum(math.comb(d,k) for k in range(min(win,lose)+1))/2**d) if d else 1.
    def mean(field):return sum(r[field] for r in records)/n if n else None
    def coverage_mean(method,field):
        return sum(r[method][field] for r in records)/n if n else None
    return {"holdout_year":year,"pairing":"Consecutive same-league fixtures, each used at most once; common history cutoff before both kickoffs",
            "pairs_evaluated":n,"original_top5_winning_pairs":old,
            "diversified_top5_winning_pairs":new,
            "original_win_rate":old/n if n else None,
            "diversified_win_rate":new/n if n else None,
            "diversified_only_wins":win,"original_only_wins":lose,
            "paired_exact_mcnemar_p":pval,
            "original_mean_joint_probability_coverage":mean("top5_joint_probability_sum"),
            "diversified_mean_joint_probability_coverage":mean("diversified_joint_probability_sum"),
            "original_mean_distinct_first_scores":coverage_mean("top5_coverage","match_1_scores"),
            "diversified_mean_distinct_first_scores":coverage_mean("diversified_coverage","match_1_scores"),
            "original_mean_distinct_second_scores":coverage_mean("top5_coverage","match_2_scores"),
            "diversified_mean_distinct_second_scores":coverage_mean("diversified_coverage","match_2_scores"),
            "by_league":{league:{"pairs":sum(r["league"]==league for r in records),
                                "original_wins":sum(r["league"]==league and r["top5_hit"] for r in records),
                                "diversified_wins":sum(r["league"]==league and r["diversified_hit"] for r in records)}
                         for league in sorted(leagues)},
            "records":records,
            "note":"Five diversified combinations can have greater marginal score coverage but lower total joint probability. No selection tuned to holdout; no past predictions altered."}

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("matches_json")
    parser.add_argument("--output",default="double_score_backtest_results.json")
    args=parser.parse_args()
    with open(args.matches_json,encoding="utf-8") as f:matches=json.load(f)
    result=evaluate(matches)
    with open(args.output,"w",encoding="utf-8") as f:json.dump(result,f,indent=2)
    print(json.dumps({k:v for k,v in result.items() if k!="records"},indent=2))
