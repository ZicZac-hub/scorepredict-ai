"""Development-only component ablation on the previously explored 2024–25 cohort."""
import json
import math
from datetime import timedelta
from backtest_dynamic_dixon_coles import evaluate, SYDNEY
from dynamic_dixon_coles import fit_forecast, five_doubles
from engine import _date
from run_dynamic_dc_pinned_replay import load_matches

CONFIGS={"full":(True,True),"without_team_venue":(False,True),
         "without_goal_band":(True,False),"without_both":(False,False)}

def compare(matches):
    reference=evaluate(matches,start="2024-08-01")
    index={m["id"]:m for m in matches}
    rows={k:{"pairs":0,"hits":0,"sum_log_loss":0.,"sum_coverage":0.} for k in CONFIGS}
    unavailable=0
    for trial in reference["matched_trials"]:
        pair=[index[x] for x in trial["fixture_ids"]]
        cutoff=min(_date(m["utcDate"]) for m in pair)-timedelta(days=1)
        past=[m for m in matches if _date(m["utcDate"])<cutoff
              and _date(m["utcDate"]).astimezone(SYDNEY).date().isoformat()<trial["sydney_date"]]
        forecasts={k:[fit_forecast(m,past,venue_deviations=v,goal_band_calibration=g)
                      for m in pair] for k,(v,g) in CONFIGS.items()}
        if any(f is None for group in forecasts.values() for f in group):
            unavailable+=1
            continue
        actual=tuple((m["score"]["fullTime"]["home"],m["score"]["fullTime"]["away"]) for m in pair)
        for k,group in forecasts.items():
            a,b=(f["matrix"] for f in group)
            picks=five_doubles(a,b)
            row=rows[k]
            row["pairs"]+=1
            row["hits"]+=int(any((s,t)==actual for _,s,t in picks))
            row["sum_coverage"]+=sum(p for p,_,_ in picks)
            for matrix,(h,g) in zip((a,b),actual):
                p=float(matrix[h,g]) if h<matrix.shape[0] and g<matrix.shape[1] else 0.
                row["sum_log_loss"]-=math.log(max(p,1e-12))
    for row in rows.values():
        n=row["pairs"]
        row["hit_rate"]=row["hits"]/n if n else None
        row["score_log_loss"]=row.pop("sum_log_loss")/(2*n) if n else None
        row["mean_coverage"]=row.pop("sum_coverage")/n if n else None
    return {"reference_pairs":reference["matched_pairs"],"ablation_unavailable":unavailable,
            "models":rows,"development_only":True,"production_promotion":False}

if __name__=="__main__":
    matches,missing=load_matches()
    print(json.dumps(compare([m for m in matches if m["utcDate"][:10] not in missing]),indent=2))
