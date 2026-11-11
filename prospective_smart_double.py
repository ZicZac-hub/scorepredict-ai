"""Prospective Smart Double AI: freeze highest-coverage same-day pair before kickoff.

Uses captured model score distributions, never retroactively manufactures picks.
"""
import json
from pathlib import Path
from datetime import timezone
from persist_prematch import parse_time, load_json, write_json, RESULTS
from smart_double_ai import rank_pairs

OUT=Path("data/prematch/smart_double_pairs")
REPORT=Path("data/prematch/smart_double_comparison.json")

def eligible_record(r):
    scores=r.get("double_top10_scores", [])
    if len(scores)<5:
        return None
    try:
        kickoff=parse_time(r["kickoff_utc"])
        captured=parse_time(r["captured_at_utc"])
        if captured>=kickoff:
            return None
        return {"fixture":{"id":r["fixture_id"],"provider":"prematch",
                 "homeTeam":{"name":r["home_team"]},
                 "awayTeam":{"name":r["away_team"]}},
                "kickoff":kickoff.astimezone(timezone.utc).isoformat(),
                "captured":captured,
                "result":{"scores":scores}}
    except (KeyError,TypeError,ValueError):
        return None

def freeze():
    OUT.mkdir(parents=True,exist_ok=True)
    created=0
    for path in sorted(Path("prematch_snapshots").glob("prematch_*.json")):
        batch=load_json(path,{})
        rows=[x for r in batch.get("records",[]) if (x:=eligible_record(r))]
        by_day={}
        for row in rows:
            by_day.setdefault(row["kickoff"][:10],[]).append(row)
        for day,day_rows in sorted(by_day.items()):
            destination=OUT/(day+".json")
            if destination.exists():
                continue
            candidates=rank_pairs(day_rows,limit=1)
            if not candidates:
                continue
            best=candidates[0]
            a,b=best["first"],best["second"]
            if max(a["captured"],b["captured"])>=min(parse_time(a["kickoff"]),parse_time(b["kickoff"])):
                continue
            picks=[{"score1":o["Score 1"],"score2":o["Score 2"],"probability":o["Joint"]}
                   for o in best["doubles"]]
            write_json(destination,{"day_utc":day,"source_snapshot":path.name,
                "captured_at_utc":max(a["captured"],b["captured"]).isoformat(),
                "first_fixture_id":str(a["fixture"]["id"]),
                "second_fixture_id":str(b["fixture"]["id"]),
                "first_match":a["fixture"]["homeTeam"]["name"]+" vs "+a["fixture"]["awayTeam"]["name"],
                "second_match":b["fixture"]["homeTeam"]["name"]+" vs "+b["fixture"]["awayTeam"]["name"],
                "first_kickoff_utc":a["kickoff"],"second_kickoff_utc":b["kickoff"],
                "estimated_coverage":best["coverage"],"five_doubles":picks})
            created+=1
    return created

def evaluate():
    results=load_json(RESULTS,{})
    rows=[]
    pending=0
    for path in sorted(OUT.glob("*.json")):
        pair=load_json(path,{})
        a=results.get(pair["first_fixture_id"])
        b=results.get(pair["second_fixture_id"])
        if not a or not b:
            pending+=1
            continue
        actual1=f"{a['home_goals']}-{a['away_goals']}"
        actual2=f"{b['home_goals']}-{b['away_goals']}"
        hit=any(p["score1"]==actual1 and p["score2"]==actual2
                for p in pair["five_doubles"])
        rows.append({"day_utc":pair["day_utc"],"first_match":pair["first_match"],
                     "second_match":pair["second_match"],"actual1":actual1,
                     "actual2":actual2,"hit":hit,
                     "estimated_coverage":pair["estimated_coverage"]})
    wins=sum(x["hit"] for x in rows)
    return {"frozen_days":len(list(OUT.glob("*.json"))),"verified_days":len(rows),
            "pending_days":pending,"wins":wins,
            "win_rate":wins/len(rows) if rows else None,"rows":rows,
            "note":"Only genuine frozen pre-kickoff pairs with two verified full-time results count. No win guarantees."}

if __name__=="__main__":
    created=freeze()
    report=evaluate()
    write_json(REPORT,report)
    print("New Smart Double frozen days:",created)
    print(json.dumps({k:v for k,v in report.items() if k!="rows"},indent=2))
