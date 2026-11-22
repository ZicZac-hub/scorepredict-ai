"""Immutable prospective comparison of five exact-score double strategies.

Pairs are frozen from the same capture batch, strictly before both kickoffs.
Older snapshots without top-ten score probabilities are not backfilled.
"""
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile
from persist_prematch import parse_time,load_json,write_json,RESULTS
from double_score_selection import select_diversified,select_balanced,score_coverage
from five_double_accuracy import _final, utc

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

def _source(row):
    keys = ("provider", "fixture_id", "home_team_id", "away_team_id",
            "captured_at_utc", "kickoff_utc")
    source = {key: row.get(key) for key in keys}
    if any(value in (None, "") for value in source.values()):
        raise ValueError("Missing original fixture identity or capture time")
    return source


def _publish(path, packet, earliest_kickoff, now):
    """Write the full packet, then publish it without replacing archived picks."""
    contents = json.dumps(packet, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with NamedTemporaryFile(dir=path.parent, mode="w", encoding="utf-8",
                            delete=False) as handle:
        staged = Path(handle.name)
        try:
            handle.write(contents)
            handle.close()
            if (now if now is not None else datetime.now(timezone.utc)) >= earliest_kickoff:
                return False
            path.hardlink_to(staged)
            return True
        except FileExistsError:
            return False
        finally:
            staged.unlink()


def freeze(*, now=None):
    PAIR_DIR.mkdir(parents=True,exist_ok=True)
    now = utc(now) if now is not None else None
    created=0
    for path in sorted(Path("prematch_snapshots").glob("prematch_*.json")):
        payload=load_json(path,{})
        for league,a,b in candidate_pairs(payload.get("records",[])):
            ids=sorted([int(a["fixture_id"]),int(b["fixture_id"])])
            key=f"{league}_{ids[0]}_{ids[1]}"
            destination=PAIR_DIR/(key+".json")
            if destination.exists():continue
            try:
                sources = [_source(row) for row in (a, b)]
                earliest_kickoff = min(utc(row["kickoff_utc"]) for row in sources)
                latest_capture = max(utc(row["captured_at_utc"]) for row in sources)
            except (ValueError, TypeError):
                continue
            options=[{"Score 1":f"{int(h)}-{int(v)}","Score 2":f"{int(x)}-{int(y)}",
                      "Joint":float(pa)*float(pb)}
                     for h,v,pa in a["double_top10_scores"][:10]
                     for x,y,pb in b["double_top10_scores"][:10]]
            original=sorted(options,key=lambda x:-x["Joint"])[:5]
            diverse=select_diversified(options,5)
            balanced=select_balanced(options,5,diversity_weight=BALANCED_WEIGHT)
            frozen_at = now if now is not None else datetime.now(timezone.utc)
            if not latest_capture <= frozen_at < earliest_kickoff:
                continue
            packet = {"pair_key":key,"league":league,
                "captured_at_utc":frozen_at.isoformat(),
                "first_source":sources[0], "second_source":sources[1],
                "first_fixture_id":int(a["fixture_id"]),
                "second_fixture_id":int(b["fixture_id"]),
                "first_provider":sources[0]["provider"],
                "second_provider":sources[1]["provider"],
                "first_match":a["home_team"]+" vs "+a["away_team"],
                "second_match":b["home_team"]+" vs "+b["away_team"],
                "first_kickoff_utc":a["kickoff_utc"],"second_kickoff_utc":b["kickoff_utc"],
                "balanced_weight":BALANCED_WEIGHT,
                "original":original,"diversified":diverse,"balanced":balanced}
            created += _publish(destination, packet, earliest_kickoff, now)
    return created

def evaluate(*, now=None):
    clock = utc(now or datetime.now(timezone.utc))
    results=load_json(RESULTS,{})
    rows=[];excluded=Counter()
    for path in sorted(PAIR_DIR.glob("*.json")):
        pair=load_json(path,{})
        first=results.get(str(pair["first_fixture_id"]))
        second=results.get(str(pair["second_fixture_id"]))
        if "first_source" in pair or "second_source" in pair:
            # New packets bind both sides to the original source identities.
            actuals = [_final(pair.get(field) or {}, result, clock)
                       for field, result in (("first_source", first), ("second_source", second))]
            if any(score is None for score in actuals):
                excluded["awaiting_both_verified_results"] += 1
                continue
            actual = tuple(f"{home}-{away}" for home, away in actuals)
        elif first and second:
            actual=(f"{first['home_goals']}-{first['away_goals']}",
                    f"{second['home_goals']}-{second['away_goals']}")
        else:
            excluded["awaiting_both_verified_results"]+=1
            continue
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
            "note":"Consecutive same-league pairs from a single snapshot batch. New pairs record the decision time before both kickoffs and require identity-matched verified finals. Both scores must match within one saved row. No automatic promotion."}

if __name__=="__main__":
    created=freeze()
    report=evaluate()
    write_json(REPORT,report)
    print("New immutable prospective double pairs:",created)
    print(json.dumps({k:v for k,v in report.items() if k!="rows"},indent=2))
