"""Read-only, budget-neutral historical statistics sampling research.

The old collector picks recent fixtures; the xG model requires six per team
and at least three home/away venue observations respectively. Compare a
deterministic venue-aware plan on the same authenticated pre-match history.
This is a candidate request plan, not evidence that the provider returns xG.
"""
from collections import Counter
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path

from v10_prospective import utc
import verified_xg_evidence_pipeline as archive

TEAM_MIN = 6
VENUE_MIN = 3


def _candidates(fixture, history, cutoff):
    home, away = str(fixture["homeTeam"]["id"]), str(fixture["awayTeam"]["id"])
    fid, kickoff = str(fixture["id"]), utc(fixture["utcDate"])
    if home == away or not cutoff < kickoff:
        raise ValueError("invalid target")
    rows = {}
    conflicts = set()
    for m in history:
        try:
            if m.get("provider") != "API-Football" or m.get("status") != "FINISHED":
                continue
            at, mid = utc(m["utcDate"]), str(m["id"])
            h, a = str(m["homeTeam"]["id"]), str(m["awayTeam"]["id"])
            if (mid==fid or h==a or not {h,a}&{home,away}
                    or at+timedelta(hours=3)>cutoff or at>=kickoff):
                continue
            item=(at,mid,h,a)
            if mid in rows and rows[mid]!=item:
                conflicts.add(mid)
            rows[mid]=item
        except (KeyError,TypeError,ValueError,AttributeError):
            continue
    return home,away,sorted((r for k,r in rows.items() if k not in conflicts),reverse=True)


def _coverage(rows,home,away):
    return (sum(home in r[2:] for r in rows),sum(away in r[2:] for r in rows),
            sum(r[2]==home for r in rows),sum(r[3]==away for r in rows))


def _quality(rows,home,away):
    c=_coverage(rows,home,away)
    caps=(TEAM_MIN,TEAM_MIN,VENUE_MIN,VENUE_MIN)
    capped=tuple(min(x,y) for x,y in zip(c,caps))
    # Prioritize simultaneous qualification, then balanced team/venue coverage.
    return (int(all(x>=y for x,y in zip(c,caps))),
            min(capped[0],capped[1]),min(capped[2],capped[3]),sum(capped))


def plan(fixture,history,cutoff,limit=6):
    if type(limit) is not int or not 0<=limit<=12:
        raise ValueError("invalid history limit")
    home,away,rows=_candidates(fixture,history,utc(cutoff))
    used={home:0,away:0}
    original=[]
    for r in rows:
        matched=[t for t in used if t in r[2:] and used[t]<limit]
        if matched:
            original.append(r)
            for t in matched:
                used[t]+=1
        if all(n>=limit for n in used.values()):
            break
    candidate=original.copy()
    # Bounded recent pool per context. Only strictly better swaps are accepted.
    # Never increase the actual number of unique historical requests.
    pool=set(original)
    for team in (home,away):
        for venue in (2,3):
            pool.update([r for r in rows if r[venue]==team][:24])
    pool=sorted(pool,reverse=True)
    for _ in range(len(candidate)):
        best=None
        score=_quality(candidate,home,away)
        for old in candidate:
            for new in pool:
                if new in candidate:
                    continue
                trial=[r for r in candidate if r!=old]+[new]
                q=_quality(trial,home,away)
                # Never trade away a team's achieved capped sample count.
                before=_coverage(candidate,home,away)
                after=_coverage(trial,home,away)
                if any(min(after[i],TEAM_MIN)<min(before[i],TEAM_MIN) for i in (0,1)):
                    continue
                if q>score:
                    best,score=trial,q
        if best is None:
            break
        candidate=best
    def detail(items):
        c=_coverage(items,home,away)
        return {"fixture_ids":[r[1] for r in sorted(items,reverse=True)],
                "requests":len(items),"home_team_matches":c[0],"away_team_matches":c[1],
                "home_venue_matches":c[2],"away_venue_matches":c[3],
                "potentially_ready":bool(_quality(items,home,away)[0])}
    return {"recent":detail(original),"venue_aware":detail(candidate),
            "eligible_history_fixtures":len(rows),"additional_requests":0}


def evaluate(*,root=archive.ROOT,provider_root=archive.PROVIDER_ROOT,now=None):
    clock=utc(now or datetime.now(timezone.utc))
    counts,rejects,rows=Counter(),Counter(),[]
    for path in sorted((Path(root)/"history").glob("*.json")):
        try:
            record=archive._authenticated_record(path,provider_root)
            if record is None or utc(record["capture"])>clock:
                rejects["unqualified_history"]+=1
                continue
            fixture,packet=record["fixture"],record["snapshot"]
            limit=(packet.get("source_limits") or {}).get("max_historical_fixtures_per_team")
            if type(limit) is not int or not 0<=limit<=12:
                rejects["missing_or_invalid_original_request_limit"]+=1
                continue
            # Use the authenticated history's own cutoff, never future results.
            raw=json.loads(path.read_text(encoding="utf-8"))
            result=plan(fixture,record["history"],raw["as_of_utc"],limit)
            before,after=result["recent"],result["venue_aware"]
            counts["authenticated_capture_records"]+=1
            counts["original_potentially_ready"]+=int(before["potentially_ready"])
            counts["venue_aware_potentially_ready"]+=int(after["potentially_ready"])
            counts["newly_potentially_ready"]+=int(after["potentially_ready"] and not before["potentially_ready"])
            counts["capture_limit_below_six"]+=int(limit<6)
            counts["actual_packets_with_xg"]+=int(bool(packet.get("verified_historical_fixture_xg")))
            counts["historical_request_slots"]+=before["requests"]
            rows.append({"fixture_id":str(fixture["id"]),"snapshot_sha256":packet["snapshot_sha256"],
                         "limit":limit,**result})
        except (KeyError,ValueError,TypeError,OSError,AttributeError,ArithmeticError):
            rejects["invalid_capture"]+=1
    return {"study_type":"venue_aware_history_request_plan_replay","evaluated_at_utc":clock.isoformat(),
            "counts":dict(counts),"rejections":dict(rejects),"rows":rows,
            "external_requests":0,"automatic_promotion":False,
            "limitations":["Capture records may repeat the same fixture.",
                           "Potential history coverage does not establish genuine provider xG availability.",
                           "No collector or frozen model changed; candidate planner is isolated."]}


if __name__=="__main__":
    print(json.dumps(evaluate(),indent=2,allow_nan=False))
