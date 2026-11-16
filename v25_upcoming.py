"""On-demand V25 double generation for any genuinely modelled future pair.

Uses existing ZicZac complete score distributions and four fixed mathematical
sensitivity transforms. Unlike the scheduled 24-hour V25 shadow experiment,
this is an EXPLICITLY USER-REQUESTED, session-only early research estimate.
A match must have a real normalized model grid; no fake score grids are
constructed from a few suggested scores. Both matches must have distinct
teams and start on the same Sydney day.

The mathematical alternatives share original goal means. The 'robust'
lower-envelope criterion is NOT a calibrated success probability.
"""
from collections import Counter, defaultdict
from datetime import datetime, timezone
from itertools import combinations
import math

from ai_five_double_engine import eligible, identity, pair_check, teams, SYDNEY
from professional_analyst_shadow_v25 import _pair, MAX_EARLY_SOURCE_AGE
from v10_prospective import utc
from v21_goal_model_comparison import variants

VERSION = "v25-on-demand-robust-sensitivity-experimental"
MAX_PAGE_PAIRS = 15
MAX_AUTO_EVALUATION = 150


def _source_time(row, clock):
    raw = (row.get("frozen_at_utc") if row.get("archive_frozen") else
           (row.get("generated_at_utc") or row.get("captured_at_utc")))
    if raw is None:
        return clock if not row.get("archive_frozen") else None
    try:
        return utc(raw)
    except (ValueError, TypeError):
        return None


def qualified(row, now):
    """Validate the goal matrix, future fixture and saved source timestamps."""
    clock = utc(now)
    if not isinstance(row, dict) or not eligible(row, now=clock):
        return None, "invalid_or_started_fixture"
    fixture = row["fixture"]
    if any(not str((fixture.get(side+"Team") or {}).get("id") or "").strip()
           for side in ("home", "away")):
        return None, "missing_team_identifiers"
    kickoff = utc(row.get("kickoff") or fixture["utcDate"])
    frozen = _source_time(row, clock)
    if frozen is None or frozen > clock or frozen >= kickoff:
        return None, "invalid_or_future_source_timestamp"
    if row.get("archive_frozen") and clock-frozen > MAX_EARLY_SOURCE_AGE:
        return None, "source_older_than_14_days"
    if (row.get("result") or {}).get("score_basis") not in (None,"regulation"):
        return None, "not_regulation_time"
    try:
        import numpy as np
        original = np.asarray((row.get("result") or {}).get("matrix"),dtype=float)
        if (original.ndim!=2 or min(original.shape)<5 or max(original.shape)>25
                or not np.isfinite(original).all() or np.any(original<0)
                or not np.isclose(float(original.sum()),1.0,atol=1e-6)):
            return None, "missing_complete_original_score_grid"
        alternatives=variants(original)
        if any(not np.isfinite(m).all() for m in alternatives.values()):
            return None, "invalid_mathematical_sensitivity"
    except (ValueError,TypeError,KeyError,AttributeError,ArithmeticError):
        return None, "missing_complete_original_score_grid"
    return {
        "row":row, "key":identity(row),
        "names":teams(row),
        "title":(str(fixture["homeTeam"]["name"]) + " vs " +
                 str(fixture["awayTeam"]["name"])),
        "kickoff":kickoff,
        "day":kickoff.astimezone(SYDNEY).date().isoformat(),
        "captured_at_utc":frozen.isoformat(),
        "early_source":(clock-frozen).total_seconds()>72*3600,
        "archive_frozen":bool(row.get("archive_frozen")),
        "alternatives":alternatives,
    },None


def available(rows, now):
    """Eligible complete-grid matches grouped by Sydney date, without odds calls."""
    output = defaultdict(list)
    excluded = Counter()
    seen = {}
    for row in rows or []:
        candidate,reason=qualified(row,now)
        if candidate is None:
            excluded[reason]+=1
            continue
        key=candidate["key"]
        old=seen.get(key)
        # Prefer most-recent genuinely available complete source on the same
        # fixture identity, preserving the old archive file independently.
        if old is None or candidate["captured_at_utc"]>old["captured_at_utc"]:
            seen[key]=candidate
    for candidate in seen.values():
        output[candidate["day"]].append(candidate)
    for values in output.values():
        values.sort(key=lambda x:(x["kickoff"],x["key"]))
    return dict(sorted(output.items())),dict(excluded)


def _pair_issue(first,second,now):
    issue=pair_check(first["row"],second["row"],now=now)
    if issue:
        return issue
    # A provider can rename/translate the same team across competitions.
    # Validate actual team IDs in addition to display names; V25 must never
    # generate a joint bet on one team twice under different labels.
    a=first["row"]["fixture"]
    b=second["row"]["fixture"]
    if str(a.get("provider") or "football-data.org")==str(
            b.get("provider") or "football-data.org"):
        identifiers=[
            str((fixture.get(side+"Team") or {}).get("id"))
            for fixture in (a,b) for side in ("home","away")]
        if len(set(identifiers))!=4:
            return "The same provider team appears in both matches."
    return None


def pair_catalog(candidates, now, *, search=""):
    """List EVERY available same-Sydney-day four-team pair without modelling it."""
    rows=sorted(candidates,key=lambda x:(x["kickoff"],x["key"]))
    result=[]
    query=search.strip().casefold()
    for a,b in combinations(rows,2):
        if query and query not in (a["title"]+" "+b["title"]).casefold():
            continue
        if _pair_issue(a,b,now):
            continue
        result.append((a,b))
    return result


def generate(first,second,now):
    """Five V25 score combinations, with explicit independent V8 comparison."""
    clock=utc(now)
    issue=_pair_issue(first,second,clock)
    if issue:
        return {"status":"ineligible_pair","reason":issue}
    # A model's saved evidence must still have been recorded before both
    # kickoffs. Revalidation prevents a stale Streamlit cache being re-used.
    for item in (first,second):
        current,why=qualified(item["row"],clock)
        if current is None or current["key"] != item["key"]:
            return {"status":"ineligible_evidence","reason":why or "Fixture changed"}
    metrics=_pair(first,second)
    selection=metrics["five_doubles"]
    if len(selection)!=5 or len({(p["score1"],p["score2"]) for p in selection})!=5:
        return {"status":"insufficient_model_support"}
    return {
        "status":"ready",
        "method":VERSION,
        "generated_at_utc":clock.isoformat(),
        "sydney_date":first["day"],
        "first":first["title"],"second":second["title"],
        "first_fixture":first["row"]["fixture"],
        "second_fixture":second["row"]["fixture"],
        "scores":selection,
        "v8_same_pair_scores":metrics["baseline_five"],
        "v8_optimal_five_estimated_mass":metrics["v8_optimal_five_model_mass"],
        "v25_five_original_v8_mass":metrics["robust_five_v8_model_mass"],
        "v25_lower_envelope_criterion":metrics["robust_lower_envelope_model_mass"],
        "source_times_utc":[first["captured_at_utc"],second["captured_at_utc"]],
        "early_frozen_source":[first["early_source"],second["early_source"]],
        "source_note":"An early source is NOT fresh team news or new bookmaker odds",
        "all_model_variants_share_original_mean":True,
        "accuracy_70pct_not_proven":True,
        "model_estimates_not_real_hit_rate":True,
        "not_auto_promoted":True,
        "no_external_api_calls":True,
    }


def best_pair(catalog, now, *, limit=MAX_AUTO_EVALUATION):
    """Auto screen the same-date catalog; never imply unseen pairs were checked."""
    if not catalog:
        return {"status":"no_pairs","screened":0,"total_pairs":0}
    # For large slates use the top pairs by V8 five-cell mass as a transparent
    # shortlist; do not call it a global optimum when not all pairs were tried.
    def potential(pair):
        a,b=pair
        import numpy as np
        p=a["alternatives"]["baseline_v8"].ravel()
        q=b["alternatives"]["baseline_v8"].ravel()
        return sum(sorted((float(x*y) for x in np.sort(p)[-5:]
                           for y in np.sort(q)[-5:]),reverse=True)[:5])
    ranked=sorted(catalog,key=lambda pair:(-potential(pair),
        pair[0]["key"],pair[1]["key"]))
    selected=ranked[:max(1,limit)]
    best=None
    for first,second in selected:
        report=generate(first,second,now)
        if report.get("status")!="ready":
            continue
        grade=report["v25_lower_envelope_criterion"]
        if best is None or grade>best[0]:
            best=(grade,first,second,report)
    if best is None:
        return {"status":"no_qualified_model","screened":len(selected),
                "total_pairs":len(catalog)}
    return {"status":"ready","first":best[1],"second":best[2],
            "report":best[3],"screened":len(selected),
            "total_pairs":len(catalog),
            "shortlist_only":len(selected)<len(catalog)}


def rows_for_page(catalog, page, now, *, page_size=MAX_PAGE_PAIRS):
    """Batch ALL pairs in user-selected pages, never silently skip later pages."""
    total=len(catalog)
    page=max(0,int(page))
    start=page*page_size
    collected=[]
    for first,second in catalog[start:start+page_size]:
        report=generate(first,second,now)
        if report["status"]!="ready":
            continue
        for i,pick in enumerate(report["scores"],1):
            collected.append({
                "Sydney date":report["sydney_date"],
                "Match 1":report["first"],"Match 2":report["second"],
                "Double":i,"Score 1":pick["score1"],"Score 2":pick["score2"],
                "V25 model":"Four goal-distribution sensitivity models",
                "Source age warning":any(report["early_frozen_source"]),
            })
    return {"total_pairs":total,"page":page,"pages":math.ceil(total/page_size),
            "shown_pairs":min(page_size,max(0,total-start)),
            "rows":collected,"not_a_bet_slip":True}
