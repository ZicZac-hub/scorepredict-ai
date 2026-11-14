"""As-of-safe context for v10 model research without inventing xG or injuries.

Only completed historical matches already filtered by the common
two-season pre-match evidence policy enter these diagnostics. Rest is a
*match-schedule feature*, not an xG/lineup estimate and it does not
silently change the outcome probabilities before validation.
"""
from datetime import datetime,timezone
import math
from research_ensemble_v8 import _asof_history
from v10_prospective import utc


def describe(fixture,evidence):
    cutoff=utc(evidence["as_of_utc"])
    kickoff=utc(fixture["utcDate"])
    population=_asof_history(fixture,evidence.get("league_history") or [],cutoff)
    counts={}
    for side in ("home","away"):
        team_id=(fixture.get(side+"Team") or {}).get("id")
        games=[r for r in population if team_id in (
            (r.get("homeTeam") or {}).get("id"),
            (r.get("awayTeam") or {}).get("id"))]
        games=sorted(games,key=lambda game:utc(game["utcDate"]))
        latest=utc(games[-1]["utcDate"]) if games else None
        rest=round((kickoff-latest).total_seconds()/86400,2) if latest else None
        counts[side]={
            "finished_matches":len(games),"last10_observed":min(10,len(games)),
            "rest_days_since_last_kickoff":rest,
            "last_match_utc":latest.isoformat() if latest else None}
    both=counts["home"]["rest_days_since_last_kickoff"],counts["away"]["rest_days_since_last_kickoff"]
    return {
        "as_of_utc":cutoff.isoformat(),
        "home":counts["home"],"away":counts["away"],
        "rest_days_differential":round(both[0]-both[1],2)
            if all(x is not None for x in both) else None,
        "league_completed_matches":len(population),
        "verified_xg_xga":{"available":False,"reason":"No licensed timestamped true xG feed configured; shots or scoring averages are not labelled xG."},
        "verified_lineup_adjustments":{"applied":False,
             "reason":"Unvalidated player and formation coefficients are not applied to model goal rates."},
        "market_context":{"available":None,"note":"See separate source-stamped market snapshot on v9 forecasts."},
        "numeric_adjustments_applied":[],
        "research_status":"descriptive_verified_schedule_context_only"
    }
