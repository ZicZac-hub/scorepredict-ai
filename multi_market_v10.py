"""Experimental v10 consensus of several verified pre-match bookmaker markets.

All vendors are part of ONE real API-Football fixture response, and all price
timestamps must precede the model cutoff. Bookmaker-margin-adjusted 1X2 and
goal total probabilities are aggregated with a robust median. This is a
separate unvalidated challenger; it never replaces the original v8/v9.
"""
from datetime import datetime, timedelta, timezone
import math
import numpy as np

from market_score_v9 import _time, _market_bets, _project, MARKET_WEIGHT, _summarize, MAX_MARKET_AGE_HOURS, NEAR_FRESHEST_HOURS
from research_ensemble_v8 import predict as research_predict

VERSION="ziczac-multibook-score-v10-shadow-experimental"
MIN_BOOKMAKERS=3


def market_consensus(items,fixture,as_of=None,min_bookmakers=MIN_BOOKMAKERS):
    cutoff=_time(as_of) if as_of is not None else datetime.now(timezone.utc)
    kickoff=_time(fixture.get("utcDate"))
    if not cutoff or not kickoff or cutoff>=kickoff:
        return None
    vendors={}
    for item in items if isinstance(items,list) else []:
        if str((item.get("fixture") or {}).get("id"))!=str(fixture.get("id")):
            continue
        stamped=_time(item.get("update"))
        if not stamped or stamped>cutoff or cutoff-stamped>timedelta(hours=MAX_MARKET_AGE_HOURS):
            continue
        for bookmaker in item.get("bookmakers") or []:
            name=str(bookmaker.get("name") or "").strip()
            if not name:
                continue
            winner,total=_market_bets(bookmaker)
            if not winner:
                continue
            key=name.casefold()
            # One vote per named bookmaker, latest verified snapshot only.
            if key not in vendors or stamped>vendors[key]["timestamp"]:
                vendors[key]={"name":name,"timestamp":stamped,
                              "winner":winner,"total":total}
    if not vendors:
        return None
    # Markets must refer to essentially the SAME pre-match information.
    # Median probabilities across quotes 12 hours apart can blend stale
    # team news with current prices and make score forecasts misleading.
    freshest=max(v["timestamp"] for v in vendors.values())
    contemporary=[v for v in vendors.values()
                  if freshest-v["timestamp"]<=timedelta(hours=NEAR_FRESHEST_HOURS)]
    if len(contemporary)<min_bookmakers:
        return None
    selected=sorted(contemporary,key=lambda row:row["name"].casefold())
    first=np.median(np.asarray([v["winner"][0] for v in selected]),axis=0)
    first=first/first.sum()
    totals=[v["total"][0] for v in selected if v["total"] is not None]
    second=None
    if len(totals)>=min_bookmakers:
        second=np.median(np.asarray(totals),axis=0)
        second=tuple(float(p/second.sum()) for p in second)
    return {
        "fixture_id":fixture["id"],"snapshot_utc":max(v["timestamp"] for v in selected).isoformat(),
        "winner":tuple(float(p) for p in first),"totals":second,
        "bookmaker":"Median of "+str(len(selected))+" independently quoted bookmakers",
        "bookmaker_count":len(selected),
        "bookmaker_names":[v["name"] for v in selected],
        "winner_overround":None,"totals_overround":None,
        "settlement":"regulation_90_minutes",
        "aggregation":"median_margin_removed_probabilities_not_independent_AI_votes",
        "validated_accuracy":False}


def predict(fixture,history,league_history=None,market=None,as_of=None,max_goals=10):
    if market is None or int(market.get("bookmaker_count",0))<MIN_BOOKMAKERS:
        return None
    instant=_time(as_of) if as_of is not None else datetime.now(timezone.utc)
    kickoff=_time(fixture.get("utcDate"))
    stamp=_time(market.get("snapshot_utc"))
    if (not instant or not stamp or not kickoff or instant>=kickoff
            or stamp>instant or instant-stamp>timedelta(hours=MAX_MARKET_AGE_HOURS)
            or str(market.get("fixture_id"))!=str(fixture.get("id"))):
        raise ValueError("Market consensus is not a valid pre-kickoff snapshot")
    baseline=research_predict(fixture,history,league_history=league_history,
                              as_of=instant,max_goals=max_goals)
    if baseline is None:
        return None
    changed=_project(baseline["matrix"],market)
    matrix=(1-MARKET_WEIGHT)*baseline["matrix"]+MARKET_WEIGHT*changed
    matrix/=matrix.sum()
    return {
        **_summarize(matrix,baseline,market),
        "model_version":VERSION,
        "market_used":True,
        "source_type":"verified_multi_bookmaker_median",
        "bookmakers_count":market["bookmaker_count"],
        "release_status":"challenger_not_prospectively_validated",
        "llm_provider_calls":0}
