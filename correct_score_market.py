"""Retain genuine 90-minute exact-score prices from an already fetched response.

Partial score-price lists are never normalized into fair probabilities. These
are single-leg observations, not executable bookmaker two-match quotes.
"""
from datetime import timedelta
import math
import re

from v10_prospective import utc

_SCORE=re.compile(r"^(\d{1,2})\s*[-:]\s*(\d{1,2})$")
_FULL_TIME_NAMES={"correct score","exact score","full time correct score",
                  "fulltime correct score","correct score (full time)",
                  "correct score - full time"}


def parse_correct_score_quotes(raw,fixture,*,observed_at,
                               verified_provider_fetched_at=None,max_age_hours=12):
    result={"ready":False,"quotes":[],"no_fair_probabilities_inferred":True,
            "executable_double_quote":False,"partial_score_market_is_not_normalized":True}
    try:
        clock=utc(observed_at)
        kickoff=utc(fixture["utcDate"])
        fid=str(fixture["id"])
        if (fixture.get("provider")!="API-Football" or clock>=kickoff or
                not fid.isdecimal() or int(fid)<=0):
            result["reason"]="verified_pre_match_API_Football_fixture_required"
            return result
    except (KeyError,TypeError,ValueError):
        result["reason"]="verified_pre_match_API_Football_fixture_required"
        return result
    seen={}
    for item in raw if isinstance(raw,list) else []:
        if not isinstance(item,dict):
            continue
        identity=item.get("fixture") or {}
        if str(identity.get("id"))!=fid:
            continue
        try:
            if identity.get("date") and abs((utc(identity["date"])-kickoff).total_seconds())>300:
                continue
        except (TypeError,ValueError):
            continue
        teams=item.get("teams") or {}
        if any(teams.get(side) and str(teams[side].get("id"))!=
               str((fixture.get(side+"Team") or {}).get("id")) for side in ("home","away")):
            continue
        for book in item.get("bookmakers") or []:
            if not isinstance(book,dict) or not str(book.get("name") or "").strip():
                continue
            name=str(book["name"]).strip()[:90]
            for bet in book.get("bets") or []:
                if not isinstance(bet,dict) or " ".join(str(bet.get("name") or "").casefold().split()) not in _FULL_TIME_NAMES:
                    continue
                update=bet.get("update") or book.get("update") or item.get("update")
                basis="bookmaker_quote_updated_at" if update else "api_response_fetched_at"
                try:
                    when=utc(update or verified_provider_fetched_at)
                    if not (when<=clock<kickoff and clock-when<=timedelta(hours=max_age_hours)):
                        continue
                except (TypeError,ValueError):
                    continue
                for value in bet.get("values") or []:
                    if not isinstance(value,dict):
                        continue
                    match=_SCORE.fullmatch(str(value.get("value") or "").strip())
                    try:
                        price=float(value["odd"])
                        if type(value["odd"]) is bool or not math.isfinite(price) or price<=1:
                            continue
                    except (KeyError,TypeError,ValueError):
                        continue
                    if match is None:
                        continue
                    score=[int(match[1]),int(match[2])]
                    key=(name,score[0],score[1])
                    quote={"provider":"API-Football","fixture_id":fid,
                           "kickoff_utc":kickoff.isoformat(),"bookmaker":name,
                           "market":"correct_score_90min","score":score,
                           "decimal_odds":price,"updated_at_utc":when.isoformat(),
                           "observed_at_utc":clock.isoformat(),"timestamp_basis":basis,
                           "source":"API-Football-prematch-odds",
                           "fair_probability":None,"executable_double_quote":False}
                    previous=seen.get(key)
                    if previous is None or when>utc(previous["updated_at_utc"]):
                        seen[key]=quote
    result["quotes"]=sorted(seen.values(),key=lambda q:(q["bookmaker"],q["score"]))
    result["ready"]=bool(result["quotes"])
    if not result["ready"]:
        result["reason"]="no_matching_time_checked_full_time_correct_score_prices"
    return result
