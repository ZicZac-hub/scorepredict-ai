"""V22 typed pre-kickoff research: genuine xG, multi-market odds, lineup checks.

All scoring is SHADOW only. Facts are used only if source identity, update
timestamp and provenance match. No API calls are made here. Injury/lineup
opinions never receive invented player-value coefficients.
"""
import hashlib
import json
import math
import re
from datetime import timedelta

import numpy as np
from v10_prospective import utc
from v21_provider_evidence import valid_snapshot


_MARKETS = ("1x2", "over_2_5", "over_3_5", "btts_yes")


def _time(value):
    try:
        return utc(value)
    except (TypeError, ValueError, KeyError):
        return None


def _price(value):
    try:
        n = float(value)
        return n if math.isfinite(n) and 1.001 < n < 200 else None
    except (TypeError, ValueError):
        return None


def _normalize(odds):
    if not odds or any(_price(v) is None for v in odds.values()):
        return None
    inv = {k: 1.0 / float(v) for k, v in odds.items()}
    total = sum(inv.values())
    if not 1.0 <= total <= 1.35:
        return None
    return {k: round(v / total, 8) for k, v in inv.items()}


def parse_markets(raw, *, observed_at, max_age_hours=12, verified_provider_fetched_at=None,
                  diagnostics=None):
    """Collect timestamped pre-match 1X2, BTTS, O/U 2.5 and 3.5.

    Multiple books may contribute but each named book must have a complete
    outcome set for that market. Never mix odds across different bookmakers.
    Prefer the bookmaker's actual quote update timestamp. When the
    bookmaker does not supply one, permit the original timestamp of a
    **verified API response fetch** (including a fresh cache) as an
    observation, distinctly labelled as such. NEVER use clock-now or
    stale-fallback response time as a fabricated bookmaker update.
    """
    seen = {}
    def count(reason):
        if diagnostics is not None:
            diagnostics[reason] = diagnostics.get(reason, 0) + 1
    now = utc(observed_at)
    for fixture in raw if isinstance(raw, list) else []:
        if not isinstance(fixture, dict):
            count("malformed_fixture")
            continue
        for book in fixture.get("bookmakers") or []:
            if not isinstance(book, dict):
                count("malformed_bookmaker")
                continue
            bookmaker = str(book.get("name") or "").strip()[:90]
            if not bookmaker:
                count("missing_bookmaker_name")
                continue
            for bet in book.get("bets") or []:
                if not isinstance(bet, dict):
                    count("malformed_market")
                    continue
                name = str(bet.get("name") or "").casefold().strip()
                name = re.sub(r"\s+", " ", name)
                raw_update=(bet.get("update") or book.get("update") or fixture.get("update"))
                update = (_time(raw_update) if raw_update else
                          _time(verified_provider_fetched_at))
                basis = ("bookmaker_quote_updated_at" if raw_update else
                         "api_response_fetched_at")
                if update is None:
                    count("missing_or_invalid_source_timestamp")
                    continue
                if update > now:
                    count("future_source_timestamp")
                    continue
                if now - update > timedelta(hours=max_age_hours):
                    count("stale_source_timestamp")
                    continue
                values = {}
                for entry in bet.get("values") or []:
                    if not isinstance(entry, dict):
                        count("malformed_price")
                        continue
                    label = str(entry.get("value") or "").strip().casefold()
                    price = _price(entry.get("odd"))
                    if price is not None:
                        values[label] = price
                matches = {}
                if name in ("match winner", "1x2") and all(
                        k in values for k in ("home", "draw", "away")):
                    matches["1x2"] = {"home": values["home"],
                                     "draw": values["draw"],
                                     "away": values["away"]}
                if name in ("both teams score", "both teams to score", "btts"):
                    if "yes" in values and "no" in values:
                        matches["btts_yes"] = {"yes": values["yes"], "no": values["no"]}
                # API-Football pre-match market commonly calls this "Goals Over/Under".
                if ("over/under" in name or name in ("goals over/under", "total goals")):
                    for level in ("2.5", "3.5"):
                        over = values.get("over " + level)
                        under = values.get("under " + level)
                        if over is not None and under is not None:
                            matches["over_" + level.replace(".", "_")] = {
                                "yes": over, "no": under}
                for kind, odds in matches.items():
                    fair = _normalize(odds)
                    if fair is not None:
                        count("accepted_bookmaker_market")
                        key = (kind, bookmaker)
                        record = {"market": kind, "bookmaker": bookmaker,
                                  "updated_at_utc": update.isoformat(),
                                  "observed_at_utc": now.isoformat(),
                                  "fair": fair,
                                  "timestamp_basis":basis,
                                  "source": "API-Football-prematch-odds"}
                        if key not in seen or record["updated_at_utc"] > seen[key]["updated_at_utc"]:
                            seen[key] = record
                    else:
                        count("invalid_price_or_overround")
                if not matches:
                    supported=(name in ("match winner","1x2","both teams score",
                               "both teams to score","btts","total goals") or
                               "over/under" in name)
                    count("incomplete_supported_market" if supported else "unsupported_market_type")
    grouped = {}
    for kind in _MARKETS:
        quotes = [q for q in seen.values() if q["market"] == kind]
        if quotes:
            quotes.sort(key=lambda q:(q["bookmaker"],q["updated_at_utc"]))
            yeskey = "home" if kind == "1x2" else "yes"
            result = {key: float(np.median([quote["fair"][key] for quote in quotes]))
                      for key in quotes[0]["fair"]}
            total = sum(result.values())
            grouped[kind] = {"probabilities":{k:round(v/total,7) for k,v in result.items()},
                             "n_bookmakers":len(quotes),
                             "quotes":quotes,
                             "last_source_update_utc":max(q["updated_at_utc"] for q in quotes),
                             "quote_timestamp_basis":[q["timestamp_basis"] for q in quotes]}
    return grouped


def attach_markets(snapshot, observed, raw):
    """Add a new hashed snapshot without changing an original saved packet."""
    if not isinstance(snapshot, dict) or snapshot.get("snapshot_sha256") is None:
        raise ValueError("Authenticated pre-match packet required")
    packet = {k:v for k,v in snapshot.items() if k!="snapshot_sha256"}
    if utc(observed)!=utc(packet["captured_at_utc"]):
        raise ValueError("Cannot attach later market quotes to an earlier packet")
    packet["multi_market"] = parse_markets(raw,observed_at=observed)
    packet["available_sources"] = dict(packet.get("available_sources") or {},
                                       time_checked_multi_market=bool(packet["multi_market"]))
    newhash = hashlib.sha256(json.dumps(packet,sort_keys=True,separators=(",",":"),
                                    allow_nan=False).encode()).hexdigest()
    packet["snapshot_sha256"] = newhash
    return packet


def xg_profile(fixture, packet, *, as_of=None, minimum=6):
    """Opponent-and-venue-context-adjusted genuine *past* xG with shrinkage.

    This requires enough provider-observed past fixtures; a 1-2 fixture
    sample is never used to adjust a score. Opponent concession strength
    requires a common league sampling frame, unavailable in a small packet.
    For this reason only direct venue differences and robust shrinkage
    are applied here, not a fictitious opponent coefficient.
    """
    if not valid_snapshot(packet,fixture,now=as_of):
        return {"ready":False,"reason":"source_identity_or_timestamp_invalid"}
    home = str(fixture["homeTeam"]["id"])
    away = str(fixture["awayTeam"]["id"])
    past = packet.get("verified_historical_fixture_xg") or []
    seen = set()
    sample = {home: [], away: []}
    for row in past:
        try:
            when = utc(row["kickoff_utc"])
            if when >= utc(packet["captured_at_utc"]) or (
                as_of is not None and when >= utc(as_of)):
                continue
            rid = str(row["fixture_id"])
            if rid in seen:
                continue
            seen.add(rid)
            h,a = str(row["home_team_id"]),str(row["away_team_id"])
            xh,xa=float(row["home_xg"]),float(row["away_xg"])
            if not all(math.isfinite(x) and 0<=x<=8 for x in (xh,xa)):
                continue
            for team,other in ((home,away),(away,home)):
                if team == h:
                    sample[team].append((when,xh,xa,"home"))
                elif team == a:
                    sample[team].append((when,xa,xh,"away"))
        except (KeyError,TypeError,ValueError):
            continue
    for values in sample.values():
        values.sort(key=lambda x:x[0],reverse=True)
    if any(len(sample[t])<minimum for t in (home,away)):
        return {"ready":False,"reason":"insufficient_genuine_pre_match_xg",
                "counts":{home:len(sample[home]),away:len(sample[away])},
                "min_per_team":minimum}
    def weighted(values,venue):
        entries=[r for r in values if venue is None or r[3]==venue][:12]
        if not entries:
            return None,0
        weights=[math.exp(-0.12*i) for i in range(len(entries))]
        sw=sum(weights)
        return {"scored":sum(r[1]*w for r,w in zip(entries,weights))/sw,
                "conceded":sum(r[2]*w for r,w in zip(entries,weights))/sw},len(entries)
    overall_h,nh=weighted(sample[home],None)
    overall_a,na=weighted(sample[away],None)
    venue_h,nvh=weighted(sample[home],"home")
    venue_a,nva=weighted(sample[away],"away")
    if nvh<3 or nva<3:
        return {"ready":False,"reason":"insufficient_verified_venue_xg",
                "home_venue_count":nvh,"away_venue_count":nva}
    def avg(x,y):
        return .65*x + .35*y
    attack_home=avg(venue_h["scored"],overall_h["scored"])
    concede_away=avg(venue_a["conceded"],overall_a["conceded"])
    attack_away=avg(venue_a["scored"],overall_a["scored"])
    concede_home=avg(venue_h["conceded"],overall_h["conceded"])
    # xG alone is not a reliable "true" future goal mean: shrink toward
    # the independent original model later, and keep use optional.
    rates=(.5*(attack_home+concede_away),.5*(attack_away+concede_home))
    return {"ready":True,"expected_goal_indicators":list(rates),
            "home_count":nh,"away_count":na,
            "home_venue_count":nvh,"away_venue_count":nva,
            "opponent_adjustment": "not_fitted_unavailable",
            "shrinkage_rule":"65% venue + 35% overall, then 50/50 attack/concession",
            "source":"genuine_API_Football_historical_fixture_xg"}


def lineup_status(fixture,packet,*,as_of=None):
    """Lineup information is useful but not an untrained goal adjustment."""
    if not valid_snapshot(packet,fixture,now=as_of):
        return {"ready":False,"reason":"invalid_evidence"}
    ids={str(fixture["homeTeam"]["id"]),str(fixture["awayTeam"]["id"])}
    valid=[]
    for row in packet.get("confirmed_starting_xi") or []:
        if str(row.get("team_id")) not in ids:
            continue
        names=row.get("starting_xi") or []
        if len(names)==11 and all(isinstance(n,str) and n for n in names):
            valid.append({"team_id":str(row["team_id"]),
                          "formation":row.get("formation"),
                          "starting_xi":names})
    return {"ready":len(valid)==2,"teams":valid,
            "numeric_player_coefficient_fitted":False,
            "missing_does_not_mean_healthy":True}


def calibrate_markets(matrix, markets, *, strength=.35, max_move=.15):
    """Entropic market tilt of full 90-minute grid with bounded influence.

    Uses at least two genuinely different markets; no defaults for missing
    prices. Fixed research-only strength, not an optimized calibration.
    """
    g=np.asarray(matrix,dtype=float).copy()
    if g.ndim!=2 or min(g.shape)<5 or not np.all(np.isfinite(g)) or np.any(g<0) or abs(g.sum()-1)>1e-5:
        raise ValueError("Valid, normalized full score grid required")
    if not isinstance(markets,dict):
        return {"ready":False,"reason":"no_market"}
    n=len([k for k in _MARKETS if k in markets])
    if n<2:
        return {"ready":False,"reason":"need_two_independent_market_types"}
    original=g/g.sum()
    ih=np.arange(g.shape[0])[:,None]
    ia=np.arange(g.shape[1])[None,:]
    masks={"home":ih>ia,"draw":ih==ia,"away":ih<ia,
           "over_2_5":ih+ia>=3,"over_3_5":ih+ia>=4,
           "btts_yes":(ih>=1)&(ia>=1)}
    selectors=[]
    one=markets.get("1x2",{}).get("probabilities") or {}
    if set(one)>= {"home","draw","away"}:
        for side in ("home","draw","away"):
            selectors.append((masks[side],float(one[side])))
    for kind in ("over_2_5","over_3_5","btts_yes"):
        value=markets.get(kind,{}).get("probabilities") or {}
        if "yes" in value:
            selectors.append((masks[kind],float(value["yes"])))
    if len(selectors)<2:
        return {"ready":False,"reason":"insufficient_valid_quote_probabilities"}
    g=np.maximum(g,1e-15)
    g/=g.sum()
    # Cap per-market log-odds adjustment and total probability change.
    for _ in range(4):
        for mask, target in selectors:
            if not (0.02<target<0.98):
                continue
            current=float(g[mask].sum())
            target=(1-strength)*current+strength*target
            if 0<current<1:
                g[mask]*=target/current
                g[~mask]*=(1-target)/(1-current)
                g/=g.sum()
    total_variation=float(np.abs(g-original).sum()/2)
    if total_variation>max_move:
        frac=max_move/total_variation
        g=(1-frac)*original+frac*g
        g/=g.sum()
    return {"ready":True,"matrix":g,
            "market_types":sorted({k for k in markets if k in _MARKETS}),
            "total_variation":float(np.abs(g-original).sum()/2),
            "research_only_no_promotion":True}


def combine_goal_evidence(baseline, xg, *, weight=.25):
    """Shrink verified past xG toward independent prior; no invented source."""
    matrix=np.asarray(baseline,dtype=float)
    if not xg.get("ready"):
        return {"ready":False,"reason":xg.get("reason","no_xg")}
    if matrix.ndim!=2 or abs(matrix.sum()-1)>1e-5:
        raise ValueError("Full grid required")
    home,away=xg["expected_goal_indicators"]
    if not all(math.isfinite(x) and .05<x<6 for x in (home,away)):
        return {"ready":False,"reason":"invalid_xg_means"}
    from v21_goal_model_comparison import _poisson
    h=np.arange(matrix.shape[0])[:,None]
    a=np.arange(matrix.shape[1])[None,:]
    base_h=float((matrix*h).sum())
    base_a=float((matrix*a).sum())
    corrected_h=.75*base_h+.25*home
    corrected_a=.75*base_a+.25*away
    other=np.outer(_poisson(corrected_h,matrix.shape[0]),
                   _poisson(corrected_a,matrix.shape[1]))
    result=(1-weight)*matrix+weight*other
    result/=result.sum()
    return {"ready":True,"matrix":result,
            "new_goal_indicators":[corrected_h,corrected_a],
            "research_only_no_promotion":True}
