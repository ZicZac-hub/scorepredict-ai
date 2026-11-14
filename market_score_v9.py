"""ZicZac v9: auditable pre-match market-informed exact-score distribution.

This is an original KL/minimum-change projection of the existing v8 score
matrix onto independent provider market marginals. It is not ScoreGPT's
proprietary algorithm and it does not call an external LLM.

The historical v8 distribution remains intact. Provider prices are fetched
on demand only, never silently inside the scheduled prediction collector.
No measured v9 accuracy is claimed without timestamped historical odds.
"""
from datetime import datetime, timezone, timedelta
import math
import numpy as np
from research_ensemble_v8 import predict as research_predict

VERSION = "market-informed-v9-experimental"
MAX_MARKET_AGE_HOURS = 12
NEAR_FRESHEST_HOURS = 3
MARKET_WEIGHT = 0.45  # fixed, unvalidated; prevents fully replacing historical forecasts


def _time(value):
    try:
        result = datetime.fromisoformat(str(value).replace("Z","+00:00"))
        return result.replace(tzinfo=timezone.utc) if result.tzinfo is None else result.astimezone(timezone.utc)
    except (ValueError,TypeError,OverflowError):
        return None


def _margin_free(odds):
    if not odds or any(not math.isfinite(x) or x <= 1.0 or x>1000 for x in odds):
        return None
    q = np.reciprocal(np.asarray(odds,dtype=float))
    overround = float(q.sum())
    if overround<0.98 or overround>1.35:
        return None
    return tuple(float(x) for x in (q/q.sum())), overround


def _market_bets(bookmaker):
    winner, total = None, None
    for bet in bookmaker.get("bets") or []:
        name = str(bet.get("name") or "").lower()
        values = {}
        for row in bet.get("values") or []:
            label = str(row.get("value") or "").strip().lower()
            try: price=float(row.get("odd"))
            except (TypeError,ValueError): continue
            if not math.isfinite(price) or price<=1 or price>1000: continue
            values[label] = price
        if "match winner" in name or name in ("1x2","full time result"):
            keys = [next((k for k in items if k in values),None) for items in
                    (("home","1","home win"),("draw","x","tie"),("away","2","away win"))]
            if all(keys):
                candidate=_margin_free([values[key] for key in keys])
                if candidate: winner=candidate
        elif "goals over/under" in name or name in ("over/under","total goals"):
            over = next((key for key in ("over 2.5","over 2.5 goals","over") if key in values),None)
            under = next((key for key in ("under 2.5","under 2.5 goals","under") if key in values),None)
            if over and under:
                candidate=_margin_free([values[under],values[over]])
                if candidate: total=candidate
    return winner,total


def select_market_response(payload,fixture,as_of=None):
    """Select one timestamped bookmaker for regulation-time 1X2 and O/U2.5.

    Reject stale, post-as-of, or wrong-fixture records; don't combine book
    prices from different snapshots or vendors. Metadata is audit-friendly.
    """
    cutoff = _time(as_of) if as_of is not None else datetime.now(timezone.utc)
    kickoff = _time(fixture.get("utcDate"))
    if cutoff is None or kickoff is None or cutoff>=kickoff:
        return None
    candidates = []
    for item in payload if isinstance(payload,list) else []:
        if str((item.get("fixture") or {}).get("id")) != str(fixture.get("id")):
            continue
        stamped = _time(item.get("update"))
        if not stamped or stamped>cutoff or cutoff-stamped>timedelta(hours=MAX_MARKET_AGE_HOURS):
            continue
        for bookmaker in item.get("bookmakers") or []:
            winner, total = _market_bets(bookmaker)
            if not winner:
                continue
            name=str(bookmaker.get("name") or "Provider bookmaker")
            priority = ("pinnacle" in name.lower(),"bet365" in name.lower(), bool(total))
            candidates.append((priority,stamped,name,winner,total))
    if not candidates:
        return None
    # Never select a 10-hour-old quote just because it includes totals
    # when another genuine, supported bookmaker updated more recently.
    # Compare market completeness/brand only within three hours of the
    # freshest verified snapshot for the SAME fixture.
    freshest=max(v[1] for v in candidates)
    near=[v for v in candidates if freshest-v[1]<=timedelta(hours=NEAR_FRESHEST_HOURS)]
    priority,stamp,name,win,total = max(
        near,key=lambda v:(v[0][2],v[0][0],v[1],v[0][1]))
    return {"bookmaker":name,"snapshot_utc":stamp.isoformat(),"winner":win[0],
            "winner_overround":win[1],"totals":total[0] if total else None,
            "totals_overround":total[1] if total else None,"fixture_id":fixture.get("id"),
            "settlement":"regulation_90_minutes"}


def _project(matrix,market,steps=50):
    """KL projection onto two sets of disjoint partitions of score cells."""
    adjusted=np.asarray(matrix,dtype=float).copy()
    adjusted /= adjusted.sum()
    hh=np.arange(adjusted.shape[0])[:,None]
    aa=np.arange(adjusted.shape[1])[None,:]
    groups=[hh>aa,hh==aa,hh<aa]
    outcome=np.asarray(market["winner"],dtype=float)
    outcome/=outcome.sum()
    under=(hh+aa)<=2
    over=~under
    totals=market["totals"]
    for _ in range(steps):
        for mask,prob in zip(groups,outcome):
            cell_sum=float(adjusted[mask].sum())
            if cell_sum<=1e-12:
                raise ValueError("Forecast grid lacks positive market-support cells")
            adjusted[mask] *= prob/cell_sum
        if totals is not None:
            for mask,prob in zip((under,over),totals):
                cell_sum=float(adjusted[mask].sum())
                if cell_sum<=1e-12:
                    raise ValueError("Forecast grid lacks positive total-goals cells")
                adjusted[mask] *= prob/cell_sum
    adjusted /= adjusted.sum()
    return adjusted


def _summarize(matrix,model,market=None):
    idx=np.argsort(-matrix.ravel(),kind="stable")[:10]
    scores=[(int(i//matrix.shape[1]),int(i%matrix.shape[1]),float(matrix.ravel()[i])) for i in idx]
    hg=np.arange(matrix.shape[0])[:,None]
    ag=np.arange(matrix.shape[1])[None,:]
    result=dict(model)
    result.update({"matrix":matrix,"scores":scores,
                   "home_xg":float((matrix*hg).sum()),"away_xg":float((matrix*ag).sum()),
                   "home_win":float(np.tril(matrix,-1).sum()),
                   "draw":float(np.trace(matrix)),
                   "away_win":float(np.triu(matrix,1).sum()),
                   "model_version":VERSION,
                   "market_used":bool(market),
                   "market_details":market or {},
                   "historical_model_version":model.get("model_version"),
                   "release_status":"experimental_market_model_not_historically_validated",
                   "llm_provider_calls":0,
                   "limitations":"Market-informed distribution is unvalidated for exact scores. No external AI models are called; no guaranteed score."})
    return result


def predict(fixture,history,league_history=None,market=None,as_of=None,max_goals=10):
    """No odds means exactly the existing v8 distribution (no silent fetch)."""
    base=research_predict(fixture,history,league_history=league_history,
                          as_of=as_of,max_goals=max_goals)
    if base is None:
        return None
    if market is None:
        return _summarize(np.asarray(base["matrix"],dtype=float),base)
    if str(market.get("fixture_id")) != str(fixture.get("id")):
        raise ValueError("Market data belongs to a different fixture")
    stamp=_time(market.get("snapshot_utc"))
    reference=_time(as_of) if as_of is not None else datetime.now(timezone.utc)
    kickoff=_time(fixture.get("utcDate"))
    if not stamp or not reference or not kickoff or stamp>reference or stamp>=kickoff or reference>=kickoff:
        raise ValueError("Market snapshot is not a verified pre-kickoff observation")
    if reference-stamp>timedelta(hours=MAX_MARKET_AGE_HOURS):
        raise ValueError("Market data is too stale for v9")
    adjusted=_project(base["matrix"],market)
    mixed=(1-MARKET_WEIGHT)*base["matrix"]+MARKET_WEIGHT*adjusted
    mixed /= mixed.sum()
    return _summarize(mixed,base,market)
