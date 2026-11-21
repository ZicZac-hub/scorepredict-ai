"""Experimental time-varying, opponent-adjusted GENUINE-xG challenger.

Uses only hash-verified API-Football pre-match packets. Numerical lineup
effects are deliberately zero until player-impact coefficients have independent
historical validation. Does not modify original forecasts or the live model.
"""
import math
from datetime import timedelta

import numpy as np

from dynamic_dixon_coles import fit_forecast, _groups, _safe_rho
from engine import _competition_history, _date, _games, _grid, _stats, league_baselines
from v21_provider_evidence import valid_snapshot

VERSION = "verified-xg-opponent-v1-shadow"
XG_WEIGHT = 0.25  # predeclared research candidate, NOT accuracy-fitted


def _num(value):
    return type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 15


def _identity(value):
    return str(value) if value is not None and type(value) is not bool else ""


def _xi_status(snapshot, fixture):
    ids = {_identity((fixture.get(side+"Team") or {}).get("id"))
           for side in ("home","away")}
    accepted = set()
    for item in snapshot.get("confirmed_starting_xi") or []:
        if not isinstance(item, dict) or item.get("provider_reported") is not True:
            continue
        tid = _identity(item.get("team_id"))
        names = item.get("starting_xi")
        if tid not in ids or not isinstance(names,list) or len(names)!=11:
            continue
        if all(isinstance(s,str) and s.strip() for s in names):
            if len({s.strip().casefold() for s in names})==11:
                accepted.add(tid)
    return "both_confirmed" if len(accepted)==2 else "partial" if accepted else "unavailable"


def _rows(fixture, history, snapshot, as_of):
    kickoff = _date(fixture.get("utcDate"))
    captured = _date(snapshot.get("captured_at_utc"))
    cutoff = _date(as_of) if as_of else captured
    if not kickoff or not captured or not cutoff or not captured<=cutoff<kickoff:
        return None, [], "invalid_cutoff"
    if fixture.get("provider") != "API-Football" or not valid_snapshot(snapshot, fixture, now=cutoff):
        return None, [], "unverified_snapshot"
    league = _competition_history(fixture, history)
    if not (fixture.get("competition") or {}).get("id") and not (fixture.get("competition") or {}).get("code"):
        return None, [], "unidentified_league"
    by_id = {}
    for m in league:
        by_id.setdefault(_identity(m.get("id")), []).append(m)
    grouped = {}
    for item in snapshot.get("verified_historical_fixture_xg") or []:
        if not isinstance(item,dict):
            continue
        fid = _identity(item.get("fixture_id"))
        if fid:
            grouped.setdefault(fid,[]).append(item)
    valid = []
    for fid, records in grouped.items():
        if len(records)!=1 or len(by_id.get(fid,[]))!=1:
            continue
        item, match = records[0], by_id[fid][0]
        d = _date(item.get("kickoff_utc"))
        md = _date(match.get("utcDate"))
        if (item.get("type")!="api_football_reported_expected_goals"
                or not d or d!=md or d+timedelta(hours=3)>captured
                or d>=cutoff or not _num(item.get("home_xg"))
                or not _num(item.get("away_xg"))):
            continue
        if ([_identity(item.get(side+"_team_id")) for side in ("home","away")]
                != [_identity((match.get(side+"Team") or {}).get("id")) for side in ("home","away")]):
            continue
        valid.append((d,match,float(item["home_xg"]),float(item["away_xg"])))
    return league, valid, "verified"


def _qualities(team, rows, league, kickoff):
    samples=[]
    for d, match, home_xg, away_xg in rows:
        ishome=_identity(match["homeTeam"]["id"])==team
        isaway=_identity(match["awayTeam"]["id"])==team
        if not (ishome or isaway):
            continue
        opponent=match["awayTeam"]["id"] if ishome else match["homeTeam"]["id"]
        # Result availability buffer: no final scores from this or simultaneous games.
        previous=[m for m in league if _date(m["utcDate"]) + timedelta(hours=3)<d]
        if len(previous)<12:
            continue
        bh,ba=league_baselines(previous)
        ref_for,ref_against=(bh,ba) if ishome else (ba,bh)
        opponent_games=_games(previous,opponent,limit=20)
        opp=_stats(opponent_games, as_of=d) if opponent_games else None
        n=opp["effective_played"] if opp else 0.
        # Regularized opponent quality; absent opponent data => neutral 1.
        defense=((n*opp["conceded"]+6*ref_for)/(n+6)/ref_for) if opp else 1.
        attack=((n*opp["scored"]+6*ref_against)/(n+6)/ref_against) if opp else 1.
        xf=home_xg if ishome else away_xg
        xa=away_xg if ishome else home_xg
        weight=2**(-max(0.,(kickoff-d).total_seconds()/86400)/180.)
        samples.append((weight,xf/ref_for/float(np.clip(defense,.6,1.8)),
                        xa/ref_against/float(np.clip(attack,.6,1.8))))
    return samples


def _factor(rows, index, prior=6.):
    # League-relative ratio shrunk to neutral 1; historical xG cannot become
    # an exact-score certainty or an assumed league-wide xG baseline.
    weight=sum(row[0] for row in rows)
    if weight<=0:
        return 1.
    value=(prior+sum(row[0]*row[index] for row in rows))/(prior+weight)
    return float(np.clip(value,.55,1.8))


def predict_shadow(fixture, history, snapshot, *, as_of=None, max_goals=10):
    """Return None for insufficient verified source coverage (never fake xG)."""
    if not isinstance(snapshot,dict):
        return None
    league, rows, status=_rows(fixture,history,snapshot,as_of)
    if status!="verified":
        return None
    baseline=fit_forecast(fixture,league,max_goals=max_goals)
    if baseline is None:
        return None
    kickoff=_date(fixture["utcDate"])
    hid=_identity(fixture["homeTeam"]["id"])
    aid=_identity(fixture["awayTeam"]["id"])
    home=_qualities(hid,rows,league,kickoff)
    away=_qualities(aid,rows,league,kickoff)
    if min(len(home),len(away))<3 or min(sum(x[0] for x in v) for v in (home,away))<1.:
        return None
    # Geometric matchup quality: attack of one team × shot-quality defence
    # of the other, conditional on VERIFIED historical xG only.
    bh,ba=league_baselines(league)
    xh=bh*math.sqrt(_factor(home,1)*_factor(away,2))
    xa=ba*math.sqrt(_factor(away,1)*_factor(home,2))
    h0,a0=baseline["home_xg"],baseline["away_xg"]
    h=float(np.clip(math.exp((1-XG_WEIGHT)*math.log(h0)+XG_WEIGHT*math.log(xh)),.15,12.))
    a=float(np.clip(math.exp((1-XG_WEIGHT)*math.log(a0)+XG_WEIGHT*math.log(xa)),.15,12.))
    matrix=_grid(h,a,max_goals,rho=_safe_rho(h,a,baseline["rho"]))
    matrix*=np.asarray(baseline["goal_band_factors"])[_groups(matrix)]
    matrix/=matrix.sum()
    lineup=_xi_status(snapshot,fixture)
    return {
        "model_version":VERSION,"matrix":matrix,"home_xg":h,"away_xg":a,
        "home_win":float(np.tril(matrix,-1).sum()),
        "draw":float(np.trace(matrix)),
        "away_win":float(np.triu(matrix,1).sum()),
        "verified_xg_matches":{"home":len(home),"away":len(away)},
        "historical_opponent_adjustment":True,
        "lineup_status":lineup,"lineup_numeric_weight":0.0,
        "xg_weight":XG_WEIGHT,"base_model_version":baseline["model_version"],
        "source_snapshot_sha256":snapshot["snapshot_sha256"],
        "note":"Research-only. Verified historical xG, opponent adjustment and decay. Lineups are evidence categories, not invented player-value coefficients."
    }
