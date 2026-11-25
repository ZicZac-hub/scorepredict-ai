"""Offline European sparse-history research prototype; not a live forecast.

Domestic venue/opponent-relative strengths are a conservative sensitivity
adjustment to the V37 European control. Cross-league absolute strength is not
identified by domestic scores, and this hypothesis requires independent testing.
No network access, production changes, or retroactive forecasts.
"""
from collections import defaultdict
from datetime import timedelta
import math

import numpy as np
from history_pipeline import build_evidence, season_year, _date

VERSION = "cross-competition-strength-research-v1"
DOMESTIC = {"PL": 39, "PD": 140, "BL1": 78, "SA": 135, "FL1": 61,
            "DED": 88, "PPL": 94, "ELC": 40}
MIN_LEAGUE = 50
MIN_TEAM = 8
MIN_VENUE = 3


def _team_id(row, side):
    return str(row[side]["id"])


def _relative_strength(rows, team, venue, cutoff):
    # Recency-weighted, regularized attack/defence ratings within one league.
    observations = []
    for row in rows:
        h, a = _team_id(row, "homeTeam"), _team_id(row, "awayTeam")
        score = row["score"]["fullTime"]
        age = (cutoff - _date(row["utcDate"])).total_seconds() / 86400
        observations.append((h, a, score["home"], score["away"], 2 ** (-age / 180)))
    games = [r for r in observations if team in r[:2]]
    venue_games = [r for r in games if r[0 if venue == "home" else 1] == team]
    if len(observations) < MIN_LEAGUE or len(games) < MIN_TEAM or len(venue_games) < MIN_VENUE:
        return None
    total = sum(r[4] for r in observations)
    base = [max(.3, sum(r[i] * r[4] for r in observations) / total) for i in (2, 3)]
    teams = {t for row in observations for t in row[:2]}
    attack = {(t, v): 1. for t in teams for v in (0, 1)}
    defence = attack.copy()
    for _ in range(6):
        an, ad, dn, dd = (defaultdict(float) for _ in range(4))
        for h, a, gh, ga, w in observations:
            for t, o, v, g in ((h, a, 0, gh), (a, h, 1, ga)):
                an[t, v] += w * g
                ad[t, v] += w * base[v] * defence[o, 1-v]
                dn[o, 1-v] += w * g
                dd[o, 1-v] += w * base[v] * attack[t, v]
        attack = {k: min(2.5, max(.4, (an[k]+8)/(ad[k]+8))) for k in attack}
        defence = {k: min(2.5, max(.4, (dn[k]+8)/(dd[k]+8))) for k in defence}
    v = 0 if venue == "home" else 1
    return {"attack": attack[team, v], "defence": defence[team, v],
            "team_matches": len(games), "venue_matches": len(venue_games),
            "league_matches": len(observations)}


def _domestic_context(fixture, team, venue, history_cache, cutoff):
    candidates = []
    season = season_year(fixture)
    for code, league in DOMESTIC.items():
        rows = []
        current_membership = False
        for year in (season, season-1):
            entry = history_cache.get(("API-Football", str(league), year))
            if not entry:
                continue
            raw, metadata, error = entry
            if error or not metadata.get("freshness_known") or metadata.get("stale") is not False:
                continue
            proxy = {**fixture, "competition": {"id": league, "code": code, "season": year}}
            evidence = build_evidence(proxy, raw, cutoff)
            # Restrict to the explicitly requested season to avoid duplicated rows.
            qualified = [r for r in evidence["history"] if season_year(r) == year]
            if year == season:
                current_membership = any(team in (_team_id(r, "homeTeam"), _team_id(r, "awayTeam"))
                                         for r in qualified)
            rows.extend(qualified)
        if current_membership:
            candidates.append((code, rows))
    if len(candidates) != 1:
        return None
    code, rows = candidates[0]
    # Duplicate provider fixture identifiers across seasons fail closed.
    ids = [str(r.get("id")) for r in rows]
    if any(r.get("id") is None for r in rows) or len(ids) != len(set(ids)):
        return None
    context = _relative_strength(rows, team, venue, cutoff)
    return {"competition": code, **context} if context else None


def research(fixture, european_grid, history_cache, *, as_of):
    """Candidate grid only when both domestic contexts are independently sufficient."""
    if fixture.get("provider") != "API-Football" or fixture.get("competition", {}).get("code") not in ("CL", "EL"):
        return {"status": "abstained", "reason": "not_supported_european_fixture"}
    cutoff = _date(as_of)
    if cutoff >= _date(fixture["utcDate"]):
        return {"status": "abstained", "reason": "not_pre_kickoff"}
    home = _domestic_context(fixture, _team_id(fixture, "homeTeam"), "home", history_cache, cutoff)
    away = _domestic_context(fixture, _team_id(fixture, "awayTeam"), "away", history_cache, cutoff)
    if not home or not away:
        return {"status": "abstained", "reason": "missing_or_ambiguous_domestic_context",
                "home_context_available": home is not None, "away_context_available": away is not None}
    p = np.asarray(european_grid, dtype=float)
    if p.ndim != 2 or not p.size or not np.isfinite(p).all() or (p < 0).any() or not math.isclose(float(p.sum()), 1., abs_tol=1e-8):
        raise ValueError("invalid_european_control")
    # Modest fixed tilt, not fitted to observed outcomes: preserve all support.
    h, a = np.indices(p.shape)
    hm = float(np.clip((home["attack"] * away["defence"]) ** .25, .85, 1.15))
    am = float(np.clip((away["attack"] * home["defence"]) ** .25, .85, 1.15))
    q = p * np.exp(np.clip(h * math.log(hm) + a * math.log(am), -3., 3.))
    q /= q.sum()
    return {"status": "research_candidate", "version": VERSION,
            "home": home, "away": away, "home_tilt": hm, "away_tilt": am,
            "control": p.tolist(), "challenger": q.tolist(),
            "absolute_cross_league_strength_estimated": False,
            "production_promoted": False}
