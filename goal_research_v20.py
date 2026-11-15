"""V20 research-driven *experimental* scoring from timestamped match evidence.

This challenger does not overwrite the validated statistical baseline. It compares
recent overall and venue goals, opposition concessions, and full score grids. No
strength label forces a result; score probabilities can move in either direction.
Missing provider data is an explicit abstention. No extra API calls.
"""
import math
from datetime import datetime, timedelta, timezone

import numpy as np


def _time(value):
    if isinstance(value, datetime):
        dt = value
    else:
        try:
            dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except (ValueError, TypeError):
            return None
    return dt.astimezone(timezone.utc) if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _summary(items):
    if not items:
        return {"matches": 0, "scored": None, "conceded": None,
                "scored_2plus": None, "conceded_2plus": None,
                "match_over_2_5": None, "last_scores": []}
    n = len(items)
    return {
        "matches": n,
        "scored": round(sum(a for a, b, _, _ in items) / n, 3),
        "conceded": round(sum(b for a, b, _, _ in items) / n, 3),
        "scored_2plus": round(sum(a >= 2 for a, b, _, _ in items) / n, 3),
        "conceded_2plus": round(sum(b >= 2 for a, b, _, _ in items) / n, 3),
        "match_over_2_5": round(sum(a + b >= 3 for a, b, _, _ in items) / n, 3),
        "last_scores": [{"gf": a, "ga": b, "venue": venue, "date_utc": date}
                        for a, b, venue, date in items[:10]]
    }


def goal_profile(fixture, history, now):
    """Strictly pre-match observations for two provider-scoped team identities."""
    kickoff = _time(fixture.get("utcDate"))
    clock = _time(now)
    if kickoff is None or clock is None:
        return {"available": False, "reason": "missing_time"}
    cutoff = min(clock, kickoff)
    home = (fixture.get("homeTeam") or {}).get("id")
    away = (fixture.get("awayTeam") or {}).get("id")
    if home is None or away is None or str(home) == str(away):
        return {"available": False, "reason": "missing_team_identity"}
    provider = str(fixture.get("provider") or "football-data.org")
    league = fixture.get("competition") or {}
    records = []
    seen = set()
    for item in history or []:
        if not isinstance(item, dict) or item.get("status") != "FINISHED":
            continue
        if str(item.get("provider") or "football-data.org") != provider:
            continue
        source_league = item.get("competition") or {}
        # Check competition when both records provide a scoped identifier.
        if league.get("id") is not None and source_league.get("id") is not None:
            if str(league["id"]) != str(source_league["id"]):
                continue
        elif league.get("code") and source_league.get("code"):
            if league["code"] != source_league["code"]:
                continue
        when = _time(item.get("utcDate"))
        if when is None or when >= cutoff:
            continue
        available = _time(item.get("finishedAt")) or when + timedelta(hours=3)
        if available > cutoff:
            continue
        teams = (item.get("homeTeam") or {}, item.get("awayTeam") or {})
        h, a = teams[0].get("id"), teams[1].get("id")
        if h is None or a is None or str(h) == str(a):
            continue
        duration = (item.get("score") or {}).get("duration")
        goals = ((item.get("score") or {}).get("regularTime") if duration in
                 ("EXTRA_TIME", "PENALTY_SHOOTOUT") else
                 (item.get("score") or {}).get("fullTime")) or {}
        gh, ga = goals.get("home"), goals.get("away")
        if type(gh) is not int or type(ga) is not int or min(gh, ga) < 0:
            continue
        key = (provider, str(item.get("id") or ""), when.isoformat(), str(h), str(a))
        if key in seen:
            continue
        seen.add(key)
        records.append((when, str(h), str(a), gh, ga))
    records.sort(reverse=True, key=lambda x: x[0])
    h10, a10, hv10, av10, meetings = [], [], [], [], []
    for when, h, a, gh, ga in records:
        iso = when.isoformat()
        if h == str(home):
            if len(h10) < 10: h10.append((gh, ga, "home", iso))
            if len(hv10) < 10: hv10.append((gh, ga, "home", iso))
        elif a == str(home):
            if len(h10) < 10: h10.append((ga, gh, "away", iso))
        if h == str(away):
            if len(a10) < 10: a10.append((gh, ga, "home", iso))
        elif a == str(away):
            if len(a10) < 10: a10.append((ga, gh, "away", iso))
            if len(av10) < 10: av10.append((ga, gh, "away", iso))
        if ((h == str(home) and a == str(away)) or
                (h == str(away) and a == str(home))):
            if len(meetings) < 6:
                meetings.append({"home_goals": gh if h == str(home) else ga,
                                 "away_goals": ga if h == str(home) else gh,
                                 "date_utc": iso})
    return {
        "available": True,
        "verified_historical_fixtures": len(records),
        "home_last10": _summary(h10), "away_last10": _summary(a10),
        "home_at_home_last10": _summary(hv10),
        "away_away_last10": _summary(av10),
        "h2h_last6": meetings,
        "h2h_is_context_not_fitted": True,
        "results_available_before_forecast_only": True,
        "no_claim_of_verified_shot_xg_or_lineup_coefficients": True
    }


def _poisson(mean, n):
    values = np.zeros(n, dtype=float)
    values[0] = math.exp(-mean)
    for k in range(1, n):
        values[k] = values[k-1] * mean / k
    values[-1] += max(0.0, 1.0 - float(values.sum()))  # retain upper tail
    return values / values.sum()


def goal_form_challenger(row, history, now):
    """Blend a complete baseline score grid with conservative matchup-goal form.

    Not a proven model: the unchanged original and this challenger must be
    backtested on the same pre-match source evidence before promotion.
    """
    fixture = row.get("fixture") or {}
    profile = goal_profile(fixture, history, now)
    result = row.get("result") or {}
    if not profile.get("available"):
        return {"available": False, "reason": profile.get("reason"), "profile": profile}
    sections = [profile[k] for k in
                ("home_last10", "away_last10", "home_at_home_last10", "away_away_last10")]
    if any(s["matches"] < (8 if i < 2 else 4) for i, s in enumerate(sections)):
        return {"available": False, "reason": "Need at least 8 prior matches per team and 4 at the relevant venue.",
                "profile": profile}
    original = result.get("matrix")
    if original is None:
        return {"available": False, "reason": "No complete original score matrix; archived top-ten scores cannot substitute for a probability distribution.",
                "profile": profile}
    try:
        matrix = np.asarray(original, dtype=float)
        if (matrix.ndim != 2 or min(matrix.shape) < 5 or
                np.any(~np.isfinite(matrix)) or np.any(matrix < 0) or
                abs(float(matrix.sum()) - 1.0) > 1e-5):
            raise ValueError("unusable grid")
    except (TypeError, ValueError):
        return {"available": False, "reason": "Original full score distribution is invalid.", "profile": profile}
    x = np.arange(matrix.shape[0], dtype=float)
    y = np.arange(matrix.shape[1], dtype=float)
    original_h = float((matrix * x[:, None]).sum())
    original_a = float((matrix * y[None, :]).sum())
    home_overall, away_overall, home_venue, away_venue = sections
    h_attack = .6 * home_venue["scored"] + .4 * home_overall["scored"]
    h_defense = .6 * home_venue["conceded"] + .4 * home_overall["conceded"]
    a_attack = .6 * away_venue["scored"] + .4 * away_overall["scored"]
    a_defense = .6 * away_venue["conceded"] + .4 * away_overall["conceded"]
    raw_h = (h_attack + a_defense) / 2
    raw_a = (a_attack + h_defense) / 2
    # Guard against noisy 8-10-match samples. No handpicked big score or
    # strong-team label is ever inserted into the final shortlist.
    new_h = max(.25, min(5.0, .60 * original_h + .40 * raw_h))
    new_a = max(.25, min(5.0, .60 * original_a + .40 * raw_a))
    new_h = max(original_h * .70, min(original_h * 1.35, new_h))
    new_a = max(original_a * .70, min(original_a * 1.35, new_a))
    form_grid = np.outer(_poisson(new_h, matrix.shape[0]),
                         _poisson(new_a, matrix.shape[1]))
    tested_weight = .35  # fixed shadow hypothesis, NOT trained or validated
    candidate = (1-tested_weight) * matrix + tested_weight * form_grid
    candidate /= candidate.sum()
    ids = np.argsort(-candidate.ravel(), kind="stable")[:10]
    scores = [[int(i // candidate.shape[1]), int(i % candidate.shape[1]),
               float(candidate.ravel()[i])] for i in ids]
    def high_score_mass(grid):
        return float(sum(grid[i, j] for i in range(grid.shape[0])
                         for j in range(grid.shape[1]) if i+j >= 4))
    return {
        "available": True, "profile": profile,
        "scores": scores, "matrix": candidate,
        "original_expected_goals": [original_h, original_a],
        "research_expected_goals": [float((candidate*x[:, None]).sum()),
                                   float((candidate*y[None, :]).sum())],
        "historical_matchup_goal_rates": [raw_h, raw_a],
        "four_plus_goal_probability_original": high_score_mass(matrix),
        "four_plus_goal_probability_research": high_score_mass(candidate),
        "model": "v20_goal_form_shadow_not_validated",
        "proven_accuracy_improvement": False
    }
