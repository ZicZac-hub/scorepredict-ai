"""Transparent, lightweight football score probability model."""
import math
from collections import defaultdict
from datetime import datetime, timezone

import numpy as np


def finished(match):
    score = match.get('score') or {}
    full = score.get('fullTime') or {}
    return (match.get('status') == 'FINISHED' and isinstance(full.get('home'), int)
            and isinstance(full.get('away'), int))


def team_stats(matches, team_id, venue=None, limit=20):
    recent = []
    for m in matches:
        if not finished(m):
            continue
        home = (m.get('homeTeam') or {}).get('id')
        away = (m.get('awayTeam') or {}).get('id')
        if team_id not in (home, away):
            continue
        is_home = team_id == home
        if venue == 'home' and not is_home or venue == 'away' and is_home:
            continue
        score = m['score']['fullTime']
        recent.append((m.get('utcDate', ''), score['home'] if is_home else score['away'],
                       score['away'] if is_home else score['home']))
    recent.sort(reverse=True)
    recent = recent[:limit]
    if not recent:
        return None
    # Most recent matches get modestly higher weight.
    weights = np.array([0.96 ** i for i in range(len(recent))], dtype=float)
    gf = np.array([r[1] for r in recent], dtype=float)
    ga = np.array([r[2] for r in recent], dtype=float)
    return {'played': len(recent), 'scored': float(np.average(gf, weights=weights)),
            'conceded': float(np.average(ga, weights=weights)),
            'wins': sum(x[1] > x[2] for x in recent),
            'draws': sum(x[1] == x[2] for x in recent),
            'losses': sum(x[1] < x[2] for x in recent)}


def league_baselines(matches):
    scores = [m['score']['fullTime'] for m in matches if finished(m)]
    if not scores:
        return 1.45, 1.15
    return (max(0.3, float(np.mean([s['home'] for s in scores]))),
            max(0.3, float(np.mean([s['away'] for s in scores]))))


def poisson(k, lam):
    return math.exp(-lam) * lam ** k / math.factorial(k)


def predict(fixture, history, max_goals=10):
    home_id = fixture['homeTeam']['id']
    away_id = fixture['awayTeam']['id']
    h_all = team_stats(history, home_id)
    a_all = team_stats(history, away_id)
    h_venue = team_stats(history, home_id, 'home')
    a_venue = team_stats(history, away_id, 'away')
    if not h_all or not a_all or min(h_all['played'], a_all['played']) < 3:
        return None
    base_h, base_a = league_baselines(history)
    # Shrink venue-specific small samples toward each team's overall averages.
    def blend(venue, overall, field):
        if not venue:
            return overall[field]
        n = venue['played']
        return (n * venue[field] + 5 * overall[field]) / (n + 5)
    h_attack = blend(h_venue, h_all, 'scored') / max(base_h, 0.3)
    h_defense = blend(h_venue, h_all, 'conceded') / max(base_a, 0.3)
    a_attack = blend(a_venue, a_all, 'scored') / max(base_a, 0.3)
    a_defense = blend(a_venue, a_all, 'conceded') / max(base_h, 0.3)
    # Regularization prevents extreme forecasts from small samples.
    lam_h = float(np.clip(base_h * (0.7 + 0.3*h_attack) * (0.7 + 0.3*a_defense), 0.25, 4.0))
    lam_a = float(np.clip(base_a * (0.7 + 0.3*a_attack) * (0.7 + 0.3*h_defense), 0.25, 4.0))
    matrix = np.array([[poisson(i, lam_h)*poisson(j, lam_a)
                        for j in range(max_goals+1)] for i in range(max_goals+1)])
    # Normalize tiny truncated tail to 1.
    matrix /= matrix.sum()
    scores = sorted(((i, j, float(matrix[i,j])) for i in range(max_goals+1)
                     for j in range(max_goals+1)), key=lambda x: x[2], reverse=True)
    return {'home_xg': lam_h, 'away_xg': lam_a,
            'home_win': float(np.tril(matrix, -1).sum()),
            'draw': float(np.trace(matrix)),
            'away_win': float(np.triu(matrix, 1).sum()),
            'scores': scores[:10], 'matrix': matrix,
            'home_stats': h_all, 'away_stats': a_all,
            'history_count': sum(finished(m) for m in history)}
