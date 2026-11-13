"""Regularized, time-decayed opponent ratings and training-fitted low-score dependence.

This is an independently implemented statistical model. References are public
football modelling papers, not proprietary outputs from any AI provider.
"""
import math
import numpy as np
from engine import _date, _competition_history, _grid


def fit(fixture, history, regularization=6.0, half_life_days=240.0, fit_dependence=True):
    """Fit only completed, same-provider/competition results before kickoff.

    Alternating Newton steps maximize a weighted Poisson likelihood with a
    ridge penalty on log attack/defence. Home/away league intercepts are fitted
    jointly with the opponent ratings. No evaluated score enters fitting.
    """
    games = _competition_history(fixture, history)
    kickoff = _date(fixture.get('utcDate'))
    if kickoff is None or len(games) < 30:
        return None
    home_id = (fixture.get('homeTeam') or {}).get('id')
    away_id = (fixture.get('awayTeam') or {}).get('id')
    teams = sorted({m[side]['id'] for m in games for side in ('homeTeam', 'awayTeam')}, key=str)
    index = {team: i for i, team in enumerate(teams)}
    if home_id not in index or away_id not in index:
        return None
    home = np.array([index[m['homeTeam']['id']] for m in games], dtype=int)
    away = np.array([index[m['awayTeam']['id']] for m in games], dtype=int)
    counts = np.bincount(np.r_[home, away], minlength=len(teams))
    if min(counts[index[home_id]], counts[index[away_id]]) < 5:
        return None
    gh = np.array([m['score']['fullTime']['home'] for m in games], dtype=float)
    ga = np.array([m['score']['fullTime']['away'] for m in games], dtype=float)
    if np.any(gh < 0) or np.any(ga < 0):
        return None
    ages = np.array([(kickoff-_date(m['utcDate'])).total_seconds()/86400 for m in games])
    weight = 2.0**(-ages/float(half_life_days))
    # Poisson forecasts require positive rates; .05 is a numerical floor, not
    # an upper bound on strong teams or a rule choosing their scoreline.
    intercept_h = math.log(max(.05, float(np.average(gh, weights=weight))))
    intercept_a = math.log(max(.05, float(np.average(ga, weights=weight))))
    attack, defence = np.zeros(len(teams)), np.zeros(len(teams))
    penalty = float(regularization)
    for _ in range(35):
        old = np.r_[attack.copy(), defence.copy(), intercept_h, intercept_a]
        h_mu = np.exp(np.clip(intercept_h+attack[home]+defence[away], -8, 5))
        a_mu = np.exp(np.clip(intercept_a+attack[away]+defence[home], -8, 5))
        gradient = (np.bincount(home, weights=weight*(gh-h_mu), minlength=len(teams))
                    +np.bincount(away, weights=weight*(ga-a_mu), minlength=len(teams))
                    -penalty*attack)
        curvature = (np.bincount(home, weights=weight*h_mu, minlength=len(teams))
                     +np.bincount(away, weights=weight*a_mu, minlength=len(teams))+penalty)
        attack += np.clip(gradient/np.maximum(curvature, 1e-9), -.5, .5)
        h_mu = np.exp(np.clip(intercept_h+attack[home]+defence[away], -8, 5))
        a_mu = np.exp(np.clip(intercept_a+attack[away]+defence[home], -8, 5))
        gradient = (np.bincount(away, weights=weight*(gh-h_mu), minlength=len(teams))
                    +np.bincount(home, weights=weight*(ga-a_mu), minlength=len(teams))
                    -penalty*defence)
        curvature = (np.bincount(away, weights=weight*h_mu, minlength=len(teams))
                     +np.bincount(home, weights=weight*a_mu, minlength=len(teams))+penalty)
        defence += np.clip(gradient/np.maximum(curvature, 1e-9), -.5, .5)
        # Given ratings, each home/away intercept has an exact likelihood update.
        intercept_h = math.log(max(.05, float(np.sum(weight*gh)/np.sum(weight*np.exp(attack[home]+defence[away])))))
        intercept_a = math.log(max(.05, float(np.sum(weight*ga)/np.sum(weight*np.exp(attack[away]+defence[home])))))
        if np.max(np.abs(np.r_[attack, defence, intercept_h, intercept_a]-old)) < 1e-6:
            break
    fitted_h = np.exp(np.clip(intercept_h+attack[home]+defence[away], -8, 5))
    fitted_a = np.exp(np.clip(intercept_a+attack[away]+defence[home], -8, 5))
    rho = 0.0
    if fit_dependence:
        # Fit rho by weighted historical likelihood, regularized toward zero.
        # Bounds enforce positive Dixon-Coles factors for EVERY training rate.
        lower = max(-.2, float(np.max(-.99/np.maximum(fitted_h, fitted_a))))
        upper = min(.2, float(np.min(.99/(fitted_h*fitted_a))))
        candidates = np.linspace(lower, upper, 81)
        likelihood = -20*candidates*candidates
        for goals, factor in (((0,0), -fitted_h*fitted_a), ((0,1), fitted_h), ((1,0), fitted_a)):
            mask = (gh==goals[0])&(ga==goals[1])
            likelihood += np.sum(weight[mask, None]*np.log(1+factor[mask, None]*candidates[None, :]), axis=0)
        likelihood += weight[(gh==1)&(ga==1)].sum()*np.log(1-candidates)
        rho = float(candidates[int(np.argmax(likelihood))])
    lam_h = float(np.exp(intercept_h+attack[index[home_id]]+defence[index[away_id]]))
    lam_a = float(np.exp(intercept_a+attack[index[away_id]]+defence[index[home_id]]))
    # A new matchup can have rates outside the training extrema. Keep every
    # correction positive without clipping or moving the chosen scoreline.
    rho = min(max(rho, -.99/max(lam_h, lam_a)), .99/(lam_h*lam_a))
    return {'home_xg': max(.05, lam_h), 'away_xg': max(.05, lam_a), 'rho': rho,
            'league_matches': len(games), 'team_matches': [int(counts[index[home_id]]), int(counts[index[away_id]])],
            'effective_league_matches': float(weight.sum()**2/np.square(weight).sum()),
            'regularization': penalty, 'half_life_days': half_life_days,
            'fitted_home_advantage': float(np.exp(intercept_h-intercept_a)),
            'fit_cutoff_utc': kickoff.isoformat(),
            'last_training_kickoff_utc': max(m['utcDate'] for m in games)}


def score_matrix(estimate, max_goals=10):
    return _grid(estimate['home_xg'], estimate['away_xg'], max_goals, rho=estimate['rho'])
