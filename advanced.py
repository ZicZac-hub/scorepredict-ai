"""Auditable advanced analytics. No generative model is represented as clairvoyance."""
from datetime import datetime, timezone
from itertools import combinations
import math
import numpy as np
from engine import predict, finished


def double_chance(p):
    return {'1X': p['home_win'] + p['draw'], 'X2': p['away_win'] + p['draw'], '12': p['home_win'] + p['away_win']}


def context_review(fixture, p, history):
    """Human-style reasoning grounded ONLY in available match results."""
    h, a = p['home_stats'], p['away_stats']
    findings = []
    if p['home_xg'] > p['away_xg'] + .45:
        findings.append('Home attack/away defence combination favours the home team.')
    elif p['away_xg'] > p['home_xg'] + .45:
        findings.append('Away attack/home defence combination favours the away team.')
    else:
        findings.append('Expected scoring rates are relatively close; the result is uncertain.')
    if h['played'] < 8 or a['played'] < 8:
        findings.append('Small recent-match sample: forecasts may be unstable.')
    if abs(p['home_win'] - p['away_win']) < .08:
        findings.append('Neither side has a clear modelled win advantage.')
    findings.append('Injuries, confirmed lineups, travel and live xG are not verified in this version.')
    return findings


def ranked_double_chance(rows, min_prob=0.0):
    picks = []
    for r in rows:
        p = r.get('result')
        if not p: continue
        for market, probability in double_chance(p).items():
            if probability >= min_prob:
                picks.append({'fixture_id':r['fixture'].get('id'), 'fixture':r['fixture'], 'league':r['league'],
                              'kickoff':r['kickoff'], 'selection':market, 'probability':float(probability),
                              'expected_goals':(p['home_xg'],p['away_xg'])})
    picks.sort(key=lambda x:(-x['probability'],x['kickoff']))
    chosen, used = [], set()
    for p in picks:
        if p['fixture_id'] in used: continue
        chosen.append(p); used.add(p['fixture_id'])
        if len(chosen)==5: break
    return chosen


def five_two_leg_doubles(rows):
    """Five illustrative two-match double-chance combinations, distinct matches per pair.
    Independence is assumed; correlation can invalidate product probabilities.
    """
    candidates = ranked_double_chance(rows)
    pairs=[]
    for x,y in combinations(candidates,2):
        if x['fixture_id'] == y['fixture_id']: continue
        pairs.append({'leg1':x,'leg2':y,'joint_probability_independent':x['probability']*y['probability']})
    return sorted(pairs,key=lambda r:-r['joint_probability_independent'])[:5]


def backtest(matches, min_prior=35, max_tests=120, variant='regularized'):
    """Walk-forward test: only completed fixtures STRICTLY before kickoff train each forecast.
    Skips games on the same timestamp, avoiding contemporaneous data leakage.
    """
    ordered = sorted([m for m in matches if finished(m) and m.get('utcDate')], key=lambda m:m['utcDate'])
    records=[]
    for idx,m in enumerate(ordered):
        if idx < min_prior: continue
        prior=[x for x in ordered[:idx] if x['utcDate'] < m['utcDate']]
        p=predict(m,prior,variant=variant)
        if p is None: continue
        actual=m['score']['fullTime']; hg,ag=actual['home'],actual['away']
        outcome='H' if hg>ag else ('D' if hg==ag else 'A')
        probs={'H':p['home_win'],'D':p['draw'],'A':p['away_win']}
        top=p['scores'][0]
        actual_score_prob=float(p['matrix'][hg,ag]) if hg<p['matrix'].shape[0] and ag<p['matrix'].shape[1] else 1e-9
        records.append({'date':m['utcDate'],'home':m['homeTeam'].get('name','Home'),
                        'away':m['awayTeam'].get('name','Away'),'actual':f'{hg}-{ag}',
                        'predicted':f'{top[0]}-{top[1]}','exact_hit':int((hg,ag)==(top[0],top[1])),
                        'outcome_hit':int(max(probs,key=probs.get)==outcome),
                        'brier':sum((v-int(k==outcome))**2 for k,v in probs.items())/3,
                        'score_log_loss':-math.log(max(actual_score_prob,1e-12)),
                        'home_goals_error':abs(hg-p['home_xg']), 'away_goals_error':abs(ag-p['away_xg']),
                        'home_win_prob':p['home_win'],'draw_prob':p['draw'],'away_win_prob':p['away_win']})
    return records[-max_tests:]


def summarize_backtest(records):
    if not records:return None
    n=len(records)
    return {'matches':n,'exact_score_accuracy':sum(r['exact_hit'] for r in records)/n,
            'result_accuracy':sum(r['outcome_hit'] for r in records)/n,
            'mean_brier':sum(r['brier'] for r in records)/n,
            'mean_score_log_loss':sum(r['score_log_loss'] for r in records)/n,
            'mean_absolute_goal_error':sum(r['home_goals_error']+r['away_goals_error'] for r in records)/(2*n)}
