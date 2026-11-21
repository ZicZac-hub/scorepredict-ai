"""Matched five-double backtest, using a fixed pair per Sydney calendar day.

Historical result inputs are not prospective proof. Assumes earlier-day final
results were available before the pair cutoff; same-day results never enter
training. Do not tune on the reported evaluation period.
"""
import argparse
from collections import defaultdict
from datetime import timedelta
import json
import math
import numpy as np
from zoneinfo import ZoneInfo
from engine import _date, _history, predict
from dynamic_dixon_coles import five_doubles

VARIANTS = ('baseline','dixon_coles','dynamic_dixon_coles')
SYDNEY = ZoneInfo('Australia/Sydney')

def evaluate(matches, start=None):
    # Reject negative regulation scores before both selection and training.
    ordered = sorted((m for m in _history(matches)
                      if min(m['score']['fullTime']['home'],m['score']['fullTime']['away'])>=0),
                     key=lambda m:(_date(m['utcDate']),str(m.get('id'))))
    days=defaultdict(list)
    for m in ordered:
        if min(m['score']['fullTime']['home'],m['score']['fullTime']['away'])<0:
            continue
        days[_date(m['utcDate']).astimezone(SYDNEY).date()].append(m)
    report={v:dict(pairs=0,hits=0,coverage=0.,brier=0.,score_log_loss=0.) for v in VARIANTS}
    offered=0
    unavailable={v:0 for v in VARIANTS}
    rows=[]
    for day, games in sorted(days.items()):
        if start and day.isoformat()<start:
            continue
        # Fixed selection independent of outcome and model probabilities.
        # Distinct teams within the same provider prevent shared-team pairs.
        pair=None
        for i,first in enumerate(games):
            first_teams={first['homeTeam']['id'],first['awayTeam']['id']}
            for second in games[i+1:]:
                if first.get('provider','football-data.org') != second.get('provider','football-data.org'):
                    continue
                if first_teams.isdisjoint({second['homeTeam']['id'],second['awayTeam']['id']}):
                    pair=(first,second)
                    break
            if pair:
                break
        if pair is None:
            continue
        offered+=1
        # Conservative additional 24-hour buffer for completion availability.
        cutoff=min(_date(m['utcDate']) for m in pair)-timedelta(days=1)
        past=[m for m in ordered if _date(m['utcDate'])<cutoff
              and _date(m['utcDate']).astimezone(SYDNEY).date()<day]
        forecasts={v:[predict(m,past,variant=v) for m in pair] for v in VARIANTS}
        missing=[v for v,group in forecasts.items() if any(f is None for f in group)]
        for v in missing:
            unavailable[v]+=1
        if missing:
            continue
        trial={'sydney_date':day.isoformat(),'fixture_ids':[m.get('id') for m in pair],
               'provider':pair[0].get('provider','football-data.org'),'models':{}}
        actual=tuple((m['score']['fullTime']['home'],m['score']['fullTime']['away']) for m in pair)
        trial['actual_scores']=[list(s) for s in actual]
        for v, group in forecasts.items():
            a,b=(f['matrix'] for f in group)
            picks=five_doubles(a,b)
            hit=int(any((s,t)==actual for _,s,t in picks))
            coverage=sum(p for p,_,_ in picks)
            row=report[v]
            row['pairs']+=1
            row['hits']+=hit
            row['coverage']+=coverage
            row['brier']+=(coverage-hit)**2
            trial['models'][v]={'hit':hit,'coverage':float(coverage),'brier':float((coverage-hit)**2)}
            diagnostics=trial['models'][v]
            probabilities=[]
            ranks=[]
            for matrix,(h,g) in zip((a,b),actual):
                inside=h<matrix.shape[0] and g<matrix.shape[1]
                p=float(matrix[h,g]) if inside else 0.
                probabilities.append(p)
                ranks.append(1+int(np.count_nonzero(matrix>p)) if inside else None)
                row['score_log_loss']-=math.log(max(p,1e-12))
            diagnostics['actual_leg_probabilities']=probabilities
            diagnostics['actual_leg_rank_minimum']=ranks
            diagnostics['actual_pair_mean_score_log_loss']=-sum(math.log(max(p,1e-12)) for p in probabilities)/2
            diagnostics['actual_joint_probability']=probabilities[0]*probabilities[1]
            diagnostics['actual_joint_rank_minimum']=(1+int(np.count_nonzero(
                np.outer(a.ravel(),b.ravel())>diagnostics['actual_joint_probability']))
                if all(rank is not None for rank in ranks) else None)
            diagnostics['five_event_brier_note']='Models may select different events; descriptive only.'
        rows.append(trial)
    for row in report.values():
        n=row['pairs']
        row['hit_rate']=row['hits']/n if n else None
        for key in ('coverage','brier','score_log_loss'):
            row[key]=row[key]/(2*n if key=='score_log_loss' else n) if n else None
    matched=report['baseline']['pairs']
    comparisons={v:{'baseline_only_hits':sum(t['models']['baseline']['hit'] and not t['models'][v]['hit'] for t in rows),
                    'challenger_only_hits':sum(t['models'][v]['hit'] and not t['models']['baseline']['hit'] for t in rows),
                    'mean_paired_score_log_loss_advantage':sum(t['models']['baseline']['actual_pair_mean_score_log_loss']-t['models'][v]['actual_pair_mean_score_log_loss'] for t in rows)/matched if matched else None}
                 for v in VARIANTS if v!='baseline'}
    return {'paired_comparisons':comparisons,'offered_pairs':offered,'matched_pairs':matched,
            'excluded_unmatched_pairs':offered-matched,
            'matched_fraction':matched/offered if offered else None,
            'unavailable_pairs_by_model':unavailable,'matched_trials':rows,
            'models':report,'note':'Historical matched-pair evaluation only; no production promotion. Joint coverage assumes independent fixtures.'}

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('matches_json')
    parser.add_argument('--start',required=True,help='Predeclared evaluation start YYYY-MM-DD')
    args=parser.parse_args()
    with open(args.matches_json,encoding='utf-8') as f:
        matches=json.load(f)
    print(json.dumps(evaluate(matches,args.start),indent=2))
