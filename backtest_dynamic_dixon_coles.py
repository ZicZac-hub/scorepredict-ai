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
from zoneinfo import ZoneInfo
from engine import _date, _history, predict
from dynamic_dixon_coles import five_doubles

VARIANTS = ('baseline','dixon_coles','dynamic_dixon_coles')
SYDNEY = ZoneInfo('Australia/Sydney')

def wilson(hits, n):
    if not n:
        return None
    z=1.959963984540054
    p=hits/n
    den=1+z*z/n
    mid=(p+z*z/(2*n))/den
    radius=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return [max(0.,mid-radius),min(1.,mid+radius)]

def paired_comparison(wins, losses):
    # Exact two-sided McNemar/binomial test on discordant pairs.
    n=wins+losses
    if not n:
        return 1.0
    k=min(wins,losses)
    logs=[math.lgamma(n+1)-math.lgamma(i+1)-math.lgamma(n-i+1)-n*math.log(2)
          for i in range(k+1)]
    return min(1.,2*sum(math.exp(x) for x in logs))

def evaluate(matches, start=None):
    ordered = sorted(_history(matches),key=lambda m:(_date(m['utcDate']),str(m.get('id'))))
    days=defaultdict(list)
    for m in ordered:
        if min(m['score']['fullTime']['home'],m['score']['fullTime']['away'])<0:
            continue
        days[_date(m['utcDate']).astimezone(SYDNEY).date()].append(m)
    report={v:dict(pairs=0,hits=0,coverage=0.,brier=0.,score_log_loss=0.) for v in VARIANTS}
    offered=0
    comparisons={v:{'candidate_only':0,'reference_only':0,'both_hit':0,'both_miss':0} for v in VARIANTS[:-1]}
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
        if any(f is None for group in forecasts.values() for f in group):
            continue
        actual=tuple((m['score']['fullTime']['home'],m['score']['fullTime']['away']) for m in pair)
        pair_hits={}
        for v, group in forecasts.items():
            a,b=(f['matrix'] for f in group)
            picks=five_doubles(a,b)
            hit=int(any((s,t)==actual for _,s,t in picks))
            pair_hits[v]=hit
            coverage=sum(p for p,_,_ in picks)
            row=report[v]
            row['pairs']+=1
            row['hits']+=hit
            row['coverage']+=coverage
            row['brier']+=(coverage-hit)**2
            for matrix,(h,g) in zip((a,b),actual):
                p=float(matrix[h,g]) if h<matrix.shape[0] and g<matrix.shape[1] else 0.
                row['score_log_loss']-=math.log(max(p,1e-12))
        for v, comparison in comparisons.items():
            c,r=pair_hits['dynamic_dixon_coles'],pair_hits[v]
            key='both_hit' if c and r else 'both_miss' if not c and not r else 'candidate_only' if c else 'reference_only'
            comparison[key]+=1
    for row in report.values():
        n=row['pairs']
        row['hit_rate']=row['hits']/n if n else None
        row['hit_rate_wilson_95']=wilson(row['hits'],n)
        for key in ('coverage','brier','score_log_loss'):
            row[key]=row[key]/(2*n if key=='score_log_loss' else n) if n else None
    n=report['baseline']['pairs']
    for v,c in comparisons.items():
        c['hit_rate_difference']=(c['candidate_only']-c['reference_only'])/n if n else None
        c['exact_p_value']=paired_comparison(c['candidate_only'],c['reference_only']) if n else None
    return {'offered_pairs':offered,'matched_pairs':n,'excluded_pairs':offered-n,
            'paired_comparisons':comparisons,
            'models':report,'note':'Historical matched-pair evaluation only; no production promotion. Joint coverage assumes independent fixtures; interval and paired test assume independent days and are descriptive when teams recur; not a promotion gate.'}

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('matches_json')
    parser.add_argument('--start',required=True,help='Predeclared evaluation start YYYY-MM-DD')
    args=parser.parse_args()
    with open(args.matches_json,encoding='utf-8') as f:
        matches=json.load(f)
    print(json.dumps(evaluate(matches,args.start),indent=2))
