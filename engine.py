"""Auditable venue-aware football forecasting. Results only; no invented injuries/xG."""
import math
from datetime import datetime, timezone
import numpy as np

MODEL_VERSION = "venue-form-ensemble-v2"

def finished(match):
    score = match.get('score') or {}
    full = score.get('fullTime') or {}
    return (match.get('status') == 'FINISHED' and
            type(full.get('home')) is int and type(full.get('away')) is int)

def _date(value):
    try:
        d = datetime.fromisoformat(str(value).replace('Z', '+00:00'))
        return d.replace(tzinfo=timezone.utc) if d.tzinfo is None else d
    except (TypeError, ValueError):
        return None

def _history(matches, before=None):
    cutoff = _date(before) if before else None
    unique = {}
    for m in matches:
        if not finished(m): continue
        d = _date(m.get('utcDate'))
        if not d or (cutoff and d >= cutoff): continue
        home = (m.get('homeTeam') or {}).get('id')
        away = (m.get('awayTeam') or {}).get('id')
        if home is None or away is None or home == away: continue
        key = (m.get('provider', 'football-data.org'), m.get('id') or (m.get('utcDate'), home, away))
        unique[key] = m
    return sorted(unique.values(), key=lambda m: m['utcDate'], reverse=True)

def _games(matches, team_id, venue=None, limit=10):
    out = []
    for m in matches:
        home = (m.get('homeTeam') or {}).get('id') == team_id
        away = (m.get('awayTeam') or {}).get('id') == team_id
        if not (home or away) or (venue == 'home' and not home) or (venue == 'away' and not away):
            continue
        score = m['score']['fullTime']
        gf,ga = (score['home'], score['away']) if home else (score['away'],score['home'])
        out.append({'date':m['utcDate'], 'gf':gf, 'ga':ga,
                    'result':'W' if gf>ga else 'D' if gf==ga else 'L',
                    'venue':'H' if home else 'A',
                    'opponent':(m.get('awayTeam') if home else m.get('homeTeam') or {}).get('name','Unknown')})
        if len(out) >= limit: break
    return out

def _stats(games):
    if not games:return None
    weights = np.array([2 ** (-i/5) for i in range(len(games))], dtype=float)
    gf = np.array([g['gf'] for g in games],dtype=float)
    ga = np.array([g['ga'] for g in games],dtype=float)
    return {'played':len(games),'scored':float(np.average(gf,weights=weights)),
            'conceded':float(np.average(ga,weights=weights)),
            'wins':sum(g['result']=='W' for g in games),
            'draws':sum(g['result']=='D' for g in games),
            'losses':sum(g['result']=='L' for g in games),
            'points_per_game':sum(3 if g['result']=='W' else 1 if g['result']=='D' else 0 for g in games)/len(games)}

def team_stats(matches, team_id, venue=None, limit=10):
    return _stats(_games(_history(matches),team_id,venue,limit))

def league_baselines(matches):
    scores=[m['score']['fullTime'] for m in _history(matches)]
    if len(scores)<12:return 1.45,1.15
    return (max(.55,float(np.mean([s['home'] for s in scores]))),
            max(.55,float(np.mean([s['away'] for s in scores]))))

def poisson(k,lam):
    return math.exp(-lam)*lam**k/math.factorial(k)

def _grid(h,a,max_goals=10,rho=0.0):
    matrix=np.array([[poisson(i,h)*poisson(j,a) for j in range(max_goals+1)]
                     for i in range(max_goals+1)],dtype=float)
    # Dixon-Coles low-score adjustment. Zero by default until league-specific
    # walk-forward evaluation supports fitting a nonzero rho.
    if rho:
        factors={(0,0):1-h*a*rho,(0,1):1+h*rho,
                 (1,0):1+a*rho,(1,1):1-rho}
        for (i,j),factor in factors.items():
            matrix[i,j]*=max(0.0,factor)
    matrix/=matrix.sum()
    return matrix

def _h2h(matches,home_id,away_id,limit=10):
    meetings=[]
    for m in matches:
        h=(m.get('homeTeam') or {}).get('id')
        a=(m.get('awayTeam') or {}).get('id')
        if {h,a}!={home_id,away_id}:continue
        score=m['score']['fullTime']
        gh,ga=(score['home'],score['away']) if h==home_id else (score['away'],score['home'])
        meetings.append({'date':m['utcDate'],'home_goals':gh,'away_goals':ga,
                         'result':'H' if gh>ga else 'D' if gh==ga else 'A'})
        if len(meetings)>=limit:break
    return meetings

def predict(fixture,history,max_goals=10):
    home_id=(fixture.get('homeTeam') or {}).get('id')
    away_id=(fixture.get('awayTeam') or {}).get('id')
    if home_id is None or away_id is None or home_id==away_id:return None
    # Strictly exclude results on/after fixture kickoff, even for live fixtures.
    past=_history(history,fixture.get('utcDate'))
    hg=_games(past,home_id,limit=10);ag=_games(past,away_id,limit=10)
    hv=_games(past,home_id,'home',limit=10);av=_games(past,away_id,'away',limit=10)
    if min(len(hg),len(ag))<5:return None
    hs,as_=_stats(hg),_stats(ag)
    hvs,avs=_stats(hv),_stats(av)
    base_h,base_a=league_baselines(past)
    # Venue-specific rates shrink to overall form, then to league averages.
    def shrink(venue,overall,field,prior=5):
        if not venue:return overall[field]
        n=venue['played']
        return (n*venue[field]+prior*overall[field])/(n+prior)
    home_for=shrink(hvs,hs,'scored')
    home_against=shrink(hvs,hs,'conceded')
    away_for=shrink(avs,as_,'scored')
    away_against=shrink(avs,as_,'conceded')
    def clamp(v,lo=.25,hi=3.8):return float(np.clip(v,lo,hi))
    # Two complementary estimators: multiplicative strengths and arithmetic
    # attack/defence goals. Equal blend is conservative, not a tuned optimum.
    lam_h_strength=base_h*(.65+.35*home_for/base_h)*(.65+.35*away_against/base_h)
    lam_a_strength=base_a*(.65+.35*away_for/base_a)*(.65+.35*home_against/base_a)
    lam_h_simple=.5*(home_for+away_against)
    lam_a_simple=.5*(away_for+home_against)
    lam_h=clamp(.55*lam_h_strength+.45*lam_h_simple)
    lam_a=clamp(.55*lam_a_strength+.45*lam_a_simple)
    # Recent form / venue / H2H are visible evidence, not arbitrarily
    # applied extra multipliers that could double-count goals.
    matrix=_grid(lam_h,lam_a,max_goals)
    scores=sorted(((i,j,float(matrix[i,j])) for i in range(max_goals+1)
                   for j in range(max_goals+1)),key=lambda x:x[2],reverse=True)
    h2h=_h2h(past,home_id,away_id)
    return {'home_xg':lam_h,'away_xg':lam_a,
            'home_win':float(np.tril(matrix,-1).sum()),
            'draw':float(np.trace(matrix)),
            'away_win':float(np.triu(matrix,1).sum()),
            'scores':scores[:10],'matrix':matrix,
            'home_stats':hs,'away_stats':as_,
            'home_venue_stats':hvs,'away_venue_stats':avs,
            'home_last10':hg,'away_last10':ag,
            'home_home10':hv,'away_away10':av,
            'h2h_last10':h2h,
            'history_count':len(past),'model_version':MODEL_VERSION,
            'methods':{'strength':(round(lam_h_strength,3),round(lam_a_strength,3)),
                       'attack_defence':(round(lam_h_simple,3),round(lam_a_simple,3))},
            'data_quality':'moderate' if min(len(hg),len(ag),len(hv),len(av))>=8 else 'limited'}
