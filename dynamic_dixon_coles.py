"""Experimental, chronological league-specific Dixon–Coles challenger.

No production promotion. Fit weighted regularized attack/defence on earlier
fixtures only; learn rho and total-goal calibration on a later training block.
"""
import math
import numpy as np
from engine import _competition_history, _date, _grid

HALF_LIFE = 180.0
MIN_HISTORY = 100
PRIOR = 8.0

def _observations(fixture, history):
    cutoff = _date(fixture.get('utcDate'))
    competition = fixture.get('competition') or {}
    # This challenger is league-specific: never pool unknown leagues.
    if cutoff is None or (competition.get('id') is None and not competition.get('code')):
        return []
    rows = []
    for m in _competition_history(fixture, history):
        d = _date(m.get('utcDate'))
        h, a = m['homeTeam']['id'], m['awayTeam']['id']
        gh, ga = m['score']['fullTime']['home'], m['score']['fullTime']['away']
        if min(gh, ga) < 0:
            continue
        w = 2 ** (-(cutoff-d).total_seconds()/86400/HALF_LIFE)
        rows.append((d, h, a, gh, ga, w))
    return sorted(rows, key=lambda r:r[0])

def _fit(rows):
    if not rows:
        return None
    teams = {t:i for i,t in enumerate(dict.fromkeys(t for r in rows for t in r[1:3]))}
    h = np.array([teams[r[1]] for r in rows])
    a = np.array([teams[r[2]] for r in rows])
    gh, ga, w = (np.array([r[i] for r in rows], dtype=float) for i in (3,4,5))
    if w.sum() < 1e-8:
        return None
    bh = max(.25, (w@gh+PRIOR*1.45)/(w.sum()+PRIOR))
    ba = max(.25, (w@ga+PRIOR*1.15)/(w.sum()+PRIOR))
    n = len(teams)
    # Pooled attack/concession ratings and league-specific venue intercepts.
    # Coordinate updates use regularization toward neutral unit strengths.
    att, defence = np.ones(n), np.ones(n)
    count = lambda ids, vals: np.bincount(ids, weights=vals, minlength=n)
    for _ in range(30):
        att = np.clip((count(h,w*gh)+count(a,w*ga)+PRIOR)/
                      (count(h,w*bh*defence[a])+count(a,w*ba*defence[h])+PRIOR), .2, 5.)
        defence = np.clip((count(a,w*gh)+count(h,w*ga)+PRIOR)/
                          (count(a,w*bh*att[h])+count(h,w*ba*att[a])+PRIOR), .2, 5.)
    # Shrunk team-specific venue deviations, estimated against opponent strength.
    ah = (count(h,w*gh)+PRIOR)/(count(h,w*bh*att[h]*defence[a])+PRIOR)
    aa = (count(a,w*ga)+PRIOR)/(count(a,w*ba*att[a]*defence[h])+PRIOR)
    dh = (count(h,w*ga)+PRIOR)/(count(h,w*ba*att[a]*defence[h])+PRIOR)
    da = (count(a,w*gh)+PRIOR)/(count(a,w*bh*att[h]*defence[a])+PRIOR)
    return teams,bh,ba,att,defence,ah,aa,dh,da

def _rates(model,h,a,*,venue_deviations=True):
    teams,bh,ba,att,defence,ah,aa,dh,da = model
    if h not in teams or a not in teams:
        return None
    i,j = teams[h],teams[a]
    vh=ah[i]*da[j] if venue_deviations else 1.
    va=aa[j]*dh[i] if venue_deviations else 1.
    return (float(np.clip(bh*att[i]*defence[j]*vh,.15,12)),
            float(np.clip(ba*att[j]*defence[i]*va,.15,12)))

def _safe_rho(h,a,rho):
    return float(np.clip(rho, max(-.2,-.999/h,-.999/a), min(.2,.999/(h*a),.999)))

def _tau(h,a,gh,ga,rho):
    r = _safe_rho(h,a,rho)
    return {(0,0):1-h*a*r,(0,1):1+h*r,(1,0):1+a*r,(1,1):1-r}.get((gh,ga),1.)

def _groups(matrix):
    i,j = np.indices(matrix.shape)
    return np.where(i+j<=1,0,np.where(i+j>=4,2,1))

def _calibrate(rows, model, *, venue_deviations=True):
    holdout = []
    for _,h,a,gh,ga,w in rows:
        rates = _rates(model,h,a,venue_deviations=venue_deviations)
        if rates is not None:
            holdout.append((*rates,gh,ga,w))
    if len(holdout)<20 or sum(row[-1] for row in holdout)<10:
        return 0., np.ones(3), 0
    # Penalized likelihood; all observations are from the calibration block.
    def objective(r):
        return sum(w*math.log(_tau(h,a,gh,ga,r)) for h,a,gh,ga,w in holdout)-8*r*r
    rho = float(max(np.linspace(-.2,.2,81), key=objective))
    observed, expected = np.zeros(3), np.zeros(3)
    for h,a,gh,ga,w in holdout:
        matrix = _grid(h,a,rho=_safe_rho(h,a,rho))
        observed[0 if gh+ga<=1 else 2 if gh+ga>=4 else 1]+=w
        expected += w*np.bincount(_groups(matrix).ravel(),weights=matrix.ravel(),minlength=3)
    factors = np.clip((observed+PRIOR)/(expected+PRIOR),.5,2.)
    return rho,factors,len(holdout)

def fit_forecast(fixture,history,max_goals=10,*,venue_deviations=True,goal_band_calibration=True):
    rows = _observations(fixture,history)
    if len(rows)<MIN_HISTORY or sum(r[-1] for r in rows)<30:
        return None
    h,a = fixture['homeTeam']['id'],fixture['awayTeam']['id']
    # Five stale appearances are not five recent-equivalent observations.
    if min(sum(t in r[1:3] for r in rows) for t in (h,a))<5:
        return None
    team_weight = [sum(r[-1] for r in rows if t in r[1:3]) for t in (h,a)]
    if min(team_weight)<3.0:
        return None
    cutoff = rows[int(.75*len(rows))][0]
    training = [r for r in rows if r[0]<cutoff]
    calibration = [r for r in rows if r[0]>=cutoff]
    early, full = _fit(training), _fit(rows)
    if early is None or full is None:
        return None
    rho,factors,n = _calibrate(calibration,early,venue_deviations=venue_deviations)
    if not goal_band_calibration:
        factors=np.ones(3)
    rates = _rates(full,h,a,venue_deviations=venue_deviations)
    if rates is None:
        return None
    h_rate,a_rate = rates
    rho = _safe_rho(h_rate,a_rate,rho)
    matrix = _grid(h_rate,a_rate,max_goals,rho=rho)
    matrix *= factors[_groups(matrix)]
    matrix /= matrix.sum()
    return dict(matrix=matrix,home_xg=h_rate,away_xg=a_rate,rho=rho,
                calibration_matches=n,goal_band_factors=factors.tolist(),
                league_history_count=len(rows),model_version=('dynamic-dixon-coles-v1-experimental' if venue_deviations and goal_band_calibration else 'dynamic-dixon-coles-ablation-research-only'))

def five_doubles(first,second):
    """Five distinct joint score events maximizing coverage under independence."""
    def top(matrix):
        return sorted(((float(p),i,j) for (i,j),p in np.ndenumerate(matrix)),
                      key=lambda r:(-r[0],r[1],r[2]))[:5]
    combinations = [(p*q,(h,a),(x,y)) for p,h,a in top(first) for q,x,y in top(second)]
    return sorted(combinations,key=lambda r:(-r[0],r[1],r[2]))[:5]
