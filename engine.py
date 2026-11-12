"""Auditable venue-aware football forecasting. Results only; no invented injuries/xG."""
import math
from datetime import datetime, timezone
import numpy as np

MODEL_VERSION = "venue-form-ensemble-v5-robust-effective-sample"
LIVE_MODEL_VERSION = "venue-form-ensemble-v5-unregularized-live"

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

def _stats(games, as_of=None):
    if not games:return None
    # Weight by elapsed calendar days, not by game index.
    reference = _date(as_of) if as_of else _date(games[0]['date'])
    ages = [max(0.0, (reference - _date(g['date'])).total_seconds()/86400.0) for g in games]
    weights = np.array([2 ** (-age/180.0) for age in ages], dtype=float)
    # Extreme one-off scorelines can dominate a small 5-10 game sample.
    # Robustify the goal-rate estimator while keeping actual results intact
    # for win/draw/loss reporting and historical accuracy evaluation.
    gf = np.array([min(float(g['gf']),5.0) for g in games],dtype=float)
    ga = np.array([min(float(g['ga']),5.0) for g in games],dtype=float)
    effective_n=float(weights.sum()**2/np.square(weights).sum())
    return {'played':len(games),'effective_played':round(effective_n,2),
            'scored':float(np.average(gf,weights=weights)),
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

def predict(fixture,history,max_goals=10,variant='baseline'):
    home_id=(fixture.get('homeTeam') or {}).get('id')
    away_id=(fixture.get('awayTeam') or {}).get('id')
    if home_id is None or away_id is None or home_id==away_id:return None
    # Strictly exclude results on/after fixture kickoff, even for live fixtures.
    past=_history(history,fixture.get('utcDate'))
    hg=_games(past,home_id,limit=10);ag=_games(past,away_id,limit=10)
    hv=_games(past,home_id,'home',limit=10);av=_games(past,away_id,'away',limit=10)
    if min(len(hg),len(ag))<5:return None
    hs,as_=_stats(hg,fixture.get('utcDate')),_stats(ag,fixture.get('utcDate'))
    hvs,avs=_stats(hv,fixture.get('utcDate')),_stats(av,fixture.get('utcDate'))
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
    # Experimental venue-adjusted arithmetic estimator; retain existing
    # baseline for production until the chronological holdout is reviewed.
    if variant == 'venue_arithmetic':
        lam_h=clamp(.70*lam_h_simple+.30*lam_h_strength)
        lam_a=clamp(.70*lam_a_simple+.30*lam_a_strength)
    # Experimental team-strength estimator: shrink team-specific attack and
    # defensive rates toward league venue averages using a fixed prior of
    # five matches. Geometric matchup strengths preserve relative differences
    # between teams while avoiding extreme ratios on thin samples.
    # This is evaluated on historical holdout only; baseline is unchanged.
    if variant == 'team_strength':
        prior = 5.0
        def stabilized(rate, reference, effective):
            return (effective * rate + prior * reference) / (effective + prior)
        hf = stabilized(home_for, base_h, hs['effective_played'])
        aa = stabilized(away_against, base_h, as_['effective_played'])
        af = stabilized(away_for, base_a, as_['effective_played'])
        ha = stabilized(home_against, base_a, hs['effective_played'])
        lam_h = clamp(base_h * math.sqrt((hf / base_h) * (aa / base_h)))
        lam_a = clamp(base_a * math.sqrt((af / base_a) * (ha / base_a)))
    # Experimental opponent-adjusted estimator. For each recent match,
    # normalize goals scored by the opponent's historical defensive rate,
    # and goals conceded by its attacking rate. Opponent ratings are computed
    # using only matches strictly BEFORE that historical match (no leakage).
    # Shrink the adjustment towards 1 to avoid noisy small samples.
    if variant == 'opponent_adjusted':
        chronological = list(reversed(past))
        def adjusted(team_id, venue, scored_prior, conceded_prior):
            team_games = [m for m in chronological if
                          (m.get('homeTeam') or {}).get('id') == team_id or
                          (m.get('awayTeam') or {}).get('id') == team_id]
            team_games = team_games[-10:]
            weighted_for = []
            weighted_against = []
            for match in team_games:
                d = _date(match['utcDate'])
                home = (match.get('homeTeam') or {}).get('id') == team_id
                opp = (match.get('awayTeam') if home else match.get('homeTeam') or {}).get('id')
                prior = [m for m in chronological if _date(m['utcDate']) < d]
                opp_games = _games(list(reversed(prior)), opp, limit=10)
                if len(opp_games) < 5:
                    continue
                opp_stats = _stats(opp_games, match['utcDate'])
                league_h, league_a = league_baselines(prior)
                reference_for = league_h if home else league_a
                reference_against = league_a if home else league_h
                # Effective sample shrinks opponent ratios to neutral.
                n = opp_stats['effective_played']
                opp_def = (n * opp_stats['conceded'] + 8 * reference_for) / (n + 8)
                opp_att = (n * opp_stats['scored'] + 8 * reference_against) / (n + 8)
                goals = match['score']['fullTime']
                gf, ga = (goals['home'], goals['away']) if home else (goals['away'], goals['home'])
                age = max(0.0, (kickoff - d).total_seconds()/86400.0) if kickoff else 0
                w = 2 ** (-age / 180.0)
                weighted_for.append((min(gf,5) * reference_for / max(.4, opp_def), w))
                weighted_against.append((min(ga,5) * reference_against / max(.4, opp_att), w))
            if len(weighted_for) < 3:
                return scored_prior, conceded_prior
            def mean(pairs):
                return sum(v*w for v,w in pairs) / sum(w for _,w in pairs)
            n = len(weighted_for)
            return ((n*mean(weighted_for)+8*scored_prior)/(n+8),
                    (n*mean(weighted_against)+8*conceded_prior)/(n+8))
        # Define kickoff here as this candidate needs it before regularization.
        kickoff = _date(fixture.get('utcDate'))
        hf, ha = adjusted(home_id, 'home', home_for, home_against)
        af, aa = adjusted(away_id, 'away', away_for, away_against)
        lam_h = clamp(.55 * base_h * (.65+.35*hf/base_h) * (.65+.35*aa/base_h)
                      + .45 * (hf+aa)/2)
        lam_a = clamp(.55 * base_a * (.65+.35*af/base_a) * (.65+.35*ha/base_a)
                      + .45 * (af+ha)/2)
    # Experimental fifth model: venue-contrast expected goals. Use relative
    # attack/defence ratios rather than forcing score variety. Reliability
    # weights temper small venue samples. No match outcomes are consulted.
    if variant == 'venue_contrast':
        def reliable_ratio(venue_stats, overall_stats, field, reference):
            if not venue_stats:
                return max(.35, overall_stats[field] / reference)
            n = venue_stats['effective_played']
            blended = (n * venue_stats[field] + 6 * overall_stats[field]) / (n + 6)
            return max(.35, blended / reference)
        home_attack = reliable_ratio(hvs, hs, 'scored', base_h)
        away_defence = reliable_ratio(avs, as_, 'conceded', base_h)
        away_attack = reliable_ratio(avs, as_, 'scored', base_a)
        home_defence = reliable_ratio(hvs, hs, 'conceded', base_a)
        # Temper multiplicative differences and preserve historical league
        # goal totals; not a manual penalty on 1-1 scorelines.
        lam_h = clamp(base_h * (home_attack * away_defence) ** .75)
        lam_a = clamp(base_a * (away_attack * home_defence) ** .75)
    if variant == 'league_opponent':
        from league_opponent_model import expected_goals
        estimate = expected_goals(fixture, past)
        if estimate is None:
            return None
        lam_h, lam_a = estimate
    # Conservative uncertainty regularization for thin historical samples.
    # This is a candidate change requiring walk-forward validation.
    kickoff=_date(fixture.get('utcDate'))
    latest_home=_date(hg[0]['date'])
    latest_away=_date(ag[0]['date'])
    age_home=max(0.0,(kickoff-latest_home).total_seconds()/86400.0) if kickoff and latest_home else 365.0
    age_away=max(0.0,(kickoff-latest_away).total_seconds()/86400.0) if kickoff and latest_away else 365.0
    freshness=2**(-max(age_home,age_away)/180.0)
    # Effective sample size accounts for calendar-decay concentration:
    # ten stale games should not look as informative as ten recent games.
    evidence=min(hs['effective_played'],as_['effective_played'],
                 (hvs['effective_played'] if hvs else 0)+2,
                 (avs['effective_played'] if avs else 0)+2)*freshness
    weight=evidence/(evidence+8.0) if variant=='regularized' else 1.0
    lam_h=clamp(weight*lam_h+(1-weight)*base_h)
    lam_a=clamp(weight*lam_a+(1-weight)*base_a)
    # Recent form / venue / H2H are visible evidence, not arbitrarily
    # applied extra multipliers that could double-count goals.
    # Experimental Dixon-Coles low-score correction. Keep live baseline
    # unchanged until out-of-sample validation shows a measurable gain.
    rho=-0.08 if variant=='dixon_coles' else 0.0
    matrix=_grid(lam_h,lam_a,max_goals,rho=rho)
    # Experimental probability smoothing only; baseline and Smart Double remain unchanged.
    # Temperature 1.35 is a candidate, NOT fitted to the 108 displayed outcomes.
    if variant == 'temperature_candidate':
        logits=np.log(np.maximum(matrix,1e-12))/1.35
        logits-=logits.max()
        matrix=np.exp(logits)
        matrix/=matrix.sum()
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
            'history_count':len(past),'model_version':('temperature-1.35-experimental' if variant=='temperature_candidate' else 'league-opponent-iterative-experimental' if variant=='league_opponent' else 'venue-contrast-experimental' if variant=='venue_contrast' else 'opponent-adjusted-experimental' if variant=='opponent_adjusted' else 'team-strength-geometric-experimental' if variant=='team_strength' else 'venue-arithmetic-experimental' if variant=='venue_arithmetic' else 'venue-form-dixon-coles-experimental' if variant=='dixon_coles' else MODEL_VERSION if variant=='regularized' else LIVE_MODEL_VERSION),
            'evidence_weight':round(weight,3),
            'history_freshness':round(freshness,3),
            'uncertainty_note':('Experimental Dixon-Coles correction; not validated for live use.' if variant=='dixon_coles' else 'Sparse and stale histories are shrunk toward league averages.' if variant=='regularized' else 'Unregularized venue-form forecast selected after historical backtests; no exact score is guaranteed.'),
            'methods':{'strength':(round(lam_h_strength,3),round(lam_a_strength,3)),
                       'attack_defence':(round(lam_h_simple,3),round(lam_a_simple,3))},
            'data_quality':'moderate' if min(len(hg),len(ag),len(hv),len(av))>=8 else 'limited'}
