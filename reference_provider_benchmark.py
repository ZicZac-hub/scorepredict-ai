"""Replay frozen models using the shared result-availability/as-of policy.

This is a historical reconstruction, not a prospective ledger. Forecast lead
is fixed before replay; evaluated results never enter their own evidence.
"""
import argparse
import json
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import timedelta
from pathlib import Path

import engine
from forecast_model import predict, PARAMETERS
from history_pipeline import build_evidence, POLICY_VERSION
from reference_model_benchmark import metric, summarize


def _summary(rows):
    if rows:
        return summarize(rows)
    return {'matches': 0, 'exact_hits': 0, 'exact_rate': None,
            'score_log_loss': None, 'outcome_brier': None,
            'top3_rate': None, 'one_one_share': None}


def evaluate(matches, holdout_year=2026, forecast_lead_minutes=60):
    if forecast_lead_minutes < 0:
        raise ValueError('Forecast lead must not be negative')
    ordered = sorted((m for m in matches if engine.finished(m) and engine._date(m.get('utcDate'))),
                     key=lambda m: engine._date(m['utcDate']))
    histories = defaultdict(list)
    baseline, revised, league_poisson, leagues, fixed_hits, empirical_hits = [], [], [], [], [], []
    fallbacks = 0
    for fixture in ordered:
        key = (fixture.get('provider', 'football-data.org'),
               str((fixture.get('competition') or {}).get('id') or (fixture.get('competition') or {}).get('code')))
        kickoff = engine._date(fixture['utcDate'])
        if kickoff.year == holdout_year:
            as_of = kickoff-timedelta(minutes=forecast_lead_minutes)
            evidence = build_evidence(fixture, histories[key], as_of)
            past = evidence['history']
            target = deepcopy(evidence['fixture'])
            target['status'] = 'SCHEDULED'
            target['score'] = {'fullTime': {'home': None, 'away': None}}
            target.pop('provider_status', None)
            old = engine.predict(target, past, league_history=evidence['league_history'])
            new = predict(target, past, variant='reference_v7', league_history=evidence['league_history'])
            if old and new:
                baseline.append(metric(old['matrix'], fixture))
                revised.append(metric(new['matrix'], fixture))
                league_poisson.append(metric(engine._grid(*engine.league_baselines(past)), fixture))
                actual = (fixture['score']['fullTime']['home'], fixture['score']['fullTime']['away'])
                fixed_hits.append(int(actual == (1, 1)))
                counts = Counter((m['score']['fullTime']['home'], m['score']['fullTime']['away']) for m in past)
                empirical = min(counts, key=lambda score: (-counts[score], score)) if counts else (1, 1)
                empirical_hits.append(int(actual == empirical))
                leagues.append((fixture.get('competition') or {}).get('code') or str(key[1]))
                fallbacks += int(bool(new.get('fallback_reason')))
        histories[key].append(fixture)
    by_league = {}
    for league in sorted(set(leagues)):
        indices = [i for i, value in enumerate(leagues) if value == league]
        by_league[league] = {'v6': _summary([baseline[i] for i in indices]),
                            'v7': _summary([revised[i] for i in indices]),
                            'league_poisson': _summary([league_poisson[i] for i in indices]),
                            'always_1_1_exact_hits': sum(fixed_hits[i] for i in indices),
                            'prior_league_mode_exact_hits': sum(empirical_hits[i] for i in indices)}
    n = len(baseline)
    return {'frozen_parameters': PARAMETERS, 'holdout_year': holdout_year,
        'forecast_lead_minutes': forecast_lead_minutes, 'policy_version': POLICY_VERSION,
        'full_input_matches': len(ordered), 'fallback_matches': fallbacks,
        'paired': {'v6': _summary(baseline), 'v7': _summary(revised),
                   'league_poisson': _summary(league_poisson)},
        'fixed_baselines': {'paired_matches': n, 'always_1_1_exact_hits': sum(fixed_hits),
                           'always_1_1_exact_rate': sum(fixed_hits)/n if n else None,
                           'prior_league_mode_exact_hits': sum(empirical_hits),
                           'prior_league_mode_exact_rate': sum(empirical_hits)/n if n else None},
        'by_league': by_league,
        'method': 'Frozen model parameters. Shared provider-qualified current/previous-season history, 730-day maximum, and results available by kickoff minus the declared forecast lead. Explicit finishedAt is used when present; otherwise kickoff plus three hours. No tuning on these results.',
        'limits': 'Historical reconstruction under a changed evidence protocol, not the 236 original live forecasts or a new untouched test. Old benchmark counts used a different cutoff and must not be compared as a model improvement.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('matches_json', type=Path)
    parser.add_argument('--holdout-year', type=int, default=2026)
    parser.add_argument('--forecast-lead-minutes', type=int, default=60)
    parser.add_argument('--output', type=Path, default=Path('work/reference_provider_validation.json'))
    args = parser.parse_args()
    report = evaluate(json.loads(args.matches_json.read_text()), args.holdout_year, args.forecast_lead_minutes)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
