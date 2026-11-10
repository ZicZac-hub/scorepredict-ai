"""Leakage-safe chronological evaluation of ZicZac's two forecast variants.

Usage:
  python backtest.py path/to/finished_matches.json
Input: JSON array of API-Football-compatible or football-data.org-compatible
finished fixtures (same normalized schema accepted by engine.predict).
No web access, credentials or fabricated fixtures are required.
"""
import argparse
import json
import math
from collections import defaultdict
from engine import predict, finished, _date

EPS = 1e-12

def evaluate(matches, min_history=30):
    ordered = sorted(
        (m for m in matches if finished(m) and _date(m.get('utcDate'))),
        key=lambda m: _date(m['utcDate'])
    )
    stats = {name: defaultdict(float) for name in ('regularized', 'baseline')}
    details = []
    segments = defaultdict(lambda: {name: defaultdict(float) for name in ('regularized', 'baseline')})
    for index, fixture in enumerate(ordered):
        if index < min_history:
            continue
        # Same-kickoff games must not leak each other's final results.
        # Keep league scoring baselines and opponent histories within the
        # same competition; unrelated leagues must not affect goal rates.
        code=(fixture.get('competition') or {}).get('code')
        past = [m for m in ordered[:index]
                if (m.get('competition') or {}).get('code') == code
                and _date(m['utcDate']) < _date(fixture['utcDate'])]
        if len(past) < min_history:
            continue
        actual = fixture['score']['fullTime']
        hg, ag = actual['home'], actual['away']
        outcomes = {}
        for label, variant in (('regularized', 'regularized'), ('baseline', 'baseline')):
            forecast = predict(fixture, past, variant=variant)
            if forecast is None:
                continue
            matrix = forecast['matrix']
            probability = float(matrix[hg, ag]) if hg < matrix.shape[0] and ag < matrix.shape[1] else EPS
            top = forecast['scores'][0]
            winner = 0 if hg > ag else 1 if hg == ag else 2
            probs = [forecast['home_win'], forecast['draw'], forecast['away_win']]
            row = stats[label]
            segment_key = f"{(fixture.get('competition') or {}).get('code', 'unknown')}|{fixture['utcDate'][:4]}"
            segment = segments[segment_key][label]
            row['matches'] += 1
            segment['matches'] += 1
            row['exact_hits'] += int((top[0], top[1]) == (hg, ag))
            segment['exact_hits'] += int((top[0], top[1]) == (hg, ag))
            row['top3_hits'] += int(any((h, a) == (hg, ag) for h, a, _ in forecast['scores'][:3]))
            segment['top3_hits'] += int(any((h, a) == (hg, ag) for h, a, _ in forecast['scores'][:3]))
            row['score_log_loss'] += -math.log(max(probability, EPS))
            segment['score_log_loss'] += -math.log(max(probability, EPS))
            row['outcome_brier'] += sum((p - int(i == winner)) ** 2 for i, p in enumerate(probs))
            segment['outcome_brier'] += sum((p - int(i == winner)) ** 2 for i, p in enumerate(probs))
            row['outcome_hits'] += int(max(range(3), key=lambda i: probs[i]) == winner)
            segment['outcome_hits'] += int(max(range(3), key=lambda i: probs[i]) == winner)
            outcomes[label] = f'{top[0]}-{top[1]}'
        if outcomes:
            details.append({'date': fixture['utcDate'],
                            'home': fixture['homeTeam'].get('name', ''),
                            'away': fixture['awayTeam'].get('name', ''),
                            'actual': f'{hg}-{ag}', **outcomes})
    def metrics(row):
        n = row['matches']
        return {'evaluated': int(n),
                'exact_hit_rate': row['exact_hits']/n if n else None,
                'top3_hit_rate': row['top3_hits']/n if n else None,
                'outcome_accuracy': row['outcome_hits']/n if n else None,
                'outcome_brier': row['outcome_brier']/n if n else None,
                'score_log_loss': row['score_log_loss']/n if n else None}
    summary = {}
    for label, row in stats.items():
        summary[label] = metrics(row)
    by_league_year = {key: {label: metrics(row) for label, row in group.items()}
                      for key, group in sorted(segments.items())}
    return {'summary': summary, 'by_league_year': by_league_year, 'details': details,
            'note': 'Baseline is the unregularized venue-form estimator, not a saved v4 model. Lower Brier/log-loss is better. No future results are used.'}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('matches_json')
    parser.add_argument('--min-history', type=int, default=30)
    parser.add_argument('--output', default='backtest_results.json')
    args = parser.parse_args()
    with open(args.matches_json, encoding='utf-8') as source:
        matches = json.load(source)
    report = evaluate(matches, min_history=args.min_history)
    with open(args.output, 'w', encoding='utf-8') as target:
        json.dump(report, target, indent=2)
    print(json.dumps(report['summary'], indent=2))
    print('Evaluated on strictly earlier fixtures only. Report:', args.output)
