"""Independent five-league check of already frozen v7 parameters.

Reads existing normalized provider history; never downloads or retunes models.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path
import engine
from forecast_model import predict, PARAMETERS
from reference_model_benchmark import metric, summarize


def evaluate(matches, holdout_year=2026):
    ordered = sorted((m for m in matches if engine.finished(m) and engine._date(m.get('utcDate'))),
                     key=lambda m: engine._date(m['utcDate']))
    histories = defaultdict(list)
    baseline, revised, leagues = [], [], []
    fallbacks = 0
    for fixture in ordered:
        key = (fixture.get('provider', 'football-data.org'), (fixture.get('competition') or {}).get('code'))
        if engine._date(fixture['utcDate']).year == holdout_year:
            past = [m for m in histories[key] if engine._date(m['utcDate']) < engine._date(fixture['utcDate'])]
            old = engine.predict(fixture, past)
            new = predict(fixture, past, variant='reference_v7', league_history=past)
            if old and new:
                baseline.append(metric(old['matrix'], fixture))
                revised.append(metric(new['matrix'], fixture))
                leagues.append(key[1])
                fallbacks += int(bool(new.get('fallback_reason')))
        histories[key].append(fixture)
    by_league = {}
    for league in sorted(set(leagues)):
        indices = [i for i, value in enumerate(leagues) if value == league]
        by_league[league] = {'v6': summarize([baseline[i] for i in indices]),
                            'v7': summarize([revised[i] for i in indices])}
    return {'frozen_parameters': PARAMETERS, 'holdout_year': holdout_year,
        'full_input_matches': len(ordered), 'fallback_matches': fallbacks,
        'paired': {'v6': summarize(baseline), 'v7': summarize(revised)}, 'by_league': by_league,
        'method': 'Frozen public-2023/24-selected parameters. Independent existing API-Football data. Expanding same-provider/league history, strict earlier kickoffs, no tuning on these results.',
        'limits': 'Historical provider replay, not the 236 original live forecasts. Earlier completed holdout matches enter later fits. No guarantee of future improvement.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('matches_json', type=Path)
    parser.add_argument('--holdout-year', type=int, default=2026)
    parser.add_argument('--output', type=Path, default=Path('work/reference_provider_validation.json'))
    args = parser.parse_args()
    report = evaluate(json.loads(args.matches_json.read_text()), args.holdout_year)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
