"""Select v7 exact-score parameters on 2023/24, replay 2024/25 development data.

Run with cached CSVs: python reference_model_benchmark.py --data-dir PATH
No downloads, AI calls, provider credentials, or final-test parameter tuning.
"""
import argparse
import csv
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import engine
from reference_model import fit, score_matrix


CONFIGURATIONS = [{'regularization': p, 'half_life_days': d}
                  for p in (2., 6., 12.) for d in (180., 365.)]
WEIGHTS = (0., .25, .5, .75, 1.)


def load_matches(folder):
    rows = []
    for season in ('2223', '2324', '2425'):
        for source, code in (('E0', 'PL'), ('D1', 'BL1')):
            for row in csv.DictReader((folder/f'{season}_{source}.csv').open(encoding='utf-8-sig')):
                if not row.get('Date') or row.get('FTHG', '') == '' or row.get('FTAG', '') == '':
                    continue
                date = datetime.strptime(row['Date'], '%d/%m/%Y').replace(tzinfo=timezone.utc)
                rows.append({'id': f'{source}-{season}-{row["Date"]}-{row["HomeTeam"]}',
                    'provider': 'football-data.co.uk', 'utcDate': date.isoformat(), 'status': 'FINISHED',
                    'homeTeam': {'id': row['HomeTeam'], 'name': row['HomeTeam']},
                    'awayTeam': {'id': row['AwayTeam'], 'name': row['AwayTeam']},
                    'competition': {'code': code}, 'dataset_season': season,
                    'score': {'fullTime': {'home': int(row['FTHG']), 'away': int(row['FTAG'])}}})
    return sorted(rows, key=lambda row: row['utcDate'])


def sample(matches, season):
    result = []
    for fixture in matches:
        if fixture['dataset_season'] != season:
            continue
        past = [m for m in matches if m['competition']==fixture['competition'] and m['utcDate'] < fixture['utcDate']]
        baseline = engine.predict(fixture, past)
        if baseline:
            result.append((fixture, past, baseline['matrix']))
    return result


def metric(matrix, fixture):
    h, a = fixture['score']['fullTime']['home'], fixture['score']['fullTime']['away']
    probability = float(matrix[h,a]) if h<matrix.shape[0] and a<matrix.shape[1] else 0
    outcome = 0 if h>a else 1 if h==a else 2
    probabilities = (float(np.tril(matrix,-1).sum()), float(np.trace(matrix)), float(np.triu(matrix,1).sum()))
    top = tuple(int(v) for v in np.unravel_index(matrix.argmax(), matrix.shape))
    order = np.argsort(-matrix.ravel(), kind='stable')[:3]
    top3 = {(int(i//matrix.shape[1]), int(i%matrix.shape[1])) for i in order}
    return {'log_loss': -math.log(max(probability, 1e-12)),
        'brier': sum((p-int(i==outcome))**2 for i,p in enumerate(probabilities)),
        'exact': int(top==(h,a)), 'top3': int((h,a) in top3), 'one_one': int(top==(1,1))}


def summarize(metrics):
    n = len(metrics)
    return {'matches': n, 'exact_hits': sum(row['exact'] for row in metrics),
        'exact_rate': sum(row['exact'] for row in metrics)/n,
        'score_log_loss': sum(row['log_loss'] for row in metrics)/n,
        'outcome_brier': sum(row['brier'] for row in metrics)/n,
        'top3_rate': sum(row['top3'] for row in metrics)/n,
        'one_one_share': sum(row['one_one'] for row in metrics)/n}


def run(matches, configuration, rows, weights=WEIGHTS):
    output = {weight: [] for weight in weights}
    paired_baseline = []
    paired_leagues = []
    for fixture, past, baseline in rows:
        fitted = fit(fixture, past, **configuration)
        if fitted is None:
            continue
        matrix = score_matrix(fitted)
        paired_baseline.append(metric(baseline, fixture))
        paired_leagues.append(fixture['competition']['code'])
        for weight in weights:
            mixture = engine.mix_score_matrices([baseline, matrix], [1-weight, weight])
            output[weight].append(metric(mixture, fixture))
    return output, paired_baseline, paired_leagues


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=Path('work/model_benchmark_data'))
    parser.add_argument('--output', type=Path, default=Path('work/reference_model_validation.json'))
    args = parser.parse_args()
    matches = load_matches(args.data_dir)
    validation = sample(matches, '2324')
    choices = []
    for configuration in CONFIGURATIONS:
        metrics, baseline, leagues = run(matches, configuration, validation)
        for weight, values in metrics.items():
            choices.append({**configuration, 'fitted_weight': weight, **summarize(values)})
        print('validation', configuration, min((x for x in choices if all(x[k]==v for k,v in configuration.items())), key=lambda x:x['score_log_loss']), flush=True)
    chosen = min(choices, key=lambda row:(-row['exact_rate'], -row['top3_rate'], row['score_log_loss']))
    parameters = {key: chosen[key] for key in ('regularization', 'half_life_days', 'fitted_weight')}
    holdout = sample(matches, '2425')
    metrics, baseline, leagues = run(matches, {key: parameters[key] for key in ('regularization','half_life_days')}, holdout, [parameters['fitted_weight']])
    candidate = metrics[parameters['fitted_weight']]
    by_league = {}
    for league in sorted(set(leagues)):
        indices = [i for i,code in enumerate(leagues) if code==league]
        by_league[league] = {'v6': summarize([baseline[i] for i in indices]),
                            'v7': summarize([candidate[i] for i in indices])}
    report = {'sources': ['https://www.football-data.co.uk/data.php'],
        'training_initial_season': '2022/23', 'parameter_selection_season': '2023/24',
        'development_replay_season': '2024/25', 'chosen_parameters': parameters,
        'selection_criterion': 'Highest paired exact-score hit rate on 2023/24 only, then top-three rate, then lower full-grid log loss. Includes v6 weight0; all same-day/future results excluded.',
        'validation_candidates': choices, 'validation_selected': chosen,
        'holdout': {'v6': summarize(baseline), 'v7': summarize(candidate)}, 'by_league': by_league,
        'limits': '2024/25 was already inspected during earlier logloss-first development and is not an untouched test for this exact-first selection policy. Independent provider2026 validation follows frozen selection. This is not the live 236-match archive and does not prove future gains.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({key:report[key] for key in ('chosen_parameters','holdout','by_league')}, indent=2), flush=True)


if __name__ == '__main__':
    main()
