"""Revise upcoming forecasts and reconstruct finished fixtures as of original capture.

Original SQLite rows and permanent prediction snapshots are read-only inputs.
Completed comparisons are retrospective evidence, never replacement live picks.
"""
import argparse
import csv
import hashlib
import json
import math
import os
import re
import sqlite3
import unicodedata
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

from api_football import LEAGUES, _convert, _request, _season
from engine import finished
from history_pipeline import build_evidence, load_evidence
from persist_prematch import earliest_snapshots, parse_time
from revision_store import save_revision_report

OUTPUT = Path('data/prematch/rescores')
LEGACY_PROVIDER = 'legacy-unknown'


def _name(value):
    return ' '.join(unicodedata.normalize('NFKC', str(value or '')).casefold().split())


def _score(value):
    if isinstance(value, str):
        try:
            value = [int(part) for part in value.split('-')]
        except ValueError:
            return None
    if isinstance(value, (list, tuple)) and len(value) == 2 and all(type(v) is int and v >= 0 for v in value):
        return list(value)
    return None


def _competition(value):
    if isinstance(value, dict):
        return value.get('code') or ''
    value = str(value or '')
    match = re.search(r'\(([A-Za-z0-9_-]+)\)\s*$', value)
    return match.group(1) if match else value


def normalize_records(snapshot_records=(), archive_rows=()):
    """Normalize both archives, retaining the earliest original per provider/fixture.

    Missing identity or invalid dates remain explicit rows for the rescore report
    to exclude; they are not silently converted into fabricated forecasts.
    """
    if isinstance(snapshot_records, dict):
        snapshot_records = snapshot_records.values()
    originals = []
    for kind, rows in (('permanent_snapshot', snapshot_records), ('sqlite_archive', archive_rows)):
        for source in rows:
            row = dict(source)
            snapshot = kind == 'permanent_snapshot'
            originals.append({
                'provider': row.get('provider') or ('API-Football' if snapshot else LEGACY_PROVIDER),
                'fixture_id': row.get('fixture_id'),
                'competition': _competition(row.get('competition') if snapshot else row.get('league')),
                'kickoff_utc': row.get('kickoff_utc') if snapshot else row.get('kickoff'),
                'captured_at_utc': row.get('captured_at_utc') if snapshot else row.get('saved_at'),
                'home_team': row.get('home_team') if snapshot else row.get('home'),
                'away_team': row.get('away_team') if snapshot else row.get('away'),
                'home_team_id': row.get('home_team_id') if snapshot else row.get('home_id'),
                'away_team_id': row.get('away_team_id') if snapshot else row.get('away_id'),
                'predicted_score': _score(row.get('predicted_score')),
                'forecast_model': (row.get('forecast_model') if snapshot else row.get('model_version')) or 'original-model-unknown',
                'original_score_probability': row.get('predicted_score_probability'),
                'source': row.get('source') or kind,
                'source_ref': row.get('source_ref'),
                'requested_as_of_utc': row.get('requested_as_of_utc'),
                'force_retrospective': row.get('force_retrospective', False),
                'comparison_context': row.get('comparison_context'),
            })
    # Earliest-source choice is based only on timestamps, never actual results.
    def order(row):
        try:
            return parse_time(row['captured_at_utc'])
        except (ValueError, TypeError):
            return datetime.max.replace(tzinfo=timezone.utc)
    selected = {}
    for row in sorted(originals, key=order):
        key = (row['provider'], str(row['fixture_id']))
        if key not in selected:
            selected[key] = row
        else:
            selected[key].setdefault('additional_original_sources', []).append(row['source'])
    return list(selected.values())


def _matches_original(record, fixture):
    source = fixture.get('provider') or 'football-data.org'
    if record['provider'] != LEGACY_PROVIDER and source != record['provider']:
        return False
    if str(fixture.get('id')) != str(record.get('fixture_id')):
        return False
    try:
        if parse_time(fixture.get('utcDate')) != parse_time(record.get('kickoff_utc')):
            return False
    except (ValueError, TypeError):
        return False
    for side in ('home', 'away'):
        team = fixture.get(side + 'Team') or {}
        original_id = record.get(side + '_team_id')
        if original_id is not None and record['provider'] != LEGACY_PROVIDER:
            if str(team.get('id')) != str(original_id):
                return False
        elif not _name(record.get(side + '_team')) or _name(team.get('name')) != _name(record.get(side + '_team')):
            return False
    return True


def _safe_history(history, fixture, cutoff):
    """Compatibility wrapper around the shared evidence policy."""
    return build_evidence(fixture, history, cutoff)['history']


def rescore_records(records, history_loader, fixture_loader, now=None, predictor=None, progress=None):
    """Return revisions without changing originals; loaders are independently testable.

    fixture_loader(record) returns a normalized fixture.
    history_loader(record, fixture, cutoff) returns normalized league results.
    progress(index, total, status_dict) is called once per processed original.
    """
    if predictor is None:
        from functools import partial
        from forecast_model import predict
        predictor = partial(predict, variant='reference_v7')
    now = parse_time(now or datetime.now(timezone.utc))
    records = list(records)
    report = {'run_id': now.strftime('%Y%m%dT%H%M%S%fZ') + '-' + uuid4().hex[:8],
              'rescored_at_utc': now.isoformat(), 'model_version': 'reference-informed-v7',
              'input_count': len(records), 'revisions': [], 'skipped': [],
              'note': 'Original forecasts remain unchanged. Upcoming revisions use current pre-kickoff results; retrospective comparisons use only results available by the original capture and never enter original live accuracy.'}
    seen = set()
    for index, original in enumerate(records, 1):
        status = {'fixture_id': original.get('fixture_id'), 'provider': original.get('provider')}
        try:
            capture, original_kickoff = parse_time(original['captured_at_utc']), parse_time(original['kickoff_utc'])
            if capture >= original_kickoff or capture > now:
                raise ValueError('original was not captured before kickoff and this run')
            if not original.get('predicted_score'):
                raise ValueError('original score is missing or invalid')
            fixture = fixture_loader(deepcopy(original))
            if not fixture or not _matches_original(original, fixture):
                raise ValueError('provider, teams or kickoff do not match the original fixture')
            provider = fixture.get('provider') or 'football-data.org'
            kickoff = parse_time(fixture['utcDate'])
            upcoming = kickoff > now and fixture.get('status') in ('SCHEDULED', 'TIMED')
            if upcoming and fixture.get('provider_status') not in (None, '', 'NS', 'TBD'):
                raise ValueError('provider does not confirm an upcoming fixture')
            if not upcoming and kickoff > now:
                raise ValueError('future fixture is cancelled, postponed or unavailable')
            forced = bool(original.get('force_retrospective') or original.get('requested_as_of_utc'))
            mode = 'upcoming' if upcoming and not forced else 'retrospective'
            cutoff = now if mode == 'upcoming' else parse_time(original.get('requested_as_of_utc') or capture)
            if cutoff > now or cutoff >= kickoff:
                raise ValueError('requested historical cutoff is after this run or not before kickoff')
            pair_future_replay = bool(upcoming and original.get('force_retrospective')
                                      and original.get('comparison_context')
                                      and original.get('requested_as_of_utc'))
            if mode == 'retrospective' and not finished(fixture) and not pair_future_replay:
                raise ValueError('fixture is not finished; ordinary retrospective comparisons require a verified final score')
            identity = (provider, str(fixture['id']), mode, cutoff.isoformat(), str(original.get('comparison_context') or ''))
            if identity in seen:
                raise ValueError('duplicate source for an already revised fixture')
            revision_id = report['run_id'] + '-' + hashlib.sha256(repr(identity).encode()).hexdigest()[:12]
            loaded_history = history_loader(deepcopy(original), deepcopy(fixture), cutoff)
            history_warnings = loaded_history.get('warnings', []) if isinstance(loaded_history, dict) else []
            source_rows = (loaded_history.get('history', loaded_history.get('matches', []))
                           if isinstance(loaded_history, dict) else loaded_history)
            source_metadata = loaded_history.get('provenance', []) if isinstance(loaded_history, dict) else None
            evidence = build_evidence(fixture, source_rows, cutoff, source_metadata=source_metadata)
            history = evidence['history']
            # Hide the target's actual score/status from the predictor itself.
            model_fixture = deepcopy(evidence['fixture'])
            model_fixture['status'] = 'SCHEDULED'
            model_fixture['score'] = {'fullTime': {'home': None, 'away': None}}
            model_fixture.pop('provider_status', None)
            forecast = predictor(model_fixture, history, league_history=history)
            if not forecast or not forecast.get('scores'):
                raise ValueError('insufficient historical evidence for this as-of prediction')
            scores = [[int(h), int(a), float(p)] for h, a, p in forecast['scores'][:10]]
            if any(not math.isfinite(p) or not 0 <= p <= 1 for _, _, p in scores):
                raise ValueError('model returned invalid exact-score probabilities')
            if any(h < 0 or a < 0 for h, a, _ in scores) or len({(h,a) for h,a,_ in scores}) != len(scores):
                raise ValueError('model returned invalid or duplicate scorelines')
            if any(p > scores[0][2] + 1e-12 for _, _, p in scores):
                raise ValueError('model score probabilities are not ranked')
            outcome_values = [float(forecast[name]) for name in ('home_win','draw','away_win')]
            if any(not math.isfinite(p) or not 0 <= p <= 1 for p in outcome_values) or abs(sum(outcome_values)-1) > 1e-6:
                raise ValueError('model returned invalid match-outcome probabilities')
            if any(not math.isfinite(float(forecast[name])) or float(forecast[name]) < 0 for name in ('home_xg','away_xg')):
                raise ValueError('model returned invalid goal rates')
            revised = scores[0][:2]
            actual = _score((fixture.get('score') or {}).get('fullTime') and [fixture['score']['fullTime'].get('home'), fixture['score']['fullTime'].get('away')]) if finished(fixture) else None
            row = {'run_id': report['run_id'], 'revision_id': revision_id, 'provider': provider,
                   'original_provider': original['provider'], 'fixture_id': int(fixture['id']),
                   'competition': (fixture.get('competition') or {}).get('code') or original['competition'],
                   'home_team': original.get('home_team'), 'away_team': original.get('away_team'),
                   'home_team_id': fixture['homeTeam']['id'], 'away_team_id': fixture['awayTeam']['id'],
                   'kickoff_utc': fixture['utcDate'], 'original_kickoff_utc': original['kickoff_utc'],
                   'rescored_at_utc': now.isoformat(), 'history_as_of_utc': cutoff.isoformat(),
                   'original_captured_at_utc': original['captured_at_utc'],
                   'mode': mode, 'label': 'Upcoming forecast revision' if mode == 'upcoming' else 'Retrospective as-of comparison',
                   'comparison_context': original.get('comparison_context'), 'source_ref': original.get('source_ref'),
                   'excluded_from_original_accuracy': True,
                   'original_score': original['predicted_score'], 'updated_score': revised,
                   'changed': revised != original['predicted_score'],
                   'original_model': original.get('forecast_model'),
                   'original_score_probability': original.get('original_score_probability'),
                   'updated_model': forecast.get('model_version', 'reference-informed-v7'),
                   'updated_score_probability': scores[0][2], 'updated_top10_scores': scores,
                   'updated_expected_goals': {'home': float(forecast['home_xg']), 'away': float(forecast['away_xg'])},
                   'updated_outcome_probabilities': {'home': float(forecast['home_win']), 'draw': float(forecast['draw']), 'away': float(forecast['away_win'])},
                   'history_count': len(history), 'history_availability_policy': 'shared two-season as-of policy; final-whistle timestamp when supplied, otherwise kickoff plus three hours',
                   'policy_version': evidence['policy_version'], 'input_digest': evidence['input_digest'],
                   'history_window_start_utc': evidence['history_window_start_utc'], 'provenance': evidence.get('provenance', []),
                   'actual_score': actual, 'original_source': original.get('source'),
                   'history_warnings': history_warnings,
                   'model_sources': forecast.get('model_sources', []), 'fallback_reason': forecast.get('fallback_reason')}
            if actual is not None and mode == 'retrospective':
                row['original_hit'] = original['predicted_score'] == actual
                row['updated_hit'] = revised == actual
            report['revisions'].append(row)
            seen.add(identity)
            status.update(status='revised', mode=mode)
        except Exception as exc:
            excluded = dict(status, reason=str(exc)[:180])
            report['skipped'].append(excluded)
            status.update(status='skipped', reason=excluded['reason'])
        if progress:
            progress(index, len(records), status)
    report['summary'] = {'input_records': len(records), 'processed': len(records),
                         'revised': len(report['revisions']),
                         'upcoming_revisions': sum(r['mode'] == 'upcoming' for r in report['revisions']),
                         'retrospective_comparisons': sum(r['mode'] == 'retrospective' for r in report['revisions']),
                         'skipped': len(report['skipped']), 'changed': sum(r['changed'] for r in report['revisions'])}
    return report


def provider_loaders(api_key='', primary_get=None):
    """Build bounded, cached loaders with provider-qualified fixture identities."""
    cache = {}
    def fixture_loader(record):
        sources = ('API-Football', 'football-data.org') if record['provider'] == LEGACY_PROVIDER else (record['provider'],)
        errors = []
        for source in sources:
            try:
                if source == 'API-Football' and api_key:
                    candidates = [_convert(raw, record['competition']) for raw in _request(api_key, '/fixtures', {'id': str(record['fixture_id'])})]
                elif source == 'football-data.org' and primary_get:
                    from result_verification import primary_result
                    candidates = [primary_result(primary_get('/matches/' + str(record['fixture_id'])))]
                else:
                    continue
                for candidate in candidates:
                    if _matches_original(record, candidate):
                        return candidate
            except Exception as exc:
                errors.append(source + ': ' + str(exc)[:100])
        raise ValueError('matching fixture unavailable from its original provider' + ('; ' + '; '.join(errors) if errors else ''))

    def history_loader(record, fixture, cutoff):
        evidence = load_evidence(fixture, as_of=cutoff, api_key=api_key,
                                 primary_get=primary_get, api_request=_request, cache=cache)
        # Keep `matches` for older callback consumers; every path receives the
        # same policy-filtered population and provenance.
        return dict(evidence, matches=evidence['history'])
    return history_loader, fixture_loader


def _redact(value, secrets):
    """Keep configured credentials out of persisted provider diagnostics."""
    if isinstance(value, str):
        for secret in secrets:
            if secret:
                value = value.replace(str(secret), '[REDACTED]')
        return value
    if isinstance(value, dict):
        return {key: _redact(item, secrets) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_redact(item, secrets) for item in value]
    return value


def run_rescoring(api_key='', primary_get=None, archive_rows=(), snapshot_records=None,
                  now=None, output_dir=OUTPUT, predictor=None, progress=None, records=None,
                  redact_values=()):
    if records is None:
        snapshots = earliest_snapshots() if snapshot_records is None else snapshot_records
        records = normalize_records(snapshots, archive_rows)
    else:
        records = deepcopy(list(records))
    secrets = tuple(value for value in (api_key, *redact_values) if value)
    safe_progress = (lambda index, total, status: progress(index, total, _redact(status, secrets))) if progress else None
    history_loader, fixture_loader = provider_loaders(api_key, primary_get)
    report = rescore_records(records, history_loader, fixture_loader, now, predictor, safe_progress)
    report = _redact(report, secrets)
    return save_revision_report(report, output_dir)


def _read_archive(path):
    path = Path(path)
    if not path.exists():
        return []
    conn = sqlite3.connect(path.resolve().as_uri() + '?mode=ro', uri=True)
    try:
        conn.row_factory = sqlite3.Row
        return [dict(row) for row in conn.execute('SELECT * FROM predictions')]
    finally:
        conn.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--archive-db', default='predictions.sqlite3')
    parser.add_argument('--archive-csv')
    parser.add_argument('--output-dir', default=str(OUTPUT))
    args = parser.parse_args()
    archive = _read_archive(args.archive_db)
    if args.archive_csv:
        with open(args.archive_csv, encoding='utf-8', newline='') as handle:
            archive.extend(csv.DictReader(handle))
    primary_key = os.environ.get('FOOTBALL_DATA_API_KEY', '')
    primary_get = None
    if primary_key:
        from data import get_json
        primary_get = lambda path: get_json(path, primary_key)
    report = run_rescoring(api_key=os.environ.get('API_FOOTBALL_KEY', ''), primary_get=primary_get,
                           archive_rows=archive, output_dir=args.output_dir)
    print(json.dumps(report['summary'], indent=2))


if __name__ == '__main__':
    main()
