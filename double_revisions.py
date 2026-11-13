"""Separate revisions of five-double sets; original forecasts remain immutable.

Retrospective comparisons never enter the original prospective win ledger.
Each leg must have been recomputed from information available by the pair's
original capture, or the pair is explicitly skipped rather than backfilled.
"""
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from double_tracker import saved_pairs as sqlite_saved_pairs
from smart_double_ai import five_doubles

DEFAULT_OUTPUT = Path('data/prematch/double_rescores')
KNOWN_PROVIDERS = {'football-data.org', 'API-Football'}


def _date(value):
    try:
        result = datetime.fromisoformat(str(value).replace('Z', '+00:00'))
        return result.replace(tzinfo=timezone.utc) if result.tzinfo is None else result
    except (ValueError, TypeError):
        return None


def _choices(raw):
    output = []
    for value in raw or []:
        try:
            output.append({'score1': value.get('score1', value.get('Score 1')),
                           'score2': value.get('score2', value.get('Score 2')),
                           'probability': float(value.get('probability', value.get('Joint')))})
        except (TypeError, ValueError, AttributeError):
            continue
    return output


def load_saved_pairs(conn=None, pair_dir='data/prematch/double_pairs',
                     smart_dir='data/prematch/smart_double_pairs'):
    """Read SQLite and explicitly API-Football-origin permanent pair archives."""
    output = []
    if conn is not None:
        for row in sqlite_saved_pairs(conn):
            try:
                original_choices = _choices(json.loads(row['predictions_json']))
            except (ValueError, TypeError):
                output.append({'pair_key': 'sqlite:'+row['snapshot_key'], 'source': 'sqlite',
                               'load_error': 'invalid saved SQLite double predictions'})
                continue
            output.append({'pair_key': 'sqlite:'+row['snapshot_key'],
                'source': 'sqlite', 'captured_at_utc': row['saved_at'],
                'first_fixture_id': row['first_id'], 'second_fixture_id': row['second_id'],
                'first_provider': row['first_provider'], 'second_provider': row['second_provider'],
                'first_match': row['first_name'], 'second_match': row['second_name'],
                'first_kickoff_utc': row['first_kickoff'], 'second_kickoff_utc': row['second_kickoff'],
                'original_models': [row['first_model'], row['second_model']],
                'original_five_doubles': original_choices})
    for directory, kind, choice_key in ((Path(pair_dir), 'permanent_pair', 'original'),
                                        (Path(smart_dir), 'permanent_smart_pair', 'five_doubles')):
        for path in sorted(directory.glob('*.json')):
            try:
                row = json.loads(path.read_text(encoding='utf-8'))
                output.append({'pair_key': kind+':'+str(row.get('pair_key') or path.stem),
                    'source': str(path), 'captured_at_utc': row['captured_at_utc'],
                    'first_fixture_id': str(row['first_fixture_id']),
                    'second_fixture_id': str(row['second_fixture_id']),
                    # These two directories are written only by API-Football
                    # snapshot jobs, unlike ambiguous historical SQLite rows.
                    'first_provider': row.get('first_provider', 'API-Football'),
                    'second_provider': row.get('second_provider', 'API-Football'),
                    'first_match': row.get('first_match'), 'second_match': row.get('second_match'),
                    'first_kickoff_utc': row.get('first_kickoff_utc'),
                    'second_kickoff_utc': row.get('second_kickoff_utc'),
                    'original_five_doubles': _choices(row.get(choice_key))})
            except (ValueError, TypeError, KeyError, OSError):
                output.append({'pair_key': kind+':'+path.stem, 'source': str(path),
                               'load_error': 'invalid saved pair record'})
    return output


def _revision_key(row):
    return str(row.get('provider')), str(row.get('fixture_id'))


def revise_saved_pairs(pairs, fixture_revisions, now=None, model_version=None):
    """Rank five revised alternatives using prior-only leg forecasts, never results."""
    now = _date(now) if now is not None else datetime.now(timezone.utc)
    if now is None:
        raise ValueError('A valid UTC run time is required')
    if isinstance(fixture_revisions, dict):
        fixture_revisions = fixture_revisions.get('revisions', [])
    indexed = {}
    for revision in fixture_revisions:
        indexed.setdefault(_revision_key(revision), []).append(revision)
    report = {'rescored_at_utc': now.isoformat(), 'model_version': model_version,
              'saved_pairs': len(pairs), 'revisions': [], 'skipped': [],
              'probability_assumption': 'Independent matches; sum of five mutually exclusive score-pair probabilities.',
              'excluded_from_original_accuracy': True}
    for pair in pairs:
        key = pair.get('pair_key', 'unknown')
        try:
            if pair.get('load_error'):
                raise ValueError(pair['load_error'])
            identities = [(str(pair.get(side+'_provider')), str(pair.get(side+'_fixture_id')))
                          for side in ('first', 'second')]
            if any(provider not in KNOWN_PROVIDERS for provider, _ in identities):
                raise ValueError('original provider unknown; cannot identify both legs safely')
            if identities[0] == identities[1]:
                raise ValueError('same fixture cannot supply both legs')
            captured = _date(pair.get('captured_at_utc'))
            if captured is None:
                raise ValueError('original capture time unavailable')
            groups = [indexed.get(identity, []) for identity in identities]
            if any(not group for group in groups):
                raise ValueError('one or both fixtures have no revised score distribution')
            kickoffs = []
            for side, group in zip(('first', 'second'), groups):
                kickoff = _date(pair.get(side+'_kickoff_utc'))
                if kickoff is None:
                    kickoff = _date(group[0].get('original_kickoff_utc') or group[0].get('kickoff_utc'))
                if kickoff is None:
                    raise ValueError('fixture kickoff unavailable')
                kickoffs.append(kickoff)
            if (pair.get('first_match') and str(pair['first_match']).casefold().strip() ==
                    str(pair.get('second_match') or '').casefold().strip() and kickoffs[0] == kickoffs[1]):
                raise ValueError('same physical match cannot supply both legs')
            mode = 'upcoming' if now < min(kickoffs) else 'retrospective'
            legs = []
            for group, kickoff in zip(groups, kickoffs):
                allowed = []
                for revision in group:
                    cutoff = _date(revision.get('history_as_of_utc'))
                    generated = _date(revision.get('rescored_at_utc'))
                    if not cutoff or not generated or cutoff >= kickoff or generated > now:
                        continue
                    if mode == 'retrospective' and cutoff > captured:
                        continue
                    if mode == 'upcoming' and (revision.get('mode') != 'upcoming' or cutoff > now):
                        continue
                    if not revision.get('updated_top10_scores'):
                        continue
                    allowed.append(revision)
                if not allowed:
                    raise ValueError('no revised leg available by the original pair cutoff')
                legs.append(max(allowed, key=lambda row: (row.get('comparison_context') == key,
                                                        _date(row['history_as_of_utc']))))
            forecasts = [{'fixture': {'provider': provider, 'id': fixture_id},
                          'result': {'scores': leg['updated_top10_scores']}}
                         for (provider, fixture_id), leg in zip(identities, legs)]
            options = five_doubles(*forecasts)
            if len(options) != 5:
                raise ValueError('fewer than five valid distinct score pairs available')
            updated = _choices(options)
            original = pair.get('original_five_doubles') or []
            record = {'original_pair_key': key, 'source': pair.get('source'),
                'mode': mode, 'rescored_at_utc': now.isoformat(),
                'original_captured_at_utc': captured.isoformat(),
                'original_was_pre_kickoff': captured < min(kickoffs),
                'first_provider': identities[0][0], 'second_provider': identities[1][0],
                'first_fixture_id': identities[0][1], 'second_fixture_id': identities[1][1],
                'first_match': pair.get('first_match'), 'second_match': pair.get('second_match'),
                'first_kickoff_utc': kickoffs[0].isoformat(), 'second_kickoff_utc': kickoffs[1].isoformat(),
                'history_as_of_utc': [leg['history_as_of_utc'] for leg in legs],
                'updated_models': [leg['updated_model'] for leg in legs],
                'leg_revision_ids': [leg.get('revision_id') for leg in legs],
                'comparison_contexts': [leg.get('comparison_context') for leg in legs],
                'original_five_doubles': original, 'updated_five_doubles': updated,
                'estimated_coverage': sum(option['probability'] for option in updated),
                'changed': [(x['score1'], x['score2']) for x in original] !=
                           [(x['score1'], x['score2']) for x in updated],
                'probability_assumption': report['probability_assumption'],
                'excluded_from_original_accuracy': True,
                'label': 'Upcoming revised five doubles' if mode == 'upcoming' else
                         'Retrospective comparison — not an original pre-match prediction'}
            identity = json.dumps(record, sort_keys=True, separators=(',', ':'))
            record['revision_id'] = hashlib.sha256(identity.encode()).hexdigest()
            report['revisions'].append(record)
        except (ValueError, TypeError, KeyError) as exc:
            report['skipped'].append({'pair_key': key, 'reason': str(exc)})
    report['rescored_pairs'] = len(report['revisions'])
    report['upcoming_pairs'] = sum(r['mode'] == 'upcoming' for r in report['revisions'])
    report['retrospective_pairs'] = sum(r['mode'] == 'retrospective' for r in report['revisions'])
    return report


def _write_once(path, value):
    """Publish a fully written file without replacing existing evidence."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile('w', dir=path.parent, suffix='.tmp',
                                         encoding='utf-8', delete=False) as target:
            temporary = target.name
            json.dump(value, target, indent=2, sort_keys=True)
            target.write('\n')
            target.flush()
            os.fsync(target.fileno())
        try:
            # A hard link publishes the complete inode atomically and refuses
            # to replace an existing immutable record, even across processes.
            os.link(temporary, path)
            return True
        except FileExistsError:
            try:
                existing = json.loads(path.read_text(encoding='utf-8'))
            except (ValueError, OSError) as exc:
                raise ValueError('Existing immutable revision is unreadable: '+str(path)) from exc
            if existing != value:
                raise ValueError('Conflicting immutable revision: '+str(path))
            return False
    finally:
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)


def persist_double_revisions(report, output_dir=DEFAULT_OUTPUT):
    """Append immutable revision/run evidence; only the latest-report pointer changes."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for record in report.get('revisions', []):
        _write_once(output/'revisions'/(record['revision_id']+'.json'), record)
    run_id = hashlib.sha256(json.dumps(report, sort_keys=True).encode()).hexdigest()
    _write_once(output/'runs'/(run_id+'.json'), report)
    with tempfile.NamedTemporaryFile('w', dir=output, suffix='.tmp', encoding='utf-8', delete=False) as target:
        json.dump(report, target, indent=2, sort_keys=True)
        target.write('\n')
        temporary = target.name
    os.replace(temporary, output/'latest_report.json')
    return report
