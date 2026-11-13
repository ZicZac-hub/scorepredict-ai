"""One background upgrade-rescoring worker for the deployed prediction archive.

The repository workflow cannot read a Streamlit instance's local SQLite file.
This worker processes that instance's original forecasts without blocking its
page render. It writes only separate revision records and a public progress
file; provider credentials and callbacks remain in process memory.
"""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import threading
import uuid

OUTPUT = Path('data/prematch/rescores')
_LOCK = threading.RLock()
_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix='forecast-revisions')
_ACTIVE = None
_ACTIVE_DIR = None


def _now():
    return datetime.now(timezone.utc).isoformat()


def _status_path(output_dir):
    return Path(output_dir) / 'server_status.json'


def _write_status(status, output_dir):
    path = _status_path(output_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    temporary.write_text(json.dumps(status, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    os.replace(temporary, path)


def get_revision_status(output_dir=OUTPUT):
    """Return progress without requiring API access or a running worker."""
    try:
        result = json.loads(_status_path(output_dir).read_text(encoding='utf-8'))
        return result if isinstance(result, dict) else {'state': 'not_started'}
    except (OSError, ValueError, TypeError):
        return {'state': 'not_started', 'input_records': 0, 'processed': 0,
                'revised': 0, 'skipped': 0, 'input_pairs': 0}


def _model_version():
    from forecast_model import LIVE_MODEL_VERSION
    return LIVE_MODEL_VERSION


def _source_records(archive_rows, snapshot_records):
    from rescore_predictions import normalize_records
    if snapshot_records is None:
        from persist_prematch import earliest_snapshots
        snapshot_records = list(earliest_snapshots().values())
    elif isinstance(snapshot_records, dict):
        snapshot_records = list(snapshot_records.values())
    return normalize_records(snapshot_records=snapshot_records, archive_rows=archive_rows)


def _saved_pairs(db_path, pair_dir, smart_dir):
    from double_revisions import load_saved_pairs
    from history_store import connect
    conn = None
    try:
        if db_path and db_path != ':memory:' and Path(db_path).exists():
            conn = connect(db_path)
        return load_saved_pairs(conn=conn, pair_dir=pair_dir, smart_dir=smart_dir)
    finally:
        if conn is not None:
            conn.close()


def _fingerprint(records, pairs, model_version):
    # Whitelist forecast identity fields. Actual results, API keys, and callback
    # representations cannot enter either the digest or the progress document.
    fields = ('provider', 'fixture_id', 'original_id', 'kickoff_utc', 'kickoff',
              'captured_at_utc', 'saved_at', 'original_captured_at_utc',
              'original_model', 'forecast_model', 'model_version', 'predicted_score',
              'original_score', 'source', 'source_id', 'snapshot_key', 'pair_id',
              'first_fixture_id', 'second_fixture_id', 'saved_at_utc', 'pair_key',
              'original_models', 'original_five_doubles', 'first_provider', 'second_provider',
              'first_kickoff_utc', 'second_kickoff_utc')
    def identities(rows):
        return sorted([{key: row[key] for key in fields if key in row}
                       for row in rows], key=lambda row: json.dumps(row, sort_keys=True, default=str))
    payload = {'model_version': model_version, 'records': identities(records),
               'pairs': identities(pairs)}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def _date(value):
    try:
        parsed = value if isinstance(value, datetime) else datetime.fromisoformat(str(value).replace('Z', '+00:00'))
        return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed
    except (ValueError, TypeError):
        return None


def _pair_cutoff_records(records, pairs, now):
    """Reconstruct both legs of historical pairs at their shared original cutoff.

    A pair whose first leg has finished can still have a future second leg.
    Reusing today's future-leg forecast would leak information acquired after
    the original pair was saved. Separate context records retain each original
    single forecast while requesting that earlier, pair-specific cutoff.
    """
    indexed = {(str(row.get('provider')), str(row.get('fixture_id'))): row
               for row in records}
    derived = []
    seen = set()
    for pair in pairs:
        captured = _date(pair.get('captured_at_utc'))
        if captured is None or captured > now or pair.get('load_error'):
            continue
        identities = [(str(pair.get(side+'_provider')), str(pair.get(side+'_fixture_id')))
                      for side in ('first', 'second')]
        legs = [indexed.get(identity) for identity in identities]
        if any(leg is None for leg in legs):
            continue
        kickoffs = [_date(pair.get(side+'_kickoff_utc') or leg.get('kickoff_utc'))
                    for side, leg in zip(('first', 'second'), legs)]
        if any(kickoff is None for kickoff in kickoffs) or now < min(kickoffs):
            continue
        context = str(pair.get('pair_key') or pair.get('pair_id') or '')
        if not context:
            continue
        for identity, leg in zip(identities, legs):
            key = (context, identity, captured.isoformat())
            if key in seen:
                continue
            seen.add(key)
            derived.append(dict(leg, force_retrospective=True,
                requested_as_of_utc=captured.isoformat(), comparison_context=context,
                source_ref=leg.get('source')))
    return derived


def _run_single_rescoring(**kwargs):
    from rescore_predictions import run_rescoring
    return run_rescoring(**kwargs)


def _run_double_rescoring(pairs, revisions, now, model_version, output_dir):
    from double_revisions import revise_saved_pairs, persist_double_revisions
    report = revise_saved_pairs(pairs, revisions, now=now, model_version=model_version)
    persist_double_revisions(report, output_dir=output_dir)
    return report


def _run_job(status, records, pairs, api_key, primary_get, now, output_dir, double_output_dir):
    status = dict(status)
    secrets = tuple(value for value in (api_key,) if value)

    def update(**changes):
        status.update(changes, updated_at_utc=_now())
        _write_status(status, output_dir)

    def progress(index, total, record):
        # Do not persist provider error strings: they can include query URLs or
        # other service details. Individual reports retain their safe reasons.
        update(processed=index, input_records=total,
               revised=status['revised'] + int(record.get('status') == 'revised'),
               skipped=status['skipped'] + int(record.get('status') == 'skipped'),
               current_fixture_id=str(record.get('fixture_id', '')),
               current_provider=str(record.get('provider', '')))

    try:
        # Freeze canonical originals before starting provider requests; an app
        # rerun cannot change the evidence set while this job is in flight.
        report = _run_single_rescoring(api_key=api_key, primary_get=primary_get,
            records=records, now=now,
            output_dir=output_dir, progress=progress)
        summary = report.get('summary') or {}
        revisions = report.get('revisions') or []
        skipped = report.get('skipped') or []
        update(processed=summary.get('processed', len(revisions) + len(skipped)),
               revised=summary.get('revised', len(revisions)),
               skipped=summary.get('skipped', len(skipped)),
               upcoming_revisions=summary.get('upcoming_revisions', 0),
               retrospective_comparisons=summary.get('retrospective_comparisons', 0),
               singles_report=str(Path(output_dir) / 'latest_report.json'))
        doubles = _run_double_rescoring(pairs, revisions, now, status['model_version'], double_output_dir)
        double_revisions = doubles.get('revisions') or []
        double_skipped = doubles.get('skipped') or []
        failure_count = status['skipped'] + len(double_skipped)
        update(state='incomplete' if failure_count else 'complete',
               completed_at_utc=_now(), failed=failure_count,
               revised_pairs=len(double_revisions), skipped_pairs=len(double_skipped),
               upcoming_pairs=doubles.get('upcoming_pairs', 0),
               retrospective_pairs=doubles.get('retrospective_pairs', 0),
               doubles_report=str(Path(double_output_dir) / 'latest_report.json'),
               current_fixture_id=None, current_provider=None)
    except Exception as exc:
        # Exception type communicates a retryable failure without persisting
        # credentials, arbitrary provider responses, or the callback closure.
        error_type = type(exc).__name__
        for secret in secrets:
            error_type = error_type.replace(secret, '[redacted]')
        update(state='error', completed_at_utc=_now(),
               unprocessed=max(0, status['input_records']-status['processed']),
               error_type=error_type[:80], error='Rescoring stopped; retry after checking provider access.')
    return dict(status)


def start_or_resume_revision_job(api_key='', primary_get=None, archive_rows=(),
        snapshot_records=None, db_path='predictions.sqlite3', force=False,
        output_dir=OUTPUT, double_output_dir=None,
        pair_dir='data/prematch/double_pairs', smart_dir='data/prematch/smart_double_pairs',
        now=None):
    """Start one versioned archive pass; reruns share it and completed work is cached.

    Automatic runs upgrade older-model originals. ``force=True`` also includes
    current-model forecasts and retries an incomplete pass. Skipped records do
    not cause an automatic retry loop against a quota-limited provider.
    """
    global _ACTIVE, _ACTIVE_DIR
    output_dir = Path(output_dir)
    double_output_dir = (Path(double_output_dir) if double_output_dir is not None
                         else output_dir.parent / 'double_rescores')
    with _LOCK:
        if _ACTIVE is not None and not _ACTIVE.done():
            return get_revision_status(_ACTIVE_DIR)
        model_version = _model_version()
        all_records = _source_records(list(archive_rows), snapshot_records)
        records = all_records
        pairs = _saved_pairs(db_path, pair_dir, smart_dir)
        def current_family(value):
            return str(value or '').startswith(model_version)
        if not force:
            records = [r for r in records if not current_family(r.get('original_model', r.get('forecast_model', r.get('model_version'))))]
            pairs = [p for p in pairs if not current_family(p.get('model_version'))
                     and not (p.get('original_models') and all(current_family(m) for m in p['original_models']))]
        run_time = _date(now) if now is not None else datetime.now(timezone.utc)
        if run_time is None:
            raise ValueError('A valid UTC run time is required')
        original_count = len(records)
        pair_legs = _pair_cutoff_records(all_records, pairs, run_time)
        records = records + pair_legs
        source_digest = _fingerprint(records, pairs, model_version)
        # Availability changes should retry a previously inaccessible source,
        # without ever persisting or hashing a secret credential value.
        capability = f'{bool(api_key)}:{primary_get is not None}'
        fingerprint = hashlib.sha256((source_digest+capability).encode()).hexdigest()
        previous = get_revision_status(output_dir)
        if not force and previous.get('source_fingerprint') == fingerprint and previous.get('state') in ('complete', 'incomplete', 'error'):
            return previous
        if not records and not pairs:
            status = {'state': 'complete', 'model_version': model_version,
                      'source_fingerprint': fingerprint, 'input_records': 0,
                      'input_original_records': 0, 'derived_pair_leg_records': 0,
                      'processed': 0, 'revised': 0, 'skipped': 0, 'failed': 0,
                      'input_pairs': 0, 'revised_pairs': 0, 'skipped_pairs': 0,
                      'completed_at_utc': _now()}
            _write_status(status, output_dir)
            return status
        status = {'state': 'running', 'model_version': model_version,
                  'source_fingerprint': fingerprint, 'input_records': len(records),
                  'input_original_records': original_count, 'derived_pair_leg_records': len(pair_legs),
                  'processed': 0, 'revised': 0, 'skipped': 0, 'failed': 0,
                  'input_pairs': len(pairs), 'revised_pairs': 0, 'skipped_pairs': 0,
                  'upcoming_revisions': 0, 'retrospective_comparisons': 0,
                  'started_at_utc': _now(), 'updated_at_utc': _now()}
        _write_status(status, output_dir)
        _ACTIVE_DIR = output_dir
        _ACTIVE = _EXECUTOR.submit(_run_job, status, records, pairs, api_key,
            primary_get, run_time, output_dir, double_output_dir)
        return dict(status)
