"""Back up and recover original private archives without creating forecasts.

Opaque archive backups are separate from the prospective comparison ledger:
restoring a historical row never claims that it was captured by the new system.
The caller supplies an authenticated ledger; connection URLs stay outside this
module and never enter status messages or archived prediction payloads.
"""
import unicodedata
from datetime import datetime, timezone

from double_tracker import setup as setup_doubles


SINGLE_COLUMNS = (
    'provider', 'fixture_id', 'saved_at', 'kickoff', 'league', 'home', 'away',
    'home_id', 'away_id', 'predicted_score', 'home_prob', 'draw_prob', 'away_prob',
    'scores_json', 'actual_score', 'verification_status', 'verified_at',
    'legacy_actual_score', 'model_version',
)
DOUBLE_COLUMNS = (
    'snapshot_key', 'saved_at', 'first_id', 'second_id', 'first_name', 'second_name',
    'predictions_json', 'first_provider', 'second_provider', 'first_kickoff',
    'second_kickoff', 'first_model', 'second_model',
)
LEGACY_PROVIDER = 'legacy-unknown'


def _text(value):
    return ' '.join(unicodedata.normalize('NFKC', str(value or '')).casefold().split())


def _time_identity(value):
    try:
        parsed = datetime.fromisoformat(str(value).replace('Z', '+00:00'))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc).isoformat()
    except (TypeError, ValueError):
        return str(value or '')


def _saved_at(row):
    try:
        parsed = datetime.fromisoformat(str(row.get('saved_at')).replace('Z', '+00:00'))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    except (TypeError, ValueError):
        return datetime.max.replace(tzinfo=timezone.utc)


def _original_identity(row):
    """Only identity equality can link a legacy row to a known source.

    A provider-qualified row remains distinct from any other known provider.
    Result values and the predicted score never choose a source identity.
    """
    return (str(row.get('fixture_id')), _time_identity(row.get('saved_at')),
            _time_identity(row.get('kickoff')), _text(row.get('home')),
            _text(row.get('away')))


def _read_rows(conn, table, allowed):
    columns = [row[1] for row in conn.execute(f'PRAGMA table_info({table})')
               if row[1] in allowed]
    if not columns:
        return []
    cursor = conn.execute(f"SELECT {','.join(columns)} FROM {table}")
    return [dict(zip(columns, row)) for row in cursor.fetchall()]


def _single_record(record):
    if not isinstance(record, dict):
        return None
    row = {key: record[key] for key in SINGLE_COLUMNS if key in record}
    if any(value is not None and not isinstance(value, (str, int, float)) for value in row.values()):
        return None
    provider = row.get('provider')
    fixture_id = row.get('fixture_id')
    if (not isinstance(provider, str) or not provider.strip()
            or isinstance(fixture_id, bool)
            or not str(fixture_id or '').isdigit()
            or not isinstance(row.get('saved_at'), str) or not row['saved_at']):
        return None
    row['fixture_id'] = int(fixture_id)
    return row


def _double_record(record):
    if not isinstance(record, dict):
        return None
    row = {key: record[key] for key in DOUBLE_COLUMNS if key in record}
    required = ('snapshot_key', 'saved_at', 'first_id', 'second_id',
                'first_name', 'second_name', 'predictions_json')
    if any(not isinstance(row.get(key), str) or not row[key] for key in required):
        return None
    return row


def _insert_missing(conn, table, row, allowed):
    actual_columns = {entry[1] for entry in conn.execute(f'PRAGMA table_info({table})')}
    columns = [key for key in allowed if key in row and key in actual_columns]
    placeholders = ','.join('?' for _ in columns)
    cursor = conn.execute(
        f"INSERT OR IGNORE INTO {table} ({','.join(columns)}) VALUES ({placeholders})",
        tuple(row[key] for key in columns),
    )
    return cursor.rowcount == 1


def _restore_singles(conn, records, status):
    local = _read_rows(conn, 'predictions', SINGLE_COLUMNS)
    incoming = []
    for record in records:
        row = _single_record(record)
        if row is None:
            status['skipped_invalid'] += 1
        else:
            incoming.append(row)

    # SQLite keeps one original per provider/fixture. A later forecast may
    # have reached central storage first, so restore by actual capture time,
    # not upload order. Known-source recovery wins only an equal-time tie;
    # different providers sharing a number still remain separate originals.
    incoming.sort(key=lambda row: (_saved_at(row), row['provider'] == LEGACY_PROVIDER))
    known = {}
    legacy = set()
    for row in local + incoming:
        if row.get('provider') == LEGACY_PROVIDER:
            continue
        known.setdefault(_original_identity(row), set()).add(row.get('provider'))
    for row in local:
        if row.get('provider') == LEGACY_PROVIDER:
            legacy.add(_original_identity(row))
    local_qualified = {(row.get('provider'), str(row.get('fixture_id'))) for row in local}

    restored = 0
    with conn:
        for row in incoming:
            identity = _original_identity(row)
            provider = row['provider']
            if provider == LEGACY_PROVIDER and len(known.get(identity, ())) == 1:
                status['skipped_legacy_duplicates'] += 1
                continue
            # A live local legacy original already occupies this evidence
            # identity. Leave its provider recovery to normal verification;
            # inserting a second copy here would supersede that original.
            if (provider != LEGACY_PROVIDER and identity in legacy
                    and len(known.get(identity, ())) == 1
                    and (provider, str(row['fixture_id'])) not in local_qualified):
                status['skipped_existing'] += 1
                continue
            if _insert_missing(conn, 'predictions', row, SINGLE_COLUMNS):
                restored += 1
                local_qualified.add((provider, str(row['fixture_id'])))
            else:
                status['skipped_existing'] += 1
    status['restored_singles'] += restored


def _restore_doubles(conn, records, status):
    grouped = {}
    for record in records:
        row = _double_record(record)
        if row is None:
            status['skipped_invalid'] += 1
        else:
            grouped.setdefault(row['snapshot_key'], []).append(row)
    core = ('snapshot_key', 'saved_at', 'first_id', 'second_id', 'first_name',
            'second_name', 'predictions_json')
    originals = []
    for versions in grouped.values():
        original = min(versions, key=_saved_at)
        # Later identity metadata may be retained only when the complete
        # original capture and displayed choices are unchanged. A restart's
        # later prediction with the same snapshot key cannot replace them.
        identical = [row for row in versions
                     if all(row.get(key) == original.get(key) for key in core)]
        originals.append(identical[-1])
    restored = 0
    with conn:
        for row in originals:
            if _insert_missing(conn, 'exact_score_doubles', row, DOUBLE_COLUMNS):
                restored += 1
            else:
                status['skipped_existing'] += 1
    status['restored_doubles'] += restored


def _upload(ledger, rows, kind, status):
    batch = getattr(ledger, 'put_archive_records', None)
    if callable(batch):
        count = batch(rows, kind=kind)
        status['uploaded_'+('singles' if kind == 'single' else 'doubles')] += int(count)
    else:
        for row in rows:
            inserted = ledger.put_archive_record(row, kind=kind)
            status['uploaded_'+('singles' if kind == 'single' else 'doubles')] += int(bool(inserted))


def sync_archive(conn, ledger, *, restore=True, upload=True):
    """Restore missing originals, then append backups of current local rows.

    Call with ``restore=True, upload=False`` before creating today's forecasts
    on a fresh app instance. After ordinary saves and reconciliation, call with
    ``restore=False, upload=True``. The default performs both operations.

    Existing local originals and settlements are never overwritten. A failed
    remote operation leaves local data intact and returns only its exception
    type, never a connection URL or provider error string.
    """
    status = {'state': 'complete', 'backend': 'unknown', 'durable': False,
              'restored_singles': 0, 'restored_doubles': 0,
              'uploaded_singles': 0, 'uploaded_doubles': 0,
              'skipped_existing': 0, 'skipped_invalid': 0,
              'skipped_legacy_duplicates': 0, 'local_singles': 0,
              'local_doubles': 0, 'errors': []}

    def failure(operation, error):
        status['errors'].append({'operation': operation, 'error_type': type(error).__name__})
        status['state'] = 'partial'

    try:
        details = ledger.status()
        if isinstance(details, dict):
            status['backend'] = details.get('backend', 'unknown')
            status['durable'] = bool(details.get('durable', False))
    except Exception as exc:
        failure('ledger_status', exc)
    try:
        setup_doubles(conn)
    except Exception as exc:
        failure('local_schema', exc)
        return status
    if restore:
        for kind, restorer in (('single', _restore_singles), ('double', _restore_doubles)):
            try:
                versions = getattr(ledger, 'archive_versions', None)
                if kind == 'double' and callable(versions):
                    # The latest-per-key view can hide a first forecast after
                    # an old app instance recreated the same daily key.
                    records = [item['record'] for item in versions(kind=kind)]
                else:
                    records = ledger.iter_archive_records(kind=kind)
                restorer(conn, records, status)
            except Exception as exc:
                failure('restore_'+kind, exc)
    for kind, table, columns in (
            ('single', 'predictions', SINGLE_COLUMNS),
            ('double', 'exact_score_doubles', DOUBLE_COLUMNS)):
        try:
            rows = _read_rows(conn, table, columns)
            status['local_'+('singles' if kind == 'single' else 'doubles')] = len(rows)
            if upload:
                _upload(ledger, rows, kind, status)
        except Exception as exc:
            failure('upload_'+kind if upload else 'read_'+kind, exc)
    return status
