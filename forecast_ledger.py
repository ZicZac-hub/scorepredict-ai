"""Append-only prospective forecasts and an explicit legacy-archive backup.

PostgreSQL is the central durable backend. SQLite is a local fallback, and the
file ledger can be committed by the repository workflow. Neither fallback is
reported as a configured central database. Existing application tables are not
modified by this module.
"""
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sqlite3
import tempfile
import threading

FILE_ROOT = Path('data/prospective_ledger')
_SECRET_KEYS = {'api_key', 'apikey', 'authorization', 'password', 'database_url',
                'dsn', 'access_token', 'refresh_token', 'service_role_key'}


class LedgerError(RuntimeError):
    pass


class LedgerValidationError(LedgerError, ValueError):
    pass


class ImmutableConflict(LedgerError):
    pass


class LedgerUnavailableError(LedgerError):
    pass


def _now():
    return datetime.now(timezone.utc).isoformat(timespec='microseconds').replace('+00:00', 'Z')


def _utc(value, field):
    try:
        date = datetime.fromisoformat(str(value).replace('Z', '+00:00'))
        if date.tzinfo is None:
            raise ValueError()
        return date.astimezone(timezone.utc).isoformat(timespec='microseconds').replace('+00:00', 'Z')
    except (ValueError, TypeError):
        raise LedgerValidationError(f'{field} must be an aware UTC timestamp') from None


def _text(value, field):
    if value is None or isinstance(value, bool) or not str(value).strip():
        raise LedgerValidationError(f'{field} is required')
    return str(value).strip()


def _identity(record):
    provider = _text(record.get('provider'), 'provider')
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,63}', provider):
        raise LedgerValidationError('Invalid provider identity')
    fixture = _text(record.get('fixture_id'), 'fixture_id')
    if not fixture.isdigit() or int(fixture) <= 0:
        raise LedgerValidationError('fixture_id must be a positive provider-scoped integer')
    return provider, str(int(fixture))


def _check_secrets(value):
    if isinstance(value, dict):
        for key, item in value.items():
            if str(key).lower().replace('-', '_') in _SECRET_KEYS:
                raise LedgerValidationError('Credentials cannot be stored in the ledger')
            _check_secrets(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _check_secrets(item)


def _canonical(record):
    _check_secrets(record)
    try:
        return json.dumps(record, sort_keys=True, separators=(',', ':'), allow_nan=False)
    except (ValueError, TypeError):
        raise LedgerValidationError('Ledger records must contain finite plain JSON values') from None


def _plain(record):
    if not isinstance(record, dict):
        raise LedgerValidationError('A ledger record must be a dictionary')
    return json.loads(_canonical(record))


def _asof(record):
    value = record.get('evidence_as_of_utc', record.get('as_of_utc', record.get('history_as_of_utc')))
    return _utc(value, 'evidence_as_of_utc')


def forecast_id(record):
    provider, fixture = _identity(record)
    identity = [provider, fixture, _text(record.get('model_version', record.get('model_key')), 'model_version'),
                _asof(record), _text(record.get('horizon', 'rolling'), 'horizon')]
    return hashlib.sha256(_canonical(identity).encode()).hexdigest()


def _forecast(record):
    row = _plain(record)
    provider, fixture = _identity(row)
    if row.get('mode', 'prospective') != 'prospective':
        raise LedgerValidationError('Retrospective records belong outside the prospective ledger')
    if any(key in row for key in ('actual_score', 'home_goals', 'away_goals', 'original_hit', 'updated_hit')):
        raise LedgerValidationError('Actual results cannot be included in a prospective forecast')
    row['mode'] = 'prospective'
    row['provider'], row['fixture_id'] = provider, fixture
    row['model_version'] = _text(row.get('model_version', row.get('model_key')), 'model_version')
    row['horizon'] = _text(row.get('horizon', 'rolling'), 'horizon')
    row['evidence_as_of_utc'] = _asof(row)
    row['captured_at_utc'] = _utc(row.get('captured_at_utc'), 'captured_at_utc')
    row['kickoff_utc'] = _utc(row.get('kickoff_utc'), 'kickoff_utc')
    if not row['evidence_as_of_utc'] <= row['captured_at_utc'] < row['kickoff_utc']:
        raise LedgerValidationError('Forecast evidence and capture must precede kickoff')
    for side in ('home', 'away'):
        row[side+'_team_id'] = _text(row.get(side+'_team_id'), side+'_team_id')
    if row['home_team_id'] == row['away_team_id']:
        raise LedgerValidationError('The two teams must have distinct provider-scoped IDs')
    _text(row.get('input_digest'), 'input_digest')
    matrix = row.get('score_matrix')
    if not isinstance(matrix, list) or not matrix or not isinstance(matrix[0], list) or not matrix[0]:
        raise LedgerValidationError('The complete score_matrix is required')
    width = len(matrix[0])
    if max(len(matrix), width) > 256:
        raise LedgerValidationError('Score matrix support is unexpectedly large')
    for line in matrix:
        if not isinstance(line, list) or len(line) != width:
            raise LedgerValidationError('The score matrix must be rectangular')
        if any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in line):
            raise LedgerValidationError('Score probabilities must be finite and nonnegative')
    mass = math.fsum(v for line in matrix for v in line)
    if abs(mass-1.0) > 1e-6:
        raise LedgerValidationError('The complete score matrix must sum to one')
    if row.get('matrix_shape') is not None and row['matrix_shape'] != [len(matrix), width]:
        raise LedgerValidationError('Score matrix dimensions disagree with metadata')
    if row.get('matrix_mass') is not None and abs(float(row['matrix_mass'])-mass) > 1e-6:
        raise LedgerValidationError('Score matrix mass disagrees with metadata')
    row['matrix_shape'], row['matrix_mass'] = [len(matrix), width], mass
    probabilities = row.get('outcome_probabilities')
    if probabilities is not None:
        derived = {'home': math.fsum(value for i, line in enumerate(matrix)
                                     for j, value in enumerate(line) if i > j),
                   'draw': math.fsum(matrix[i][i] for i in range(min(len(matrix), width))),
                   'away': math.fsum(value for i, line in enumerate(matrix)
                                     for j, value in enumerate(line) if i < j)}
        if (not isinstance(probabilities, dict) or any(
                type(probabilities.get(key)) not in (int, float)
                or not math.isfinite(probabilities[key])
                or abs(probabilities[key]-value) > 1e-6 for key, value in derived.items())):
            raise LedgerValidationError('Outcome probabilities disagree with the complete score grid')
    for candidate in row.get('top10_scores') or []:
        if (not isinstance(candidate, list) or len(candidate) != 3
                or type(candidate[0]) is not int or type(candidate[1]) is not int
                or not 0 <= candidate[0] < len(matrix) or not 0 <= candidate[1] < width
                or type(candidate[2]) not in (int, float) or not math.isfinite(candidate[2])
                or abs(candidate[2]-matrix[candidate[0]][candidate[1]]) > 1e-9):
            raise LedgerValidationError('Top score metadata disagrees with the complete score grid')
    expected = forecast_id(row)
    given = row.get('forecast_id', row.get('record_id'))
    if given is not None and given != expected:
        raise LedgerValidationError('Forecast ID does not match its immutable identity')
    row['forecast_id'] = expected
    return row


def _settlement(record):
    row = _plain(record)
    row['provider'], row['fixture_id'] = _identity(row)
    if not (row.get('verified') is True or row.get('verification_status') == 'verified'
            or row.get('evidence_version', 0) == 2):
        raise LedgerValidationError('Only verified provider results can settle forecasts')
    if row.get('score_basis') != 'regulation':
        raise LedgerValidationError('Settlement must use the regulation-time score')
    if row.get('status') not in ('FINISHED', 'FT', 'AET', 'PEN'):
        raise LedgerValidationError('The fixture must be confirmed finished')
    row['kickoff_utc'] = _utc(row.get('kickoff_utc'), 'kickoff_utc')
    row['verified_at_utc'] = _utc(row.get('verified_at_utc'), 'verified_at_utc')
    if row['verified_at_utc'] <= row['kickoff_utc']:
        raise LedgerValidationError('A final score must be verified after kickoff')
    for side in ('home', 'away'):
        row[side+'_team_id'] = _text(row.get(side+'_team_id'), side+'_team_id')
        if type(row.get(side+'_goals')) is not int or row[side+'_goals'] < 0:
            raise LedgerValidationError('Final goals must be nonnegative integers')
    if row['home_team_id'] == row['away_team_id']:
        raise LedgerValidationError('Settlement teams must be distinct')
    identity = [row['provider'], row['fixture_id'], row['kickoff_utc']]
    expected = hashlib.sha256(_canonical(identity).encode()).hexdigest()
    if row.get('settlement_id') is not None and row['settlement_id'] != expected:
        raise LedgerValidationError('Settlement ID does not match fixture identity')
    row['settlement_id'] = expected
    return row


def _settlement_signature(row):
    return tuple(row[key] for key in ('provider', 'fixture_id', 'kickoff_utc', 'home_team_id',
                                     'away_team_id', 'score_basis', 'home_goals', 'away_goals'))


def _matches(forecast, result):
    return all(forecast[key] == result[key] for key in
               ('provider', 'fixture_id', 'kickoff_utc', 'home_team_id', 'away_team_id'))


_SCHEMA = (
    '''CREATE TABLE IF NOT EXISTS forecast_ledger_forecasts (
        forecast_id TEXT PRIMARY KEY, provider TEXT NOT NULL, fixture_id TEXT NOT NULL,
        model_version TEXT NOT NULL, evidence_as_of_utc TEXT NOT NULL, horizon TEXT NOT NULL,
        kickoff_utc TEXT NOT NULL, captured_at_utc TEXT NOT NULL, input_digest TEXT NOT NULL,
        payload_json TEXT NOT NULL, inserted_at_utc TEXT NOT NULL, model_key TEXT,
        numerical_model_version TEXT, policy_version TEXT, pair_id TEXT, eligible INTEGER NOT NULL,
        UNIQUE(provider,fixture_id,model_version,evidence_as_of_utc,horizon))''',
    '''CREATE TABLE IF NOT EXISTS forecast_ledger_settlements (
        settlement_id TEXT PRIMARY KEY, provider TEXT NOT NULL, fixture_id TEXT NOT NULL,
        kickoff_utc TEXT NOT NULL, payload_json TEXT NOT NULL, inserted_at_utc TEXT NOT NULL,
        UNIQUE(provider,fixture_id,kickoff_utc))''',
    '''CREATE TABLE IF NOT EXISTS forecast_archive_snapshots (
        snapshot_id TEXT PRIMARY KEY, kind TEXT NOT NULL, logical_id TEXT NOT NULL,
        provider_scope TEXT NOT NULL, payload_hash TEXT NOT NULL, payload_json TEXT NOT NULL,
        observed_at_utc TEXT NOT NULL)''',
    '''CREATE INDEX IF NOT EXISTS forecast_ledger_fixture_idx
        ON forecast_ledger_forecasts(provider,fixture_id,kickoff_utc)''',
    '''CREATE INDEX IF NOT EXISTS forecast_archive_original_idx
        ON forecast_archive_snapshots(kind,logical_id,provider_scope,observed_at_utc)''',
)


def _archive(record, kind):
    if kind not in ('single', 'double'):
        raise LedgerValidationError('Archive kind must be single or double')
    row = _plain(record)
    if kind == 'single':
        normalize = lambda value: ' '.join(str(value or '').casefold().split())
        identity = [_text(row.get('fixture_id'), 'fixture_id'),
                    _text(row.get('saved_at', row.get('captured_at_utc')), 'saved_at'),
                    _text(row.get('kickoff', row.get('kickoff_utc')), 'kickoff'),
                    normalize(row.get('home', row.get('home_team'))),
                    normalize(row.get('away', row.get('away_team')))]
        provider = str(row.get('provider') or 'legacy-unknown')
        scope = provider if provider not in ('legacy-unknown', 'None', '') else 'legacy-unknown'
    else:
        identity = [_text(row.get('snapshot_key'), 'snapshot_key')]
        scope = ''
    logical = hashlib.sha256(_canonical([kind, identity]).encode()).hexdigest()
    payload = _canonical(row)
    digest = hashlib.sha256(payload.encode()).hexdigest()
    key = hashlib.sha256(_canonical([kind, logical, scope, digest]).encode()).hexdigest()
    return key, kind, logical, scope, digest, payload, _now()


def _earliest_double(versions):
    """Recover original choices by capture time, never by upload order.

    At the earliest capture, later uploads may enrich identity/model metadata
    only when the ordered legs and the original five choices still agree.
    Conflicting versions remain available through archive_versions().
    """
    timed = []
    for version in versions:
        try:
            timed.append((_utc(version['record'].get('saved_at'), 'saved_at'), version))
        except LedgerValidationError:
            continue
    if not timed:
        return versions[0]['record']
    earliest = min(timestamp for timestamp, _ in timed)
    candidates = [version for timestamp, version in timed if timestamp == earliest]
    original = candidates[0]['record']

    def original_identity(record):
        return _canonical([record.get(key) for key in
                           ('first_id', 'second_id', 'first_name', 'second_name', 'predictions_json')])

    signature = original_identity(original)
    eligible = []
    for version in candidates:
        record = version['record']
        if original_identity(record) != signature:
            continue
        # Preserve an already known leg identity. NULL legacy fields may be
        # enriched, but an upload cannot substitute another known provider or
        # kickoff under the same snapshot key.
        compatible = True
        for side in ('first', 'second'):
            for suffix in ('provider', 'kickoff'):
                field = side+'_'+suffix
                value = original.get(field)
                if value not in (None, '', 'None', 'legacy-unknown') and record.get(field) != value:
                    compatible = False
        if compatible:
            eligible.append(record)
    # Two different recovered identities under one formerly opaque snapshot
    # are not permission to choose whichever instance uploaded last.
    unknown = (None, '', 'None', 'legacy-unknown')
    for side in ('first', 'second'):
        for suffix in ('provider', 'kickoff'):
            field = side+'_'+suffix
            if original.get(field) in unknown:
                known = {record.get(field) for record in eligible if record.get(field) not in unknown}
                if len(known) > 1:
                    eligible = [record for record in eligible if record.get(field) in unknown]
    return eligible[-1] if eligible else original


class LedgerStore:
    def __init__(self, database_url=None, path='forecast_ledger.sqlite3'):
        self._lock = threading.RLock()
        self._closed = False
        self.backend = 'postgresql' if database_url else 'sqlite'
        self.durable = bool(database_url)
        self._connection = None
        try:
            if database_url:
                import psycopg
                self._connection = psycopg.connect(database_url, connect_timeout=10)
            else:
                if str(path) != ':memory:':
                    Path(path).parent.mkdir(parents=True, exist_ok=True)
                self._connection = sqlite3.connect(str(path), timeout=30, check_same_thread=False)
            with self._transaction():
                for statement in _SCHEMA:
                    self._execute(statement)
        except LedgerError:
            self.close()
            raise
        except Exception as exc:
            self.close()
            raise LedgerUnavailableError(f'Ledger initialization failed ({type(exc).__name__})') from None

    def _sql(self, statement):
        return statement.replace('?', '%s') if self.backend == 'postgresql' else statement

    def _execute(self, statement, parameters=()):
        return self._connection.execute(self._sql(statement), parameters)

    def _many(self, statement, parameters):
        cursor = self._connection.cursor()
        cursor.executemany(self._sql(statement), parameters)
        return max(0, cursor.rowcount)

    @contextmanager
    def _transaction(self):
        with self._lock:
            if self._closed:
                raise LedgerUnavailableError('Ledger is closed')
            try:
                yield
                self._connection.commit()
            except Exception as exc:
                try:
                    self._connection.rollback()
                except Exception:
                    # A lost connection must not replace the safe public
                    # exception with a driver's connection-string diagnostic.
                    pass
                if isinstance(exc, LedgerError):
                    raise
                raise LedgerUnavailableError(f'Ledger operation failed ({type(exc).__name__})') from None

    def _payloads(self, table, id_column, identities):
        output = {}
        identities = list(identities)
        for start in range(0, len(identities), 400):
            group = identities[start:start+400]
            statement = f'SELECT {id_column},payload_json FROM {table} WHERE {id_column} IN ('+','.join('?' for _ in group)+')'
            output.update(self._execute(statement, group).fetchall())
        return output

    def put_forecast(self, record):
        return bool(self.put_forecasts([record]))

    def put_forecasts(self, records):
        prepared = {}
        for record in records:
            row = _forecast(record)
            payload = _canonical(row)
            key = row['forecast_id']
            if key in prepared and prepared[key][1] != payload:
                raise ImmutableConflict('Forecast identity has conflicting payloads in this batch')
            prepared[key] = row, payload
        if not prepared:
            return 0
        with self._transaction():
            existing = self._payloads('forecast_ledger_forecasts', 'forecast_id', prepared)
            if any(payload != prepared[key][1] for key, payload in existing.items()):
                raise ImmutableConflict('An immutable forecast already exists with different contents')
            values = [(key, row['provider'], row['fixture_id'], row['model_version'], row['evidence_as_of_utc'],
                       row['horizon'], row['kickoff_utc'], row['captured_at_utc'], row['input_digest'], payload, _now(),
                       row.get('model_key'), row.get('numerical_model_version'), row.get('policy_version'),
                       row.get('pair_id'), int(row.get('eligible_for_comparison') is True))
                      for key, (row, payload) in prepared.items() if key not in existing]
            count = self._many('''INSERT INTO forecast_ledger_forecasts VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                                  ON CONFLICT DO NOTHING''', values) if values else 0
            # Recheck after concurrent inserts; an ON CONFLICT must never hide
            # a different forecast published by another process.
            actual = self._payloads('forecast_ledger_forecasts', 'forecast_id', prepared)
            if any(actual.get(key) != payload for key, (_, payload) in prepared.items()):
                raise ImmutableConflict('A concurrent forecast conflicts with this immutable identity')
            return count

    def iter_forecasts(self):
        with self._transaction():
            return [json.loads(row[0]) for row in self._execute(
                'SELECT payload_json FROM forecast_ledger_forecasts ORDER BY captured_at_utc,forecast_id').fetchall()]

    def iter_forecast_metadata(self):
        """Read compact capture slots without transferring historical score grids."""
        fields = ('provider', 'fixture_id', 'model_key', 'model_version',
                  'numerical_model_version', 'evidence_as_of_utc', 'horizon', 'policy_version',
                  'pair_id', 'input_digest', 'kickoff_utc', 'captured_at_utc', 'eligible')
        slot = ("payload_json::jsonb->>'slot_key'" if self.backend == 'postgresql'
                else "json_extract(payload_json,'$.slot_key')")
        with self._transaction():
            rows = self._execute('SELECT '+','.join(fields)+','+slot+
                                 ' FROM forecast_ledger_forecasts ORDER BY captured_at_utc,forecast_id').fetchall()
        output = [dict(zip(fields+('slot_key',), row)) for row in rows]
        for row in output:
            row['eligible_for_comparison'] = bool(row.pop('eligible'))
        return output

    def put_settlement(self, record):
        return bool(self.put_settlements([record]))

    def put_settlements(self, records):
        prepared = {}
        for record in records:
            row = _settlement(record)
            key = row['settlement_id']
            if key in prepared and _settlement_signature(prepared[key]) != _settlement_signature(row):
                raise ImmutableConflict('Conflicting verified scores in the settlement batch')
            prepared[key] = row
        if not prepared:
            return 0
        with self._transaction():
            for row in prepared.values():
                forecasts = [json.loads(item[0]) for item in self._execute(
                    '''SELECT payload_json FROM forecast_ledger_forecasts
                       WHERE provider=? AND fixture_id=? AND kickoff_utc=?''',
                    (row['provider'], row['fixture_id'], row['kickoff_utc'])).fetchall()]
                if row.get('forecast_id'):
                    forecasts = [f for f in forecasts if f['forecast_id'] == row['forecast_id']]
                if not forecasts or not any(_matches(forecast, row) for forecast in forecasts):
                    raise LedgerValidationError('Result does not match an immutable forecast identity')
            existing = self._payloads('forecast_ledger_settlements', 'settlement_id', prepared)
            for key, payload in existing.items():
                if _settlement_signature(json.loads(payload)) != _settlement_signature(prepared[key]):
                    raise ImmutableConflict('A verified settlement already exists with a different score')
            values = [(key, row['provider'], row['fixture_id'], row['kickoff_utc'], _canonical(row), _now())
                      for key, row in prepared.items() if key not in existing]
            count = self._many('''INSERT INTO forecast_ledger_settlements VALUES (?,?,?,?,?,?)
                                  ON CONFLICT DO NOTHING''', values) if values else 0
            actual = self._payloads('forecast_ledger_settlements', 'settlement_id', prepared)
            if any(_settlement_signature(json.loads(actual[key])) != _settlement_signature(row)
                   for key, row in prepared.items()):
                raise ImmutableConflict('A concurrent settlement conflicts with this verified score')
            return count

    def get_settlements(self):
        with self._transaction():
            return [json.loads(row[0]) for row in self._execute(
                'SELECT payload_json FROM forecast_ledger_settlements ORDER BY kickoff_utc,settlement_id').fetchall()]

    def put_archive_record(self, record, kind='single'):
        return bool(self.put_archive_records([record], kind))

    def put_archive_records(self, records, kind='single'):
        prepared = {}
        for record in records:
            value = _archive(record, kind)
            prepared[value[0]] = value
        if not prepared:
            return 0
        with self._transaction():
            return self._many('''INSERT INTO forecast_archive_snapshots VALUES (?,?,?,?,?,?,?)
                                 ON CONFLICT DO NOTHING''', list(prepared.values()))

    def archive_versions(self, kind=None):
        if kind is not None and kind not in ('single', 'double'):
            raise LedgerValidationError('Archive kind must be single or double')
        with self._transaction():
            statement = 'SELECT kind,logical_id,provider_scope,payload_json,observed_at_utc FROM forecast_archive_snapshots'
            rows = self._execute(statement+(' WHERE kind=?' if kind else '')+' ORDER BY observed_at_utc,snapshot_id',
                                 (kind,) if kind else ()).fetchall()
            return [{'kind': row[0], 'logical_id': row[1], 'provider_scope': row[2],
                     'record': json.loads(row[3]), 'observed_at_utc': row[4]} for row in rows]

    def iter_archive_records(self, kind='single'):
        versions = self.archive_versions(kind)
        if kind == 'double':
            originals = {}
            for version in versions:
                originals.setdefault(version['logical_id'], []).append(version)
            return [_earliest_double(group) for group in originals.values()]
        groups = {}
        for version in versions:
            groups.setdefault(version['logical_id'], {})[version['provider_scope']] = version['record']
        output = []
        for scopes in groups.values():
            known = [scope for scope in scopes if scope != 'legacy-unknown']
            # A unique provider recovery supersedes only the display of the
            # opaque legacy version. Ambiguous cross-provider originals remain.
            output.extend(record for scope, record in scopes.items()
                          if not (kind == 'single' and scope == 'legacy-unknown' and len(known) == 1))
        return output

    def stats(self):
        with self._transaction():
            counts = {key: self._execute(f'SELECT COUNT(*) FROM {table}').fetchone()[0]
                      for key, table in (('forecasts', 'forecast_ledger_forecasts'),
                                         ('settlements', 'forecast_ledger_settlements'),
                                         ('archive_versions', 'forecast_archive_snapshots'))}
        return dict(counts, backend=self.backend, durable=self.durable,
                    durability='central_postgresql' if self.durable else 'local_only', connected=True)

    status = stats

    def close(self):
        with self._lock:
            if self._connection is not None and not self._closed:
                self._connection.close()
            self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def open_ledger(database_url=None, path='forecast_ledger.sqlite3'):
    if database_url is None:
        database_url = os.environ.get('DATABASE_URL') or os.environ.get('FORECAST_DATABASE_URL')
    return LedgerStore(database_url=database_url or None, path=path)


def _root(root, kind):
    if kind not in ('forecasts', 'settlements'):
        raise LedgerValidationError('File ledger kind must be forecasts or settlements')
    path = Path(root)
    return path.parent if path.name == kind else path


def _file_path(row, root, kind):
    directory = _root(root, kind) / kind
    if kind == 'forecasts':
        return directory / row['provider'] / row['fixture_id'] / (row['forecast_id']+'.json')
    return directory / (row['settlement_id']+'.json')


def append_file_record(record, root=FILE_ROOT, kind='forecasts'):
    row = _forecast(record) if kind == 'forecasts' else _settlement(record)
    path = _file_path(row, root, kind)
    if kind == 'settlements':
        forecasts = read_file_records(_root(root, kind), 'forecasts', provider=row['provider'], fixture_id=row['fixture_id'])
        if not any(_matches(forecast, row) for forecast in forecasts):
            raise LedgerValidationError('File settlement has no matching immutable forecast')
    payload = _canonical(row)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix='.ledger-', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(descriptor, 'w', encoding='utf-8') as handle:
            handle.write(payload+'\n')
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
            return True
        except FileExistsError:
            try:
                existing = json.loads(path.read_text(encoding='utf-8'))
            except (OSError, ValueError):
                raise ImmutableConflict('Existing immutable file cannot be read safely') from None
            equivalent = (_canonical(existing) == payload if kind == 'forecasts' else
                          _settlement_signature(existing) == _settlement_signature(row))
            if not equivalent:
                raise ImmutableConflict('An immutable ledger file already has different contents')
            return False
    finally:
        Path(temporary).unlink(missing_ok=True)


def read_file_records(root=FILE_ROOT, kind='forecasts', provider=None, fixture_id=None):
    directory = _root(root, kind) / kind
    if kind == 'forecasts' and provider is not None and fixture_id is not None:
        # Validate identity before using it in a filesystem path.
        source, identifier = _identity({'provider': provider, 'fixture_id': fixture_id})
        directory = directory / source / identifier
    records = []
    for path in sorted(directory.rglob('*.json')):
        try:
            record = json.loads(path.read_text(encoding='utf-8'))
            records.append(_forecast(record) if kind == 'forecasts' else _settlement(record))
        except (OSError, ValueError, TypeError):
            raise LedgerValidationError('An immutable file ledger record is invalid') from None
    return records


def sync_file_ledger(store, root=FILE_ROOT):
    forecasts = read_file_records(root, 'forecasts')
    settlements = read_file_records(root, 'settlements')
    return {'forecasts': store.put_forecasts(forecasts),
            'settlements': store.put_settlements(settlements)}
