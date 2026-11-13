"""Immutable, provider-qualified forecasts and identity-checked settlements.

Legacy rows remain available for export. Their old results are preserved but
excluded from accuracy until the fixture identity is verified again.
"""
import json
import sqlite3
from datetime import datetime, timezone

LEGACY_PROVIDER = 'legacy-unknown'
SCHEMA = '''CREATE TABLE IF NOT EXISTS predictions (
 provider TEXT NOT NULL, fixture_id INTEGER NOT NULL,
 saved_at TEXT NOT NULL, kickoff TEXT, league TEXT,
 home TEXT, away TEXT, home_id TEXT, away_id TEXT,
 predicted_score TEXT, home_prob REAL, draw_prob REAL, away_prob REAL,
 scores_json TEXT, actual_score TEXT, verification_status TEXT NOT NULL DEFAULT 'pending',
 verified_at TEXT, legacy_actual_score TEXT,
 PRIMARY KEY (provider, fixture_id));'''


def _time(value):
    try:
        result = datetime.fromisoformat(str(value).replace('Z', '+00:00'))
        return result.replace(tzinfo=timezone.utc) if result.tzinfo is None else result
    except (ValueError, TypeError):
        return None


def _provider(match):
    # football-data.org supplies its native schema without a provider field.
    return str(match.get('provider') or 'football-data.org').strip()


def _name(value):
    return ' '.join(str(value or '').casefold().split())


def _migrate(conn):
    columns = {row[1] for row in conn.execute('PRAGMA table_info(predictions)')}
    if not columns:
        conn.execute(SCHEMA)
        conn.commit()
        return
    if 'provider' in columns:
        return
    # Do not infer a provider from a number: ID ranges overlap across APIs.
    # Keep the complete original table as well as the previous result per row.
    with conn:
        conn.execute('BEGIN IMMEDIATE')
        # Another Streamlit session may have completed the migration while
        # this connection waited for the write lock. Never migrate that
        # already-qualified table using a stale pre-lock schema observation.
        columns = {row[1] for row in conn.execute('PRAGMA table_info(predictions)')}
        if 'provider' in columns:
            return
        backup = 'predictions_legacy_v1'
        suffix = 1
        while conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (backup,)).fetchone():
            suffix += 1
            backup = f'predictions_legacy_v1_{suffix}'
        conn.execute(f'ALTER TABLE predictions RENAME TO {backup}')
        conn.execute(SCHEMA)
        conn.execute(f'''INSERT INTO predictions
            (provider,fixture_id,saved_at,kickoff,league,home,away,predicted_score,
             home_prob,draw_prob,away_prob,scores_json,verification_status,legacy_actual_score)
            SELECT ?,fixture_id,saved_at,kickoff,league,home,away,predicted_score,
                   home_prob,draw_prob,away_prob,scores_json,'legacy_unverified',actual_score
            FROM {backup}''', (LEGACY_PROVIDER,))


def connect(path='predictions.sqlite3'):
    conn = sqlite3.connect(path)
    try:
        _migrate(conn)
    except Exception:
        conn.close()
        raise
    return conn


def save_predictions(conn, rows):
    conn.row_factory = sqlite3.Row
    saved = 0
    now = datetime.now(timezone.utc)
    for row in rows:
        match, forecast = row['fixture'], row['result']
        kickoff = _time(row.get('kickoff'))
        if not forecast or match.get('id') is None or not kickoff or kickoff <= now:
            continue
        home, away = match.get('homeTeam') or {}, match.get('awayTeam') or {}
        if home.get('id') is None or away.get('id') is None or home['id'] == away['id']:
            continue
        scores = forecast.get('scores') or []
        if not scores:
            continue
        # Recover source identity before creating a later forecast for an
        # already archived match. The original pre-kickoff prediction and its
        # timestamp must remain the accuracy baseline after migration.
        legacy = conn.execute('SELECT * FROM predictions WHERE provider=? AND fixture_id=?',
                              (LEGACY_PROVIDER, match['id'])).fetchone()
        candidate = dict(match, utcDate=match.get('utcDate') or row['kickoff'])
        if legacy is not None and identity_matches(legacy, candidate):
            conn.execute('''UPDATE predictions SET provider=?,home_id=?,away_id=?
                WHERE provider=? AND fixture_id=?''',
                (_provider(match), str(home['id']), str(away['id']), LEGACY_PROVIDER, match['id']))
            continue
        top = scores[0]
        values = (_provider(match), match['id'], now.isoformat(), row['kickoff'], row.get('league'),
                  home.get('name'), away.get('name'), str(home['id']), str(away['id']),
                  f'{top[0]}-{top[1]}', forecast['home_win'], forecast['draw'],
                  forecast['away_win'], json.dumps(scores))
        cur = conn.execute('''INSERT OR IGNORE INTO predictions
            (provider,fixture_id,saved_at,kickoff,league,home,away,home_id,away_id,
             predicted_score,home_prob,draw_prob,away_prob,scores_json)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)''', values)
        saved += cur.rowcount
    conn.commit()
    return saved


def _same_fixture(saved, match, legacy=False):
    saved_time, actual_time = _time(saved['kickoff']), _time(match.get('utcDate'))
    if not saved_time or not actual_time or saved_time != actual_time:
        return False
    home, away = match.get('homeTeam') or {}, match.get('awayTeam') or {}
    if home.get('id') is None or away.get('id') is None or home['id'] == away['id']:
        return False
    if legacy:
        return (bool(_name(saved['home'])) and bool(_name(saved['away']))
                and _name(saved['home']) == _name(home.get('name'))
                and _name(saved['away']) == _name(away.get('name')))
    return (str(saved['home_id']) == str(home['id'])
            and str(saved['away_id']) == str(away['id']))



def identity_matches(saved, match):
    """Whether a provider response identifies this exact archived fixture.

    Legacy rows can be reidentified by fixture number, ordered team names, and
    exact kickoff. Known rows additionally require the original provider and
    provider-scoped team IDs. No score or prediction values are consulted.
    """
    saved = dict(saved)
    if saved.get('fixture_id') is None or match.get('id') is None:
        return False
    if str(saved['fixture_id']) != str(match['id']):
        return False
    legacy = saved.get('provider') == LEGACY_PROVIDER
    provider = _provider(match)
    if provider == LEGACY_PROVIDER or (not legacy and saved.get('provider') != provider):
        return False
    return _same_fixture(saved, match, legacy=legacy)


def reconcile(conn, finished_matches):
    count = 0
    conn.row_factory = sqlite3.Row
    for match in finished_matches:
        if match.get('status') != 'FINISHED' or match.get('id') is None:
            continue
        score = match.get('score') or {}
        if score.get('duration') not in (None, 'REGULAR'):
            # Never silently settle a regulation-time forecast from extra time.
            continue
        full = score.get('fullTime') or {}
        if any(type(full.get(side)) is not int or full[side] < 0 for side in ('home', 'away')):
            continue
        provider = _provider(match)
        if not provider or provider == LEGACY_PROVIDER:
            continue
        saved = conn.execute('SELECT * FROM predictions WHERE provider=? AND fixture_id=?',
                             (provider, match['id'])).fetchone()
        if saved is None:
            saved = conn.execute('SELECT * FROM predictions WHERE provider=? AND fixture_id=?',
                                 (LEGACY_PROVIDER, match['id'])).fetchone()
        if saved is None or saved['verification_status'] == 'verified':
            continue
        if not identity_matches(saved, match):
            continue
        cur = conn.execute('''UPDATE predictions SET provider=?,home_id=?,away_id=?,
            actual_score=?,verification_status='verified',verified_at=?
            WHERE provider=? AND fixture_id=? AND verification_status != 'verified' ''',
            (provider, str(match['homeTeam']['id']), str(match['awayTeam']['id']),
             f"{full['home']}-{full['away']}", datetime.now(timezone.utc).isoformat(),
             saved['provider'], match['id']))
        count += cur.rowcount
    conn.commit()
    return count


def load(conn):
    conn.row_factory = sqlite3.Row
    return [dict(row) for row in conn.execute('SELECT * FROM predictions ORDER BY kickoff DESC,provider')]
