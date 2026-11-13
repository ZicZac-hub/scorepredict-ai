"""Immutable exact-score double snapshots and provider-qualified settlement."""
import hashlib
import json
import math
import re
from datetime import datetime, timezone
from smart_double_ai import same_fixture

KNOWN_PROVIDERS = {'football-data.org', 'API-Football'}


def _date(value):
    try:
        result = datetime.fromisoformat(str(value).replace('Z', '+00:00'))
        return result.replace(tzinfo=timezone.utc) if result.tzinfo is None else result
    except (ValueError, TypeError):
        return None


def setup(conn):
    # Serialize additive schema upgrades across simultaneous Streamlit sessions.
    if not conn.in_transaction:
        conn.execute('BEGIN IMMEDIATE')
    conn.execute("""CREATE TABLE IF NOT EXISTS exact_score_doubles (
        snapshot_key TEXT PRIMARY KEY, saved_at TEXT NOT NULL,
        first_id TEXT NOT NULL, second_id TEXT NOT NULL,
        first_name TEXT NOT NULL, second_name TEXT NOT NULL,
        predictions_json TEXT NOT NULL)""")
    columns = {row[1] for row in conn.execute('PRAGMA table_info(exact_score_doubles)')}
    additions = {'first_provider': 'TEXT', 'second_provider': 'TEXT',
                 'first_kickoff': 'TEXT', 'second_kickoff': 'TEXT',
                 'first_model': 'TEXT', 'second_model': 'TEXT'}
    for name, definition in additions.items():
        if name not in columns:
            conn.execute(f'ALTER TABLE exact_score_doubles ADD COLUMN {name} {definition}')
    conn.commit()


def save(conn, first, second, options, now=None):
    """Freeze five displayed alternatives before both kickoffs, once per model/day."""
    setup(conn)
    now = _date(now) if now is not None else datetime.now(timezone.utc)
    a, b = first['fixture'], second['fixture']
    provider1 = str(a.get('provider') or 'football-data.org')
    provider2 = str(b.get('provider') or 'football-data.org')
    kickoff1 = _date(first.get('kickoff') or a.get('utcDate'))
    kickoff2 = _date(second.get('kickoff') or b.get('utcDate'))
    if (now is None or a.get('id') is None or b.get('id') is None
            or provider1 not in KNOWN_PROVIDERS or provider2 not in KNOWN_PROVIDERS
            or same_fixture(first, second)
            or not kickoff1 or not kickoff2 or now >= min(kickoff1, kickoff2)):
        return False
    model1 = str((first.get('result') or {}).get('model_version') or 'unknown')
    model2 = str((second.get('result') or {}).get('model_version') or 'unknown')
    identity = [now.date().isoformat(), provider1, str(a['id']), provider2, str(b['id']), model1, model2]
    key = 'v2-' + hashlib.sha256(json.dumps(identity).encode()).hexdigest()
    entries = [{'score1': o['Score 1'], 'score2': o['Score 2'],
                'probability': float(o['Joint'])} for o in options[:5]]
    if (not entries or len({(x['score1'],x['score2']) for x in entries}) != len(entries)
            or any(not re.fullmatch(r'\d+-\d+', str(x[side])) for x in entries for side in ('score1','score2'))
            or any(not math.isfinite(x['probability']) or not 0 <= x['probability'] <= 1 for x in entries)
            or sum(x['probability'] for x in entries) > 1.0+1e-8):
        return False
    cur = conn.execute("""INSERT OR IGNORE INTO exact_score_doubles
        (snapshot_key,saved_at,first_id,second_id,first_name,second_name,predictions_json,
         first_provider,second_provider,first_kickoff,second_kickoff,first_model,second_model)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (key, now.isoformat(), str(a['id']), str(b['id']),
         a['homeTeam']['name']+' vs '+a['awayTeam']['name'],
         b['homeTeam']['name']+' vs '+b['awayTeam']['name'], json.dumps(entries),
         provider1, provider2, kickoff1.isoformat(), kickoff2.isoformat(), model1, model2))
    conn.commit()
    return bool(cur.rowcount)


def saved_pairs(conn):
    """Return originals without guessing a provider for legacy ``None`` keys."""
    setup(conn)
    cursor = conn.execute('SELECT * FROM exact_score_doubles ORDER BY saved_at DESC')
    names = [item[0] for item in cursor.description]
    output = []
    for values in cursor.fetchall():
        row = dict(zip(names, values))
        pieces = row['snapshot_key'].split(':')
        old1 = pieces[1] if len(pieces) == 5 else None
        old2 = pieces[3] if len(pieces) == 5 else None
        row['first_provider'] = row['first_provider'] or (old1 if old1 in KNOWN_PROVIDERS else 'legacy-unknown')
        row['second_provider'] = row['second_provider'] or (old2 if old2 in KNOWN_PROVIDERS else 'legacy-unknown')
        output.append(row)
    return output


def settle(conn, finished):
    """Settle only provider-qualified outcomes; ambiguous old sources stay pending."""
    output = []
    for row in saved_pairs(conn):
        score1 = finished.get((row['first_provider'], row['first_id']))
        score2 = finished.get((row['second_provider'], row['second_id']))
        if row['first_provider'] not in KNOWN_PROVIDERS or row['second_provider'] not in KNOWN_PROVIDERS:
            score1 = score2 = None
        for rank, option in enumerate(json.loads(row['predictions_json']), 1):
            status = ('Pending' if score1 is None or score2 is None else
                      'Won' if option['score1'] == score1 and option['score2'] == score2 else 'Lost')
            output.append({'Saved UTC': row['saved_at'][:16], 'Double': rank,
                           'Match 1': row['first_name'], 'Prediction 1': option['score1'],
                           'Final 1': score1 or '—', 'Match 2': row['second_name'],
                           'Prediction 2': option['score2'], 'Final 2': score2 or '—',
                           'Status': status, 'Estimated probability': option['probability']})
    return output
