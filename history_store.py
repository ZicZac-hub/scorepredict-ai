"""Local prediction snapshots. Local disks may be ephemeral on free cloud hosting."""
import json, sqlite3
from datetime import datetime, timezone

SCHEMA='''CREATE TABLE IF NOT EXISTS predictions (
 fixture_id INTEGER PRIMARY KEY, saved_at TEXT NOT NULL, kickoff TEXT, league TEXT,
 home TEXT, away TEXT, predicted_score TEXT, home_prob REAL, draw_prob REAL,
 away_prob REAL, scores_json TEXT, actual_score TEXT);'''

def connect(path='predictions.sqlite3'):
    conn=sqlite3.connect(path)
    conn.execute(SCHEMA)
    return conn

def save_predictions(conn, rows):
    saved=0
    for r in rows:
        m,p=r['fixture'],r['result']
        if not p or m.get('id') is None:continue
        top=p['scores'][0]
        values=(m['id'],datetime.now(timezone.utc).isoformat(),r['kickoff'],r['league'],
                m['homeTeam'].get('name'),m['awayTeam'].get('name'),f'{top[0]}-{top[1]}',
                p['home_win'],p['draw'],p['away_win'],json.dumps(p['scores']))
        cur=conn.execute('''INSERT OR IGNORE INTO predictions
        (fixture_id,saved_at,kickoff,league,home,away,predicted_score,home_prob,draw_prob,away_prob,scores_json)
        VALUES (?,?,?,?,?,?,?,?,?,?,?)''',values)
        saved+=cur.rowcount
    conn.commit();return saved

def reconcile(conn, finished_matches):
    count=0
    for m in finished_matches:
        if m.get('status')!='FINISHED':continue
        score=(m.get('score') or {}).get('fullTime') or {}
        if not isinstance(score.get('home'),int) or not isinstance(score.get('away'),int):continue
        cur=conn.execute('UPDATE predictions SET actual_score=? WHERE fixture_id=? AND actual_score IS NULL',
                         (f"{score['home']}-{score['away']}",m.get('id')))
        count+=cur.rowcount
    conn.commit();return count

def load(conn):
    conn.row_factory=sqlite3.Row
    return [dict(r) for r in conn.execute('SELECT * FROM predictions ORDER BY kickoff DESC')]
