"""Immutable snapshots and settlement of exact-score doubles (regular full-time score)."""
import json
from datetime import datetime, timezone

def setup(conn):
    conn.execute("""CREATE TABLE IF NOT EXISTS exact_score_doubles (
        snapshot_key TEXT PRIMARY KEY, saved_at TEXT NOT NULL,
        first_id TEXT NOT NULL, second_id TEXT NOT NULL,
        first_name TEXT NOT NULL, second_name TEXT NOT NULL,
        predictions_json TEXT NOT NULL)""")
    conn.commit()

def save(conn, first, second, options):
    """Save the five displayed combinations once; never rewrite old predictions."""
    setup(conn)
    a,b=first['fixture'],second['fixture']
    if a.get('id') is None or b.get('id') is None:
        return False
    # The same fixture pair may have new forecasts on later days.
    date=datetime.now(timezone.utc).date().isoformat()
    key=f"{date}:{a.get('provider')}:{a['id']}:{b.get('provider')}:{b['id']}"
    entries=[{'score1':o['Score 1'],'score2':o['Score 2'],
              'probability':float(o['Joint'])} for o in options[:5]]
    cur=conn.execute("INSERT OR IGNORE INTO exact_score_doubles VALUES (?,?,?,?,?,?,?)",
                     (key,datetime.now(timezone.utc).isoformat(),str(a['id']),str(b['id']),
                      a['homeTeam']['name']+' vs '+a['awayTeam']['name'],
                      b['homeTeam']['name']+' vs '+b['awayTeam']['name'],json.dumps(entries)))
    conn.commit()
    return bool(cur.rowcount)

def settle(conn, finished):
    """finished maps provider-qualified fixture IDs to final regulation/full-time scores."""
    setup(conn)
    output=[]
    for key,saved,a,b,an,bn,raw in conn.execute(
        "SELECT snapshot_key,saved_at,first_id,second_id,first_name,second_name,predictions_json FROM exact_score_doubles ORDER BY saved_at DESC"):
        # The key contains provider names, so a score from another API cannot settle it.
        pieces=key.split(':')
        provider1=pieces[1] if len(pieces)>4 else ''
        provider2=pieces[3] if len(pieces)>4 else ''
        score1=finished.get((provider1,a))
        score2=finished.get((provider2,b))
        for rank,o in enumerate(json.loads(raw),1):
            status=('Pending' if score1 is None or score2 is None else
                    'Won' if o['score1']==score1 and o['score2']==score2 else 'Lost')
            output.append({'Saved UTC':saved[:16],'Double':rank,'Match 1':an,
                           'Prediction 1':o['score1'],'Final 1':score1 or '—',
                           'Match 2':bn,'Prediction 2':o['score2'],
                           'Final 2':score2 or '—','Status':status,
                           'Estimated probability':o['probability']})
    return output
