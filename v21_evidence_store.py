"""Private append-only PostgreSQL storage for validated V21 provider packets.

No secrets or paid-model prompts are stored. A caller must check owner
authentication before invoking. Source snapshots remain accessible only
through the authenticated website; never copied to public GitHub data.
"""
import json

from v21_provider_evidence import valid_snapshot

TABLE="""CREATE TABLE IF NOT EXISTS ziczac_v21_source_packets (
  snapshot_sha256 TEXT PRIMARY KEY,
  provider TEXT NOT NULL,
  fixture_id TEXT NOT NULL,
  captured_at_utc TEXT NOT NULL,
  payload_json TEXT NOT NULL
)"""


def persist(packet, fixture, database_url):
    if not database_url:
        return {"durable":False,"state":"database_not_configured"}
    if not valid_snapshot(packet,fixture):
        return {"durable":False,"state":"invalid_source_snapshot"}
    import psycopg
    try:
        with psycopg.connect(database_url,connect_timeout=7) as conn:
            with conn.cursor() as cursor:
                cursor.execute(TABLE)
                cursor.execute(
                    """INSERT INTO ziczac_v21_source_packets
                       (snapshot_sha256,provider,fixture_id,captured_at_utc,payload_json)
                       VALUES (%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING""",
                    (packet["snapshot_sha256"],packet["provider"],
                     packet["fixture_id"],packet["captured_at_utc"],
                     json.dumps(packet,sort_keys=True,separators=(",",":"),
                                allow_nan=False)))
                return {"durable":True,"state":"saved" if cursor.rowcount else "original_preserved"}
    except Exception as exc:
        return {"durable":False,"state":"database_unavailable",
                "error_type":type(exc).__name__}


def earliest(fixture,database_url,before=None):
    """The first valid matching snapshot, never a chosen-after-the-fact one."""
    if not database_url:
        return None
    import psycopg
    try:
        with psycopg.connect(database_url,connect_timeout=7) as conn:
            with conn.cursor() as cursor:
                cursor.execute(TABLE)
                cursor.execute(
                    """SELECT payload_json FROM ziczac_v21_source_packets
                       WHERE provider=%s AND fixture_id=%s
                       ORDER BY captured_at_utc ASC""",
                    (fixture.get("provider"),str(fixture.get("id"))))
                for (value,) in cursor.fetchall():
                    packet=json.loads(value)
                    if valid_snapshot(packet,fixture,now=before):
                        return packet
    except Exception:
        pass
    return None


def latest(fixture,database_url,before=None):
    """Newest valid pre-match packet for a *new* forecast, append-only history.

    Earlier immutable model predictions are still tied to the actual
    evidence SHA they used; this never rewrites those decisions. A newer
    packet is accepted only if independently verified before kickoff.
    """
    if not database_url:
        return None
    import psycopg
    try:
        with psycopg.connect(database_url,connect_timeout=7) as conn:
            with conn.cursor() as cursor:
                cursor.execute(TABLE)
                cursor.execute(
                    """SELECT payload_json FROM ziczac_v21_source_packets
                       WHERE provider=%s AND fixture_id=%s
                       ORDER BY captured_at_utc DESC,snapshot_sha256 DESC""",
                    (fixture.get("provider"),str(fixture.get("id"))))
                for (value,) in cursor.fetchall():
                    packet=json.loads(value)
                    if valid_snapshot(packet,fixture,now=before):
                        return packet
    except Exception:
        pass
    return None
