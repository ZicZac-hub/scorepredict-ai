"""Shared app persistence: restore originals, freeze paired forecasts, then grade."""
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import engine
import forecast_model
from central_archive import sync_archive
from forecast_ledger import open_ledger, append_file_record, sync_file_ledger
from history_store import save_predictions, reconcile, load
from prospective_ledger import capture_pair, evaluate, horizon_for, pair_slot_key


def _date(value):
    date = datetime.fromisoformat(str(value).replace('Z', '+00:00'))
    return date.replace(tzinfo=timezone.utc) if date.tzinfo is None else date.astimezone(timezone.utc)


def _slots(metadata):
    groups = defaultdict(dict)
    for row in metadata:
        groups[(row['pair_id'], row['input_digest'])][row.get('model_key')] = row
    output = set()
    for models in groups.values():
        if not all(key in models for key in ('v6', 'v7', 'league_poisson', 'always_1_1')):
            continue
        old, new = models['v6'], models['v7']
        if not all(record.get('eligible_for_comparison', False) for record in models.values()):
            continue
        if old['model_version'] != engine.LIVE_MODEL_VERSION:
            continue
        if new['model_version'] not in (forecast_model.LIVE_MODEL_VERSION, forecast_model.LIVE_MODEL_VERSION+'-fallback-v6'):
            continue
        if old.get('slot_key') and all(record.get('slot_key') == old['slot_key'] for record in models.values()):
            output.add(old['slot_key'])
    return output


def _verified_results(conn):
    for row in load(conn):
        if row.get('verification_status') != 'verified' or not row.get('actual_score'):
            continue
        try:
            home, away = (int(value) for value in row['actual_score'].split('-'))
            yield {'provider': row['provider'], 'fixture_id': row['fixture_id'],
                   'home_team_id': row['home_id'], 'away_team_id': row['away_id'],
                   'kickoff_utc': row['kickoff'], 'home_goals': home, 'away_goals': away,
                   'verified_at_utc': row.get('verified_at') or datetime.now(timezone.utc).isoformat(),
                   'verification_status': 'verified', 'status': 'FINISHED', 'score_basis': 'regulation'}
        except (ValueError, TypeError):
            continue


def maintain_archive(conn, rows, histories, database_url='', ledger_path='forecast_ledger.sqlite3',
                     file_root='data/prospective_ledger'):
    """Keep legacy and prospective ledgers separate; never overwrite a forecast.

    A failed central connection leaves an explicit local outbox. Connection
    strings and provider exceptions are never returned to the UI or written.
    """
    status = {'state': 'complete', 'central_configured': bool(database_url), 'errors': [],
              'captured_slots': 0, 'skipped_slots': 0, 'forecast_records_written': 0}
    try:
        store = open_ledger(database_url=database_url or '', path=ledger_path)
    except Exception as exc:
        status['errors'].append({'operation': 'central_connection', 'error_type': type(exc).__name__})
        store = open_ledger(database_url='', path=ledger_path)
    with store:
        status.update(store.status())
        try:
            status['imported'] = sync_file_ledger(store, root=file_root)
        except Exception as exc:
            status['errors'].append({'operation': 'file_sync', 'error_type': type(exc).__name__})
        # Remote originals must arrive BEFORE INSERT OR IGNORE saves today's
        # forecast, otherwise an app restart could supersede an earlier pick.
        status['restored'] = sync_archive(conn, store, restore=True, upload=False)
        restore_ok = (not database_url or
                      (status.get('durable') and status['restored'].get('state') == 'complete'))
        status['original_saves_allowed'] = bool(restore_ok)
        if restore_ok:
            save_predictions(conn, [row for row in rows if row.get('result')])
        else:
            status['deferred_original_saves'] = sum(bool(row.get('result')) for row in rows)
            status['errors'].append({'operation': 'original_save_deferred', 'error_type': 'CentralRestoreUnavailable'})
        for history in histories.values():
            reconcile(conn, history)
        status['archive'] = sync_archive(conn, store, restore=False, upload=True)
        known = _slots(store.iter_forecast_metadata())
        new_records = []
        for row in rows:
            evidence, fixture = row.get('evidence'), row.get('fixture') or {}
            if not row.get('result') or not evidence:
                continue
            try:
                observed = datetime.now(timezone.utc)
                horizon = horizon_for(fixture['utcDate'], observed)
                slot = pair_slot_key(fixture, horizon, evidence['policy_version'])
                if slot in known:
                    continue
                records = capture_pair(fixture, evidence, observed_at=observed)
                for record in records:
                    append_file_record(record, root=file_root)
                new_records.extend(records)
                known.add(slot)
                status['captured_slots'] += 1
            except Exception as exc:
                status['skipped_slots'] += 1
                status['errors'].append({'operation': 'prospective_capture', 'error_type': type(exc).__name__})
        try:
            status['forecast_records_written'] = store.put_forecasts(new_records)
        except Exception as exc:
            status['errors'].append({'operation': 'forecast_write', 'error_type': type(exc).__name__})
        forecasts = store.iter_forecasts()
        identities = {(r['provider'], str(r['fixture_id']), _date(r['kickoff_utc'])) for r in forecasts}
        results = [row for row in _verified_results(conn)
                   if (row['provider'], str(row['fixture_id']), _date(row['kickoff_utc'])) in identities]
        for result in results:
            try:
                store.put_settlement(result)
                # A private database may contain forecasts from another app
                # instance; only mirror a settlement when its file is present.
                try:
                    append_file_record(result, root=file_root, kind='settlements')
                except ValueError:
                    pass
            except Exception as exc:
                status['errors'].append({'operation': 'settlement_write', 'error_type': type(exc).__name__})
        report = evaluate(forecasts, store.get_settlements())
        status['stats'] = store.stats()
    if status['errors'] or any(status.get(key, {}).get('errors') for key in ('restored', 'archive')):
        status['state'] = 'partial'
    return status, report


def backup_after_changes(conn, database_url='', ledger_path='forecast_ledger.sqlite3'):
    """Immediately back up a newly frozen double, without restoring or replacing it."""
    try:
        with open_ledger(database_url=database_url or '', path=ledger_path) as store:
            return sync_archive(conn, store, restore=False, upload=True)
    except Exception as exc:
        return {'state': 'partial', 'durable': False,
                'errors': [{'operation': 'archive_backup', 'error_type': type(exc).__name__}]}
