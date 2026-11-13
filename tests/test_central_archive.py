"""Restore original private archives without backfilling prospective evidence."""
import json
from copy import deepcopy

import pytest

from central_archive import sync_archive, SINGLE_COLUMNS, DOUBLE_COLUMNS
from double_tracker import setup as setup_doubles
from forecast_ledger import open_ledger
from history_store import connect, load


def single(fixture_id=123, provider='API-Football', **changes):
    row = {'provider': provider, 'fixture_id': fixture_id,
           'saved_at': '2026-10-01T10:00:00+00:00',
           'kickoff': '2026-10-02T12:00:00Z', 'league': 'PL',
           'home': 'Home Club', 'away': 'Away Club', 'home_id': '10', 'away_id': '20',
           'predicted_score': '2-1', 'home_prob': .5, 'draw_prob': .3, 'away_prob': .2,
           'scores_json': '[[2,1,0.13],[1,1,0.12]]', 'actual_score': '2-1',
           'verification_status': 'verified', 'verified_at': '2026-10-02T15:00:00Z',
           'legacy_actual_score': None, 'model_version': 'preserved-original-model'}
    row.update(changes)
    return row


def double(**changes):
    row = {'snapshot_key': '2026-10-01:None:123:None:456',
           'saved_at': '2026-10-01T10:00:00+00:00', 'first_id': '123', 'second_id': '456',
           'first_name': 'Home Club vs Away Club', 'second_name': 'Other Home vs Other Away',
           'predictions_json': '[{"score1":"2-1","score2":"1-0","probability":0.01}]',
           'first_provider': None, 'second_provider': None,
           'first_kickoff': None, 'second_kickoff': None,
           'first_model': None, 'second_model': None}
    row.update(changes)
    return row


def insert(conn, table, row):
    keys = list(row)
    conn.execute(f"INSERT INTO {table} ({','.join(keys)}) VALUES ({','.join('?' for _ in keys)})",
                 tuple(row[key] for key in keys))
    conn.commit()


def read_doubles(conn):
    cursor = conn.execute('SELECT * FROM exact_score_doubles ORDER BY snapshot_key')
    return [dict(zip([entry[0] for entry in cursor.description], row)) for row in cursor.fetchall()]


@pytest.fixture
def ledger(tmp_path):
    with open_ledger(database_url='', path=tmp_path/'central.sqlite3') as store:
        yield store


@pytest.fixture
def local():
    conn = connect(':memory:')
    yield conn
    conn.close()


def test_empty_instance_recovers_exact_originals_and_legacy_double_metadata(local, ledger):
    original, pair = single(), double()
    ledger.put_archive_record(original)
    ledger.put_archive_record(pair, kind='double')
    status = sync_archive(local, ledger)
    assert status['state'] == 'complete'
    assert status['restored_singles'] == status['restored_doubles'] == 1
    assert load(local) == [original]
    assert read_doubles(local) == [pair]
    assert ledger.stats()['forecasts'] == 0


def test_all_236_originals_survive_a_local_database_restart(tmp_path, ledger):
    first = connect(':memory:')
    for fixture_id in range(1, 237):
        insert(first, 'predictions', single(fixture_id))
    status = sync_archive(first, ledger, restore=False)
    assert status['uploaded_singles'] == 236
    first.close()
    restarted = connect(':memory:')
    try:
        restored = sync_archive(restarted, ledger, upload=False)
        assert restored['restored_singles'] == 236
        assert len(load(restarted)) == 236
        assert all(row['saved_at'] == '2026-10-01T10:00:00+00:00' for row in load(restarted))
        assert ledger.stats()['forecasts'] == 0
    finally:
        restarted.close()


def test_existing_local_prediction_and_verified_result_are_never_overwritten(local, ledger):
    current = single(predicted_score='3-0', actual_score='3-0')
    insert(local, 'predictions', current)
    ledger.put_archive_record(single(predicted_score='1-1', actual_score='0-1'))
    status = sync_archive(local, ledger)
    assert status['restored_singles'] == 0
    assert status['skipped_existing'] == 1
    assert load(local) == [current]
    assert len(ledger.archive_versions('single')) == 2


def test_existing_original_double_choices_are_never_overwritten(local, ledger):
    setup_doubles(local)
    current = double(predictions_json='[{"score1":"3-0","score2":"2-0","probability":0.02}]')
    insert(local, 'exact_score_doubles', current)
    ledger.put_archive_record(double(), kind='double')
    status = sync_archive(local, ledger)
    assert status['restored_doubles'] == 0
    assert read_doubles(local) == [current]


def test_known_provider_id_clashes_remain_two_original_forecasts(local, ledger):
    a = single(provider='API-Football')
    b = single(provider='football-data.org', predicted_score='1-0')
    ledger.put_archive_records([a, b])
    status = sync_archive(local, ledger)
    assert status['restored_singles'] == 2
    assert {row['provider'] for row in load(local)} == {'API-Football', 'football-data.org'}


def test_unique_recovered_provider_does_not_restore_an_old_unknown_duplicate(local, ledger):
    unknown = single(provider='legacy-unknown', home_id=None, away_id=None,
                     actual_score=None, verification_status='legacy_unverified')
    ledger.put_archive_record(unknown)
    ledger.put_archive_record(single())
    status = sync_archive(local, ledger)
    assert status['restored_singles'] == 1
    assert load(local)[0]['provider'] == 'API-Football'
    assert len(ledger.archive_versions('single')) == 2


def test_multiple_known_sources_cannot_claim_an_ambiguous_legacy_original(local, ledger):
    rows = [single(provider='legacy-unknown', home_id=None, away_id=None,
                   actual_score=None, verification_status='legacy_unverified'),
            single(provider='API-Football'), single(provider='football-data.org')]
    ledger.put_archive_records(rows)
    sync_archive(local, ledger)
    assert len(load(local)) == 3


def test_local_legacy_original_is_not_superseded_by_a_remote_qualified_duplicate(local, ledger):
    unknown = single(provider='legacy-unknown', home_id=None, away_id=None,
                     actual_score=None, verification_status='legacy_unverified')
    insert(local, 'predictions', unknown)
    ledger.put_archive_record(single())
    status = sync_archive(local, ledger, upload=False)
    assert status['restored_singles'] == 0
    assert load(local) == [unknown]


def test_same_id_with_other_teams_does_not_suppress_a_separate_unknown_original(local, ledger):
    unknown = single(provider='legacy-unknown', home='Different Home', away='Different Away',
                     home_id=None, away_id=None, actual_score=None, verification_status='legacy_unverified')
    insert(local, 'predictions', unknown)
    ledger.put_archive_record(single())
    sync_archive(local, ledger)
    assert len(load(local)) == 2


def test_uploads_allowlisted_columns_only_and_remain_idempotent(local, ledger):
    local.execute('ALTER TABLE predictions ADD COLUMN database_password TEXT')
    original = dict(single(), database_password='never-archive-this-value')
    insert(local, 'predictions', original)
    first = sync_archive(local, ledger, restore=False)
    second = sync_archive(local, ledger, restore=False)
    assert first['uploaded_singles'] == 1
    assert second['uploaded_singles'] == 0
    assert 'never-archive-this-value' not in json.dumps(ledger.archive_versions())
    assert set(ledger.iter_archive_records()[0]) == set(SINGLE_COLUMNS)


def test_database_error_status_never_exposes_credentials(local):
    class Unavailable:
        def status(self):
            raise RuntimeError('postgresql://private-password@private-host/database')
        def iter_archive_records(self, **kwargs):
            raise RuntimeError('private-password')
        def put_archive_records(self, *args, **kwargs):
            raise RuntimeError('private-password')
    insert(local, 'predictions', single())
    result = sync_archive(local, Unavailable())
    assert result['state'] == 'partial'
    assert result['errors']
    assert 'private-password' not in json.dumps(result)
    assert len(load(local)) == 1


def test_restore_ignores_unexpected_payload_fields_without_interpolating_sql(local, ledger):
    # The central storage accepts opaque archives; restoration has a separate
    # SQL-column allowlist, so future/unknown keys cannot become SQL syntax.
    payload = dict(single(), **{'made_up_column); DROP TABLE predictions; --': 'ignored'})
    ledger.put_archive_record(payload)
    result = sync_archive(local, ledger, upload=False)
    assert result['state'] == 'complete'
    assert load(local) == [single()]


def test_restore_and_upload_can_be_split_around_normal_prediction_saves(local, ledger):
    ledger.put_archive_record(single())
    restored = sync_archive(local, ledger, upload=False)
    assert restored['restored_singles'] == 1
    assert restored['uploaded_singles'] == 0
    insert(local, 'predictions', single(456, provider='football-data.org'))
    uploaded = sync_archive(local, ledger, restore=False)
    assert uploaded['restored_singles'] == 0
    assert uploaded['uploaded_singles'] == 1


def test_earliest_single_wins_even_when_later_forecast_was_uploaded_first(local, ledger):
    later = single(saved_at='2026-10-01T12:00:00Z', predicted_score='2-0')
    earlier = single(saved_at='2026-10-01T11:00:00Z', predicted_score='1-1')
    ledger.put_archive_record(later)
    ledger.put_archive_record(earlier)
    result = sync_archive(local, ledger, upload=False)
    assert result['restored_singles'] == 1
    assert load(local) == [earlier]


def test_single_capture_order_uses_utc_instants_not_timestamp_strings(local, ledger):
    later = single(saved_at='2026-10-01T12:00:00Z', predicted_score='2-0')
    earlier = single(saved_at='2026-10-01T22:00:00+11:00', predicted_score='1-1')
    ledger.put_archive_records([later, earlier])
    sync_archive(local, ledger, upload=False)
    assert load(local) == [earlier]


def test_earlier_remote_original_never_replaces_an_existing_local_forecast(local, ledger):
    current = single(saved_at='2026-10-01T12:00:00Z', predicted_score='2-0')
    earlier = single(saved_at='2026-10-01T11:00:00Z', predicted_score='1-1')
    insert(local, 'predictions', current)
    ledger.put_archive_record(earlier)
    sync_archive(local, ledger, upload=False)
    assert load(local) == [current]


def test_double_restores_earliest_core_with_only_its_matching_metadata_updates(local, ledger):
    original = double(saved_at='2026-10-01T11:00:00Z')
    recovered = dict(original, first_provider='API-Football', second_provider='football-data.org')
    later = double(saved_at='2026-10-01T12:00:00Z',
                   predictions_json='[{"score1":"3-0","score2":"3-0","probability":0.02}]')
    ledger.put_archive_record(original, kind='double')
    ledger.put_archive_record(recovered, kind='double')
    ledger.put_archive_record(later, kind='double')
    assert ledger.iter_archive_records(kind='double') == [recovered]
    result = sync_archive(local, ledger, upload=False)
    assert result['restored_doubles'] == 1
    assert read_doubles(local) == [recovered]
