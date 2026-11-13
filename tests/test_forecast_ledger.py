"""Backend parity and immutable-evidence storage invariants.

Set TEST_DATABASE_URL to an isolated PostgreSQL database for the same tests to
run against PostgreSQL; CI supplies a temporary PostgreSQL service.
"""
from copy import deepcopy
import json
import os
import sqlite3
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
import uuid

import pytest

from forecast_ledger import (ImmutableConflict, LedgerUnavailableError,
    LedgerValidationError, append_file_record, forecast_id, open_ledger,
    read_file_records, sync_file_ledger)


def forecast(**changes):
    identifier = int(uuid.uuid4().hex[:12], 16)
    record = {'provider': 'API-Football', 'fixture_id': identifier,
        'home_team_id': 1, 'away_team_id': 2,
        'home_team': 'Home', 'away_team': 'Away',
        'model_key': 'v6', 'model_version': 'test-v6',
        'numerical_model_version': 'venue-form-v6',
        'evidence_as_of_utc': '2026-11-12T12:00:00Z',
        'captured_at_utc': '2026-11-12T12:00:01Z',
        'kickoff_utc': '2026-11-13T12:00:00Z',
        'horizon': 'T24h', 'policy_version': 'two-season-v1',
        'input_digest': 'f'*64, 'pair_id': uuid.uuid4().hex,
        'eligible_for_comparison': True, 'mode': 'prospective',
        'score_basis': 'regulation',
        'score_matrix': [[.2, .1], [.3, .4]], 'matrix_shape': [2, 2], 'matrix_mass': 1.,
        'top10_scores': [[1, 1, .4], [1, 0, .3], [0, 0, .2], [0, 1, .1]],
        'outcome_probabilities': {'home': .3, 'draw': .6, 'away': .1}}
    record.update(changes)
    return record


def settlement(record, **changes):
    result = {key: record[key] for key in
              ('provider', 'fixture_id', 'kickoff_utc', 'home_team_id', 'away_team_id')}
    result.update(verified=True, status='FINISHED', score_basis='regulation',
                  home_goals=2, away_goals=1, verified_at_utc='2026-11-13T14:30:00Z')
    result.update(changes)
    return result


@pytest.fixture(params=['sqlite', 'postgresql'])
def ledger(request):
    if request.param == 'postgresql':
        url = os.environ.get('TEST_DATABASE_URL')
        if not url:
            pytest.skip('An isolated TEST_DATABASE_URL is required for PostgreSQL integration')
        store = open_ledger(database_url=url)
    else:
        store = open_ledger(database_url='', path=':memory:')
    try:
        yield store
    finally:
        store.close()


def test_complete_grid_round_trip_and_idempotency(ledger):
    original = forecast()
    assert ledger.put_forecast(original)
    assert not ledger.put_forecast(original)
    stored = next(row for row in ledger.iter_forecasts() if row['forecast_id'] == forecast_id(original))
    assert stored['score_matrix'] == original['score_matrix']
    assert stored['input_digest'] == original['input_digest']
    assert original['fixture_id'] == int(stored['fixture_id'])
    metadata = next(row for row in ledger.iter_forecast_metadata() if row['fixture_id'] == str(original['fixture_id']))
    assert 'score_matrix' not in metadata
    assert metadata['model_version'] == original['model_version']
    assert metadata['eligible_for_comparison'] is True
    assert ledger.status()['durable'] == (ledger.backend == 'postgresql')


def test_forecast_collision_rejects_changes_and_preserves_original(ledger):
    original = forecast()
    ledger.put_forecast(original)
    changed = deepcopy(original)
    changed['input_digest'] = 'a'*64
    with pytest.raises(ImmutableConflict):
        ledger.put_forecast(changed)
    stored = next(row for row in ledger.iter_forecasts() if row['forecast_id'] == forecast_id(original))
    assert stored['input_digest'] == 'f'*64


def test_changed_evidence_horizon_and_model_get_separate_identities(ledger):
    original = forecast()
    records = [original, dict(original, model_version='test-v7', model_key='v7'),
               dict(original, horizon='research'),
               dict(original, evidence_as_of_utc='2026-11-12T11:00:00Z')]
    assert ledger.put_forecasts(records) == 4
    assert ledger.put_forecasts(records) == 0


def test_equivalent_offsets_and_numeric_ids_share_identity():
    original = forecast()
    offset = dict(original, fixture_id=str(original['fixture_id']),
                  evidence_as_of_utc='2026-11-12T23:00:00+11:00')
    assert forecast_id(original) == forecast_id(offset)


def test_batch_conflict_rolls_back_all_new_rows(ledger):
    original, fresh = forecast(), forecast()
    ledger.put_forecast(original)
    changed = dict(original, input_digest='changed')
    with pytest.raises(ImmutableConflict):
        ledger.put_forecasts([fresh, changed])
    assert forecast_id(fresh) not in {row['forecast_id'] for row in ledger.iter_forecasts()}


def test_concurrent_idempotent_forecasts(ledger):
    original = forecast()
    with ThreadPoolExecutor(max_workers=4) as workers:
        inserted = list(workers.map(lambda _: ledger.put_forecast(original), range(8)))
    assert sum(inserted) == 1


def test_settlement_requires_exact_provider_teams_kickoff_and_regulation(ledger):
    original = forecast()
    ledger.put_forecast(original)
    for changes in ({'provider': 'football-data.org'}, {'home_team_id': 99},
                    {'kickoff_utc': '2026-11-13T13:00:00Z'},
                    {'score_basis': 'extra_time'}, {'status': 'LIVE'}, {'verified': False},
                    {'home_goals': True}, {'verified_at_utc': '2026-11-13T11:00:00Z'}):
        with pytest.raises(LedgerValidationError):
            ledger.put_settlement(settlement(original, **changes))
    assert ledger.put_settlement(settlement(original))
    assert not ledger.put_settlement(settlement(original, verified_at_utc='2026-11-13T15:00:00Z'))
    with pytest.raises(ImmutableConflict):
        ledger.put_settlement(settlement(original, home_goals=3))


def test_orphan_settlement_cannot_enter_verified_ledger(ledger):
    with pytest.raises(LedgerValidationError):
        ledger.put_settlement(settlement(forecast()))


@pytest.mark.parametrize('changes', [
    {'mode': 'retrospective'}, {'actual_score': [2, 1]},
    {'captured_at_utc': '2026-11-13T12:00:00Z'},
    {'evidence_as_of_utc': '2026-11-12T12:00:02Z'},
    {'captured_at_utc': '2026-11-12T12:00:01'},
    {'score_matrix': [[.5, .5], [.1]]},
    {'score_matrix': [[.2, .1], [.3, float('nan')]]},
    {'score_matrix': [[.2, .1], [.3, .2]]},
    {'matrix_shape': [3, 3]}, {'matrix_mass': .5},
    {'top10_scores': [[1, 1, .9]]},
    {'outcome_probabilities': {'home': .9, 'draw': .05, 'away': .05}},
    {'home_team_id': 2}, {'forecast_id': 'invented'},
    {'provenance': {'api_key': 'must-never-write'}}])
def test_invalid_or_backfilled_forecasts_are_rejected(changes):
    with open_ledger(database_url='', path=':memory:') as store:
        with pytest.raises(LedgerValidationError):
            store.put_forecast(forecast(**changes))
        assert store.stats()['forecasts'] == 0


def old_single(provider='legacy-unknown', **changes):
    row = {'fixture_id': int(uuid.uuid4().hex[:12], 16), 'provider': provider,
           'saved_at': '2026-11-01T10:00:00Z', 'kickoff': '2026-11-02T12:00:00Z',
           'home': 'Home', 'away': 'Away', 'predicted_score': '1-1',
           'actual_score': None, 'home_id': None, 'away_id': None}
    row.update(changes)
    return row


def test_legacy_archive_is_separate_versioned_and_provider_recovery_safe(ledger):
    original = old_single()
    assert ledger.put_archive_record(original)
    assert not ledger.put_archive_record(original)
    recovered = dict(original, provider='API-Football', home_id='1', away_id='2', actual_score='2-1')
    assert ledger.put_archive_record(recovered)
    versions = [v for v in ledger.archive_versions('single') if v['record']['fixture_id'] == original['fixture_id']]
    assert len(versions) == 2
    latest = [r for r in ledger.iter_archive_records() if r['fixture_id'] == original['fixture_id']]
    assert latest == [recovered]
    assert not any(str(r['fixture_id']) == str(original['fixture_id']) for r in ledger.iter_forecasts())


def test_ambiguous_known_provider_collision_preserves_both_and_unknown(ledger):
    original = old_single()
    records = [original, dict(original, provider='API-Football'),
               dict(original, provider='football-data.org')]
    assert ledger.put_archive_records(records) == 3
    latest = [r for r in ledger.iter_archive_records() if r['fixture_id'] == original['fixture_id']]
    assert len(latest) == 3


def test_236_legacy_records_survive_restart_and_batch_reupload(tmp_path):
    path = tmp_path/'central.sqlite'
    records = [old_single() for _ in range(236)]
    with open_ledger(database_url='', path=path) as store:
        assert store.put_archive_records(records) == 236
        assert store.put_archive_records(records) == 0
    with open_ledger(database_url='', path=path) as store:
        assert len(store.iter_archive_records()) == 236
        assert store.status()['forecasts'] == 0


def test_legacy_doubles_preserve_null_metadata_and_versions(ledger):
    original = {'snapshot_key': uuid.uuid4().hex, 'saved_at': '2026-11-01T10:00:00Z',
                'first_id': '12', 'second_id': '13', 'first_provider': None,
                'second_provider': None, 'predictions_json': '[]'}
    assert ledger.put_archive_record(original, kind='double')
    recovered = dict(original, first_provider='API-Football')
    assert ledger.put_archive_record(recovered, kind='double')
    assert recovered in ledger.iter_archive_records(kind='double')
    assert original in [v['record'] for v in ledger.archive_versions('double')]


def test_double_original_is_earliest_capture_even_when_uploaded_out_of_order(ledger):
    key = uuid.uuid4().hex
    later = {'snapshot_key': key, 'saved_at': '2026-11-01T12:00:00Z',
             'first_id': '12', 'second_id': '13', 'first_name': 'A vs B',
             'second_name': 'C vs D', 'predictions_json': '[{"score1":"2-0"}]'}
    earlier = dict(later, saved_at='2026-11-01T11:00:00Z',
                   predictions_json='[{"score1":"1-1"}]')
    ledger.put_archive_record(later, kind='double')
    ledger.put_archive_record(earlier, kind='double')
    ledger.put_archive_record(dict(later, first_provider='API-Football'), kind='double')
    restored = next(row for row in ledger.iter_archive_records('double') if row['snapshot_key'] == key)
    assert restored == earlier
    assert len([v for v in ledger.archive_versions('double') if v['record']['snapshot_key'] == key]) == 3


def test_double_metadata_enrichment_cannot_rewrite_earliest_choices_or_legs(ledger):
    original = {'snapshot_key': uuid.uuid4().hex, 'saved_at': '2026-11-01T11:00:00Z',
                'first_id': '12', 'second_id': '13', 'first_name': 'A vs B',
                'second_name': 'C vs D', 'first_provider': None,
                'predictions_json': '[{"score1":"1-1"}]'}
    ledger.put_archive_record(original, kind='double')
    ledger.put_archive_record(dict(original, predictions_json='[{"score1":"4-0"}]'), kind='double')
    ledger.put_archive_record(dict(original, first_id='99'), kind='double')
    enriched = dict(original, first_provider='API-Football', first_model='known-original-model')
    ledger.put_archive_record(enriched, kind='double')
    restored = next(row for row in ledger.iter_archive_records('double')
                    if row['snapshot_key'] == original['snapshot_key'])
    assert restored == enriched
    assert restored['predictions_json'] == original['predictions_json']
    assert restored['first_id'] == original['first_id']


def test_double_capture_order_uses_parsed_time_not_lexical_offsets(ledger):
    original = {'snapshot_key': uuid.uuid4().hex, 'saved_at': '2026-11-01T12:00:00Z',
                'first_id': '12', 'second_id': '13', 'predictions_json': '[]'}
    earlier = dict(original, saved_at='2026-11-01T22:00:00+11:00', predictions_json='[1]')
    ledger.put_archive_records([original, earlier], kind='double')
    restored = next(row for row in ledger.iter_archive_records('double')
                    if row['snapshot_key'] == original['snapshot_key'])
    assert restored == earlier


def test_ambiguous_double_provider_recovery_does_not_guess_by_upload_order(ledger):
    original = {'snapshot_key': uuid.uuid4().hex, 'saved_at': '2026-11-01T11:00:00Z',
                'first_id': '12', 'second_id': '13', 'first_provider': None,
                'predictions_json': '[]'}
    ledger.put_archive_records([original, dict(original, first_provider='API-Football'),
                               dict(original, first_provider='football-data.org')], kind='double')
    restored = next(row for row in ledger.iter_archive_records('double')
                    if row['snapshot_key'] == original['snapshot_key'])
    assert restored['first_provider'] is None


def test_opaque_archive_refuses_credentials(ledger):
    with pytest.raises(LedgerValidationError):
        ledger.put_archive_record(old_single(password='private'))


def test_file_journal_is_immutable_and_sync_is_batched_idempotent(tmp_path):
    original = forecast()
    assert append_file_record(original, tmp_path)
    assert not append_file_record(original, tmp_path)
    assert append_file_record(settlement(original), tmp_path, 'settlements')
    assert not append_file_record(settlement(original, verified_at_utc='2026-11-13T15:00:00Z'), tmp_path, 'settlements')
    with pytest.raises(ImmutableConflict):
        append_file_record(dict(original, input_digest='changed'), tmp_path)
    with open_ledger(database_url='', path=':memory:') as store:
        assert sync_file_ledger(store, tmp_path) == {'forecasts': 1, 'settlements': 1}
        assert sync_file_ledger(store, tmp_path) == {'forecasts': 0, 'settlements': 0}


def test_file_settlement_needs_matching_forecast_and_bad_files_are_not_silently_skipped(tmp_path):
    with pytest.raises(LedgerValidationError):
        append_file_record(settlement(forecast()), tmp_path, 'settlements')
    bad = tmp_path/'forecasts'/'bad.json'
    bad.parent.mkdir()
    bad.write_text('{broken')
    with pytest.raises(LedgerValidationError):
        read_file_records(tmp_path)


def test_file_root_can_be_the_forecasts_subdirectory(tmp_path):
    assert append_file_record(forecast(), tmp_path/'forecasts')
    assert len(read_file_records(tmp_path/'forecasts')) == 1


def test_existing_application_tables_are_untouched(tmp_path):
    path = tmp_path/'app.sqlite'
    connection = sqlite3.connect(path)
    connection.execute('CREATE TABLE predictions (original TEXT)')
    connection.execute("INSERT INTO predictions VALUES ('unchanged')")
    connection.commit()
    connection.close()
    with open_ledger(database_url='', path=path) as store:
        store.put_forecast(forecast())
    connection = sqlite3.connect(path)
    assert connection.execute('SELECT * FROM predictions').fetchall() == [('unchanged',)]
    connection.close()


def test_database_failures_do_not_expose_connection_credentials(monkeypatch):
    def connect(*_, **__):
        raise RuntimeError('postgresql://user:must-not-print@private-host/db')
    monkeypatch.setitem(sys.modules, 'psycopg', SimpleNamespace(connect=connect))
    with pytest.raises(LedgerUnavailableError) as error:
        open_ledger(database_url='postgresql://user:must-not-print@private-host/db')
    assert 'must-not-print' not in str(error.value)
    assert 'private-host' not in str(error.value)


def test_closed_connection_is_explicit():
    store = open_ledger(database_url='', path=':memory:')
    store.close()
    with pytest.raises(LedgerUnavailableError):
        store.iter_forecasts()
