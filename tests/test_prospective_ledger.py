from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json

import numpy as np
import pytest

import prospective_ledger as ledger
from history_pipeline import build_evidence

NOW = datetime(2026, 11, 13, 12, tzinfo=timezone.utc)


def fixture():
    return {'id': 1234, 'provider': 'API-Football', 'utcDate': (NOW+timedelta(hours=1)).isoformat(),
            'competition': {'id': 39, 'code': 'PL', 'season': 2026}, 'status': 'SCHEDULED',
            'provider_status': 'NS', 'homeTeam': {'id': 1, 'name': 'A'},
            'awayTeam': {'id': 2, 'name': 'B'}, 'score': {'fullTime': {'home': None, 'away': None}}}


def history():
    output = []
    for cycle in range(2):
        for home in range(1, 7):
            for away in range(1, 7):
                if home == away:
                    continue
                index = len(output)
                output.append({'id': index+1, 'provider': 'API-Football',
                    'utcDate': (NOW-timedelta(days=100-index)).isoformat(),
                    'competition': {'id': 39, 'code': 'PL', 'season': 2026}, 'status': 'FINISHED',
                    'homeTeam': {'id': home, 'name': str(home)}, 'awayTeam': {'id': away, 'name': str(away)},
                    'score': {'duration': 'REGULAR', 'fullTime': {'home': 3 if home == 1 else 1, 'away': 1}}})
    return output


def records(monkeypatch):
    monkeypatch.setattr(ledger, '_utcnow', lambda: NOW)
    evidence = build_evidence(fixture(), history(), NOW,
        source_metadata={'provider': 'API-Football', 'fetched_at_utc': NOW.isoformat(), 'stale': False})
    return ledger.capture_pair(fixture(), evidence, observed_at=NOW)


def result(score=(1, 1)):
    return {'fixture_id': 1234, 'provider': 'API-Football', 'home_team_id': 1, 'away_team_id': 2,
            'kickoff_utc': fixture()['utcDate'], 'verified_at_utc': (NOW+timedelta(hours=4)).isoformat(),
            'home_goals': score[0], 'away_goals': score[1], 'score_basis': 'regulation',
            'status': 'FT', 'evidence_version': 2}


def test_capture_stores_actual_v6_v7_and_simple_baselines_with_complete_grids(monkeypatch):
    values = records(monkeypatch)
    assert {row['model_key'] for row in values} == set(ledger.MODELS)
    assert len({row['input_digest'] for row in values}) == 1
    assert len({row['pair_id'] for row in values}) == 1
    assert all(row['horizon'] == 'T60m' for row in values)
    assert all(np.isclose(np.asarray(row['score_matrix']).sum(), 1) for row in values)
    assert values[1]['numerical_model_version'] == 'reference-informed-v7'
    assert values[1]['eligible_for_comparison']
    json.dumps(values, allow_nan=False)


def test_horizon_labels_follow_real_capture_time_not_user_claim():
    assert ledger.horizon_for(NOW+timedelta(hours=24), NOW) == 'T24h'
    assert ledger.horizon_for(NOW+timedelta(minutes=30), NOW) == 'T30m'
    assert ledger.horizon_for(NOW+timedelta(minutes=45), NOW) == 'T60m'
    assert ledger.horizon_for(NOW+timedelta(days=3), NOW) == 'research'
    with pytest.raises(ValueError):
        ledger.horizon_for(NOW, NOW)


def test_slot_key_is_stable_for_refreshes_but_scoped_to_policy_model_and_horizon():
    same = ledger.pair_slot_key(fixture(), 'T60m', 'policy-v1')
    assert same == ledger.pair_slot_key(fixture(), 'T60m', 'policy-v1')
    assert same != ledger.pair_slot_key(fixture(), 'T30m', 'policy-v1')
    assert same != ledger.pair_slot_key(fixture(), 'T60m', 'policy-v2')
    assert same != ledger.pair_slot_key(fixture(), 'T60m', 'policy-v1', release_version='next-release')


def test_context_fetch_provenance_is_preserved_without_secrets_or_extra_features(monkeypatch):
    monkeypatch.setattr(ledger, '_utcnow', lambda: NOW)
    evidence = build_evidence(fixture(), history(), NOW)
    evidence['provenance']['fixture_source'] = {'endpoint': '/fixtures', 'fetched_at_utc': NOW.isoformat(), 'api_key': 'secret'}
    evidence['provenance']['news_sources'] = [{'endpoint': '/fixtures/lineups', 'stale': False, 'api_key': 'secret'}]
    values = ledger.capture_pair(fixture(), evidence)
    assert values[0]['provenance']['fixture_source']['endpoint'] == '/fixtures'
    assert values[0]['provenance']['news_sources'][0]['stale'] is False
    assert 'secret' not in json.dumps(values)
    assert values[0]['input_digest'] == evidence['input_digest']


def test_cannot_backfill_a_finished_match_or_fake_old_observation(monkeypatch):
    monkeypatch.setattr(ledger, '_utcnow', lambda: NOW)
    evidence = build_evidence(fixture(), history(), NOW)
    with pytest.raises(ValueError, match='Historical'):
        ledger.capture_pair(fixture(), evidence, observed_at=NOW-timedelta(days=1))
    target = fixture()
    target['status'] = 'FINISHED'
    with pytest.raises(ValueError, match='upcoming'):
        ledger.capture_pair(target, evidence)


def test_actual_target_score_is_rejected_before_models_run(monkeypatch):
    monkeypatch.setattr(ledger, '_utcnow', lambda: NOW)
    target = fixture()
    evidence = build_evidence(target, history(), NOW)
    target['score']['fullTime'] = {'home': 5, 'away': 0}
    with pytest.raises(ValueError, match='eventual score'):
        ledger.capture_pair(target, evidence)


def test_changed_history_or_digest_cannot_masquerade_as_same_evidence(monkeypatch):
    monkeypatch.setattr(ledger, '_utcnow', lambda: NOW)
    evidence = build_evidence(fixture(), history(), NOW)
    evidence['league_history'] = deepcopy(evidence['league_history'])
    evidence['league_history'][0]['score']['fullTime']['home'] += 1
    with pytest.raises(ValueError, match='digest'):
        ledger.capture_pair(fixture(), evidence)


def test_computation_crossing_kickoff_cannot_create_snapshot(monkeypatch):
    clock = iter([NOW, NOW+timedelta(hours=2)])
    monkeypatch.setattr(ledger, '_utcnow', lambda: next(clock))
    evidence = build_evidence(fixture(), history(), NOW)
    with pytest.raises(ValueError, match='finished after kickoff'):
        ledger.capture_pair(fixture(), evidence)


def test_full_grid_scores_actual_outside_top_ten_and_reports_tail_policy(monkeypatch):
    values = records(monkeypatch)
    report = ledger.evaluate(values, [result((8, 7))])
    assert report['paired_verified_matches'] == 1
    assert np.isfinite(report['models']['v7']['score_log_loss'])
    assert report['models']['always_1_1']['score_log_loss'] is None
    outside = ledger.evaluate(values, [result((100, 99))])
    assert outside['models']['v6']['out_of_grid_actuals'] == 1
    assert np.isclose(outside['models']['v6']['score_log_loss'], -np.log(ledger.EPS))


@pytest.mark.parametrize('change', [
    {'provider': 'football-data.org'}, {'home_team_id': 9}, {'away_team_id': 9},
    {'kickoff_utc': (NOW+timedelta(hours=2)).isoformat()}, {'score_basis': 'extra_time'},
    {'verified_at_utc': NOW.isoformat()}, {'evidence_version': 1}, {'status': 'LIVE'},
])
def test_result_requires_verified_exact_identity_and_post_capture_time(monkeypatch, change):
    values = records(monkeypatch)
    actual = result()
    actual.update(change)
    report = ledger.evaluate(values, [actual])
    assert report['paired_verified_matches'] == 0
    assert report['pending_common_forecasts'] == 1


def test_stored_grid_required_not_legacy_summary_or_reconstruction(monkeypatch):
    values = records(monkeypatch)
    values[1].pop('score_matrix')
    report = ledger.evaluate(values, [result()])
    assert report['paired_verified_matches'] == 0
    assert report['exclusions']['invalid_forecast'] == 1
    assert report['exclusions']['no_common_v6_v7_snapshot'] == 1


def test_earliest_common_valid_snapshot_selected_without_old_original_block(monkeypatch):
    values = records(monkeypatch)
    later = deepcopy(values)
    for row in later:
        row['captured_at_utc'] = (NOW+timedelta(minutes=2)).isoformat()
        row['evidence_as_of_utc'] = (NOW+timedelta(minutes=2)).isoformat()
        row['pair_id'] = 'later-pair'
    # A stale older v6-only original cannot prevent a new common v6/v7 pair.
    older = deepcopy(values[0])
    older['captured_at_utc'] = (NOW-timedelta(minutes=2)).isoformat()
    older['evidence_as_of_utc'] = older['captured_at_utc']
    older['pair_id'] = 'old-v6-only'
    report = ledger.evaluate([older]+later+values, [result()])
    assert report['paired_verified_matches'] == 1
    assert report['rows'][0]['captured_at_utc'] == NOW.isoformat()
    assert report['models']['league_poisson']['evaluated'] == 1


def test_partial_early_v6_v7_write_does_not_block_later_complete_baseline_package(monkeypatch):
    values = records(monkeypatch)
    later = deepcopy(values)
    for row in later:
        row['captured_at_utc'] = (NOW+timedelta(minutes=2)).isoformat()
        row['evidence_as_of_utc'] = row['captured_at_utc']
        row['pair_id'] = 'later-complete-package'
    report = ledger.evaluate(values[:2]+later, [result()])
    assert report['paired_verified_slots'] == 1
    assert report['rows'][0]['captured_at_utc'] == (NOW+timedelta(minutes=2)).isoformat()
    assert {row['evaluated'] for row in report['models'].values()} == {1}


def test_only_v6_v7_without_simple_baselines_remains_an_incomplete_package(monkeypatch):
    report = ledger.evaluate(records(monkeypatch)[:2], [result()])
    assert report['paired_verified_slots'] == 0
    assert report['exclusions']['no_complete_four_model_snapshot'] == 1
    assert all(row['evaluated'] == 0 for row in report['models'].values())


def test_fallback_is_not_counted_as_an_independent_v7_model(monkeypatch):
    monkeypatch.setattr(ledger, '_utcnow', lambda: NOW)
    target = fixture()
    selected = [match for match in history() if 1 in (match['homeTeam']['id'], match['awayTeam']['id']) or 2 in (match['homeTeam']['id'], match['awayTeam']['id'])]
    evidence = build_evidence(target, selected, NOW)
    values = ledger.capture_pair(target, evidence)
    candidate = next(row for row in values if row['model_key'] == 'v7')
    assert not candidate['eligible_for_comparison']
    report = ledger.evaluate(values, [result()])
    assert report['paired_verified_matches'] == 0
    assert report['exclusions']['v7_fallback_or_ineligible'] == 1


def test_mismatched_required_contributor_does_not_make_unpaired_metrics(monkeypatch):
    values = records(monkeypatch)
    values[1]['home_team_id'] = 99
    report = ledger.evaluate(values, [result()])
    assert report['paired_verified_matches'] == 0
    assert report['models']['v6']['evaluated'] == 0


def test_conflicting_verified_results_are_excluded_not_chosen(monkeypatch):
    report = ledger.evaluate(records(monkeypatch), [result((1, 1)), result((2, 0))])
    assert report['paired_verified_matches'] == 0
    assert report['exclusions']['conflicting_verified_results'] == 1


def test_report_has_paired_counts_strata_and_no_automatic_promotion(monkeypatch):
    values = records(monkeypatch)
    before = deepcopy(values)
    report = ledger.evaluate(values, [result()])
    assert values == before
    assert report['paired']['matches'] == 1
    assert 'T60m' in report['by_horizon']
    assert 'PL' in report['by_league']
    assert report['paired']['mcnemar_exact_p'] is not None
    assert report['models']['always_1_1']['exact_hits'] == 1


def test_storage_normalized_and_raw_records_deduplicate_as_same_snapshot(monkeypatch, tmp_path):
    from forecast_ledger import append_file_record, read_file_records
    values = records(monkeypatch)
    for row in values:
        append_file_record(row, root=tmp_path)
    normalized = read_file_records(tmp_path)
    report = ledger.evaluate(values+normalized, [result()])
    assert report['paired_verified_slots'] == 1
    assert 'conflicting_forecast' not in report['exclusions']


def test_canonical_database_verification_status_is_accepted(monkeypatch):
    actual = result()
    actual.pop('evidence_version')
    actual['verification_status'] = 'verified'
    report = ledger.evaluate(records(monkeypatch), [actual])
    assert report['paired_verified_matches'] == 1


def test_different_horizons_are_reported_as_slots_not_independent_matches(monkeypatch):
    values = records(monkeypatch)
    later = deepcopy(values)
    for row in later:
        row['captured_at_utc'] = (NOW+timedelta(minutes=30)).isoformat()
        row['evidence_as_of_utc'] = row['captured_at_utc']
        row['horizon'] = 'T30m'
        row['pair_id'] = 'later-horizon'
    report = ledger.evaluate(values+later, [result()])
    assert report['paired_verified_slots'] == 2
    assert report['unique_verified_fixtures'] == 1
    assert report['common_unique_fixtures'] == 1
    assert set(report['by_horizon']) == {'T60m', 'T30m'}
    assert report['paired']['mcnemar_exact_p'] is None
    assert report['paired']['mcnemar_status'] == 'suppressed_repeated_fixture_horizons'
    assert report['paired']['repeated_fixture_slots'] == 1
    assert report['paired_by_horizon']['T60m']['mcnemar_exact_p'] is not None
    assert report['paired_by_horizon']['T30m']['mcnemar_exact_p'] is not None
    assert report['models']['v6']['exact_rate_wilson95'] is None
    assert report['by_horizon']['T60m']['v6']['exact_rate_wilson95'] is not None


def test_cli_persists_only_verified_matching_results_and_preserves_original_accuracy(monkeypatch, tmp_path):
    from forecast_ledger import append_file_record, read_file_records
    import evaluate_prospective_ledger as cli
    root, output, actuals = tmp_path/'ledger', tmp_path/'comparison.json', tmp_path/'verified.json'
    for row in records(monkeypatch):
        append_file_record(row, root=root)
    original_accuracy = tmp_path/'accuracy_report.json'
    original_accuracy.write_text('{"old_original_hits":14}\n')
    actuals.write_text(json.dumps({'1234': result(), '999': dict(result(), fixture_id=999)}))
    monkeypatch.setenv('DATABASE_URL', '')
    monkeypatch.setenv('FORECAST_DATABASE_URL', '')
    monkeypatch.setattr('sys.argv', ['evaluate_prospective_ledger.py', '--ledger-dir', str(root), '--results', str(actuals), '--output', str(output)])
    cli.main()
    report = json.loads(output.read_text())
    assert report['paired_verified_slots'] == 1
    assert report['persistence']['file_settlements_written'] == 1
    assert len(read_file_records(root, kind='settlements')) == 1
    assert original_accuracy.read_text() == '{"old_original_hits":14}\n'


def test_cli_reports_configured_database_failure_without_secret_or_false_durability(monkeypatch, tmp_path, capsys):
    from forecast_ledger import append_file_record
    import evaluate_prospective_ledger as cli
    root, output, actuals = tmp_path/'ledger', tmp_path/'comparison.json', tmp_path/'verified.json'
    for row in records(monkeypatch):
        append_file_record(row, root=root)
    actuals.write_text(json.dumps({'1234': result()}))
    secret = 'postgresql://user:do-not-print@example.invalid/database'
    monkeypatch.setenv('DATABASE_URL', secret)
    def unavailable(**kwargs):
        raise RuntimeError(secret)
    monkeypatch.setattr(cli, 'open_ledger', unavailable)
    monkeypatch.setattr('sys.argv', ['evaluate_prospective_ledger.py', '--ledger-dir', str(root), '--results', str(actuals), '--output', str(output)])
    with pytest.raises(SystemExit, match='incomplete ledger persistence'):
        cli.main()
    report = json.loads(output.read_text())
    assert report['paired_verified_slots'] == 1
    assert report['persistence']['state'] == 'partial'
    assert report['persistence']['database']['state'] == 'unavailable'
    assert secret not in output.read_text()+capsys.readouterr().out
