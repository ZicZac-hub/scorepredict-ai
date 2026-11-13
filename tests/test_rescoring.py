import json
import sqlite3
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from rescore_predictions import (normalize_records, rescore_records, provider_loaders,
                                 run_rescoring, _read_archive)
from revision_store import save_revision_report

NOW = datetime(2026, 11, 13, 12, tzinfo=timezone.utc)


def original(fid=12, kickoff='2026-11-21T12:00:00Z', provider='API-Football', capture='2026-11-01T12:00:00Z'):
    return {'provider': provider, 'fixture_id': fid, 'competition': 'PL',
            'kickoff_utc': kickoff, 'captured_at_utc': capture,
            'home_team': 'A', 'away_team': 'B', 'home_team_id': 1, 'away_team_id': 2,
            'predicted_score': [1, 1], 'forecast_model': 'v6', 'source': 'test-original'}


def fixture(row=None, actual=None):
    row = row or original()
    return {'id': row['fixture_id'], 'provider': row['provider'],
            'competition': {'id': 39, 'code': 'PL', 'season': 2026},
            'utcDate': row['kickoff_utc'], 'status': 'FINISHED' if actual else 'SCHEDULED',
            'provider_status': 'FT' if actual else 'NS',
            'homeTeam': {'id': 1, 'name': 'A'}, 'awayTeam': {'id': 2, 'name': 'B'},
            'score': {'fullTime': {'home': actual[0], 'away': actual[1]} if actual else {'home': None, 'away': None}, 'duration': 'REGULAR'}}


def game(fid=1, date='2026-10-15T12:00:00Z', provider='API-Football', competition=39):
    item = fixture(original(fid=fid, kickoff=date, provider=provider), actual=(2, 0))
    item['competition']['id'] = competition
    return item


def forecast(target, history, league_history=None):
    return {'scores': [(2, 0, .16), (1, 0, .14), (1, 1, .11)],
            'model_version': 'reference-informed-v7', 'home_xg': 2.1, 'away_xg': .7,
            'home_win': .65, 'draw': .20, 'away_win': .15}


def test_upcoming_uses_current_cutoff_and_keeps_original_unchanged():
    row = original()
    before = deepcopy(row)
    loader = Mock(return_value=[game(), game(2, '2026-11-13T11:00:00Z'),
                                game(3, '2026-11-14T12:00:00Z'), game(12, '2026-10-01T00:00:00Z'),
                                game(5, provider='football-data.org'), game(6, competition=78)])
    predictor = Mock(side_effect=forecast)
    progress = Mock()
    report = rescore_records([row], loader, lambda r:fixture(r), NOW, predictor, progress)
    revision = report['revisions'][0]
    assert revision['mode'] == 'upcoming'
    assert revision['history_as_of_utc'] == NOW.isoformat()
    assert revision['updated_top10_scores'][0] == [2, 0, .16]
    assert len(predictor.call_args.args[1]) == 1
    assert row == before
    assert revision['excluded_from_original_accuracy'] is True
    assert report['summary']['changed'] == 1
    progress.assert_called_once()


def test_finished_reconstruction_uses_original_capture_not_later_results():
    row = original(kickoff='2026-11-10T12:00:00Z')
    predictor = Mock(side_effect=forecast)
    history = [game(), game(2, '2026-11-02T00:00:00Z'), game(3, '2026-11-01T11:00:00Z')]
    report = rescore_records([row], lambda *args:history, lambda r:fixture(r, (2, 0)), NOW, predictor)
    revision = report['revisions'][0]
    assert revision['mode'] == 'retrospective'
    assert revision['history_as_of_utc'] == '2026-11-01T12:00:00+00:00'
    assert revision['actual_score'] == [2, 0]
    assert revision['original_hit'] is False and revision['updated_hit'] is True
    assert predictor.call_args.args[0]['score']['fullTime'] == {'home': None, 'away': None}
    assert predictor.call_args.args[0]['status'] == 'SCHEDULED'
    assert len(predictor.call_args.args[1]) == 1


@pytest.mark.parametrize('alter', ['provider', 'teams', 'kickoff'])
def test_wrong_identity_never_reaches_predictor(alter):
    def wrong(row):
        item = fixture(row)
        if alter == 'provider': item['provider'] = 'football-data.org'
        if alter == 'teams': item['homeTeam']['id'] = 99
        if alter == 'kickoff': item['utcDate'] = '2026-11-20T12:00:00Z'
        return item
    predictor = Mock()
    result = rescore_records([original()], Mock(), wrong, NOW, predictor)
    assert result['summary']['skipped'] == 1
    predictor.assert_not_called()


def test_after_kickoff_original_is_excluded_with_reason():
    row = original(capture='2026-11-22T00:00:00Z')
    result = rescore_records([row], Mock(), Mock(), NOW, forecast)
    assert not result['revisions']
    assert 'before kickoff' in result['skipped'][0]['reason']


def test_normalization_preserves_earliest_and_provider_collisions():
    first = original()
    later = original(capture='2026-11-02T12:00:00Z')
    later['predicted_score'] = [3, 1]
    archive = {'provider': 'football-data.org', 'fixture_id': 12, 'league': 'England · Premier League (PL)',
               'saved_at': first['captured_at_utc'], 'kickoff': first['kickoff_utc'],
               'home': 'A', 'away': 'B', 'home_id': '1', 'away_id': '2',
               'predicted_score': '2-1', 'model_version': 'stored-v6'}
    rows = normalize_records([later, first], [archive])
    assert len(rows) == 2
    assert rows[0]['predicted_score'] == [1, 1]
    assert rows[1]['provider'] == 'football-data.org'
    assert rows[1]['forecast_model'] == 'stored-v6'
    assert rows[1]['competition'] == 'PL'


def test_append_only_revisions_archive_previous_latest(tmp_path):
    old = {'fixture_id': 12, 'updated_score': [1, 1], 'updated_model': 'v6'}
    old_pointer = tmp_path / '12.json'
    old_pointer.write_text(json.dumps(old))
    report = rescore_records([original()], lambda *args:[game()], lambda r:fixture(r), NOW, forecast)
    save_revision_report(report, tmp_path)
    originals = list((tmp_path / 'previous_latest' / '12.json').glob('*.json'))
    assert len(originals) == 1
    assert json.loads(originals[0].read_text()) == old
    immutable = tmp_path / 'runs' / (report['run_id'] + '.json')
    before = immutable.read_bytes()
    revised_report = deepcopy(report)
    revised_report['summary']['changed'] = 999
    with pytest.raises(ValueError, match='different contents'):
        save_revision_report(revised_report, tmp_path)
    assert immutable.read_bytes() == before


def raw_game(fid=12, season=2026):
    return {'fixture': {'id': fid, 'date': '2026-11-21T12:00:00Z', 'status': {'short': 'NS'}},
            'league': {'id': 39, 'season': season},
            'teams': {'home': {'id': 1, 'name': 'A'}, 'away': {'id': 2, 'name': 'B'}},
            'goals': {'home': None, 'away': None}, 'score': {'fulltime': {'home': None, 'away': None}}}


def test_history_uses_fixture_seasons_and_caches_across_records():
    row = original(kickoff='2022-11-21T12:00:00Z', capture='2022-11-01T12:00:00Z')
    target = fixture(row)
    target['competition']['season'] = 2022
    with patch('rescore_predictions._request', return_value=[]) as request:
        history_loader, _ = provider_loaders(api_key='test-only')
        history_loader(row, target, NOW)
        history_loader(row, target, NOW)
    assert [call.args[2]['season'] for call in request.call_args_list] == [2022, 2021]


def test_legacy_fixture_lookup_can_fall_back_after_api_error():
    row = original(provider='legacy-unknown')
    primary = fixture(original(provider='football-data.org'))
    with patch('rescore_predictions._request', side_effect=RuntimeError('quota')):
        _, loader = provider_loaders(api_key='test-only', primary_get=Mock(return_value=primary))
        result = loader(row)
    assert result['provider'] == 'football-data.org'


def test_missing_prior_season_does_not_discard_available_history():
    available = raw_game(fid=100)
    available['fixture'].update(date='2026-10-15T12:00:00Z', status={'short':'FT'})
    available['goals'] = {'home':2,'away':0}
    available['score']['fulltime'] = {'home':2,'away':0}
    with patch('rescore_predictions._request', side_effect=[[available], RuntimeError('not licensed')]):
        loader, _ = provider_loaders(api_key='test-only')
        result = loader(original(), fixture(), NOW)
    assert len(result['matches']) == 1
    assert any('history unavailable' in warning for warning in result['warnings'])


def test_read_sqlite_archive_does_not_migrate_or_mutate_it(tmp_path):
    path = tmp_path / 'original.sqlite3'
    conn = sqlite3.connect(path)
    conn.execute('CREATE TABLE predictions (fixture_id INTEGER, predicted_score TEXT)')
    conn.execute('INSERT INTO predictions VALUES (12, "1-1")')
    conn.commit(); conn.close()
    before = path.read_bytes()
    assert _read_archive(path) == [{'fixture_id': 12, 'predicted_score': '1-1'}]
    assert path.read_bytes() == before


def test_run_rescoring_accepts_frozen_canonical_records(tmp_path):
    canonical = [original()]
    with patch('rescore_predictions.provider_loaders', return_value=(lambda *args:[game()], lambda r:fixture(r))):
        report = run_rescoring(records=canonical, now=NOW, output_dir=tmp_path, predictor=forecast)
    assert report['summary']['revised'] == 1
    assert canonical[0]['predicted_score'] == [1, 1]


def test_mixed_double_future_leg_replays_at_pair_capture_without_replacing_single(tmp_path):
    single = original()
    pair_leg = dict(single, force_retrospective=True,
                    requested_as_of_utc='2026-11-03T12:00:00Z', comparison_context='saved-pair-1')
    predictor = Mock(side_effect=forecast)
    report = rescore_records([single, pair_leg], lambda *args:[game()], lambda r:fixture(r), NOW, predictor)
    assert [row['mode'] for row in report['revisions']] == ['upcoming', 'retrospective']
    retrospective = report['revisions'][1]
    assert retrospective['history_as_of_utc'] == '2026-11-03T12:00:00+00:00'
    assert retrospective['original_captured_at_utc'] == single['captured_at_utc']
    assert retrospective['comparison_context'] == 'saved-pair-1'
    assert retrospective['revision_id'] != report['revisions'][0]['revision_id']
    save_revision_report(report, tmp_path)
    assert len(list((tmp_path / 'records' / 'API-Football' / '12').glob('*.json'))) == 2
    assert json.loads((tmp_path / '12.json').read_text())['mode'] == 'upcoming'


def test_requested_replay_cutoff_cannot_reach_target_result():
    row = dict(original(), force_retrospective=True, requested_as_of_utc='2026-12-01T00:00:00Z')
    result = rescore_records([row], Mock(), lambda r:fixture(r), NOW, Mock())
    assert result['summary']['skipped'] == 1
    assert 'cutoff' in result['skipped'][0]['reason']


def test_invalid_candidate_is_skipped_without_breaking_other_records():
    def bad(target, history, league_history=None):
        result = forecast(target, history)
        result['scores'] = [(2, 0, float('nan'))]
        return result
    report = rescore_records([original()], lambda *args:[game()], lambda r:fixture(r), NOW, bad)
    assert report['summary']['skipped'] == 1
    assert 'invalid exact-score' in report['skipped'][0]['reason']


@pytest.mark.parametrize('status', ['SCHEDULED', 'TIMED', 'LIVE', 'CANCELLED'])
def test_past_unfinished_or_cancelled_fixture_is_not_a_finished_comparison(status):
    row = original(kickoff='2026-11-10T12:00:00Z')
    target = fixture(row)
    target['status'] = status
    predictor = Mock()
    report = rescore_records([row], Mock(), lambda record:target, NOW, predictor)
    assert report['summary']['retrospective_comparisons'] == 0
    assert report['summary']['skipped'] == 1
    assert 'not finished' in report['skipped'][0]['reason']
    predictor.assert_not_called()


def test_pending_replay_without_pair_context_is_not_a_finished_comparison():
    row = dict(original(), force_retrospective=True, requested_as_of_utc='2026-11-03T00:00:00Z')
    report = rescore_records([row], Mock(), lambda record:fixture(record), NOW, Mock())
    assert report['summary']['retrospective_comparisons'] == 0
    assert report['summary']['skipped'] == 1


def test_provider_secret_is_absent_from_reports_artifacts_and_progress(tmp_path):
    secret = 'private-provider-key-987654321'
    extra_secret = 'private-primary-key-123456789'
    progress = Mock()
    def failing_loader(*args):
        raise RuntimeError('provider error echoed ' + secret + ' and ' + extra_secret)
    with patch('rescore_predictions.provider_loaders', return_value=(failing_loader, lambda row:fixture(row))):
        report = run_rescoring(api_key=secret, records=[original()], now=NOW,
                               output_dir=tmp_path, predictor=forecast, progress=progress,
                               redact_values=(extra_secret,))
    assert secret not in json.dumps(report) and extra_secret not in json.dumps(report)
    assert '[REDACTED]' in report['skipped'][0]['reason']
    assert secret not in str(progress.call_args_list)
    assert extra_secret not in str(progress.call_args_list)
    for path in tmp_path.rglob('*.json'):
        assert secret not in path.read_text()
        assert extra_secret not in path.read_text()
