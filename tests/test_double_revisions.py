import copy
import itertools
import json
import sqlite3
import pytest
from pathlib import Path

from double_tracker import save, saved_pairs, settle
from double_revisions import load_saved_pairs, revise_saved_pairs, persist_double_revisions
from smart_double_ai import five_doubles, rank_pairs

NOW = '2026-11-13T12:00:00Z'
SCORES = [(1, 1, .2), (2, 1, .15), (1, 0, .12), (2, 0, .1), (0, 0, .08), (3, 1, .07)]


def fixture(idx, provider='API-Football', kickoff='2026-11-15T12:00:00Z', model='reference-informed-v7'):
    return {'fixture': {'id': idx, 'provider': provider,
                        'homeTeam': {'id': idx*10, 'name': f'A{idx}'},
                        'awayTeam': {'id': idx*10+1, 'name': f'B{idx}'}},
            'kickoff': kickoff, 'result': {'scores': SCORES, 'model_version': model}}


def pair(kickoff='2026-11-15T12:00:00Z'):
    return {'pair_key': 'original-1', 'source': 'test',
            'captured_at_utc': '2026-11-10T10:00:00Z',
            'first_provider': 'API-Football', 'second_provider': 'API-Football',
            'first_fixture_id': '1', 'second_fixture_id': '2',
            'first_match': 'A1 vs B1', 'second_match': 'A2 vs B2',
            'first_kickoff_utc': kickoff, 'second_kickoff_utc': kickoff,
            'original_five_doubles': [{'score1': '0-0', 'score2': '0-0', 'probability': .01}]}


def revisions(mode='upcoming', cutoff=NOW, kickoff='2026-11-15T12:00:00Z'):
    return [{'provider': 'API-Football', 'fixture_id': idx, 'mode': mode,
             'history_as_of_utc': cutoff, 'rescored_at_utc': NOW,
             'original_kickoff_utc': kickoff, 'kickoff_utc': kickoff,
             'updated_model': 'reference-informed-v7', 'updated_top10_scores': SCORES,
             # Results must have no influence on the selected five alternatives.
             'actual_score': [7, 7]} for idx in (1, 2)]


def test_five_pairs_maximize_independent_covered_mass():
    a, b = fixture(1), fixture(2)
    selected = five_doubles(a, b)
    brute = sorted([p*q for _, _, p in SCORES for _, _, q in SCORES], reverse=True)[:5]
    assert [row['Joint'] for row in selected] == brute
    assert len({(row['Score 1'], row['Score 2']) for row in selected}) == 5
    assert sum(row['Joint'] for row in selected) == sum(brute)


def test_duplicates_and_invalid_probabilities_cannot_inflate_coverage():
    a, b = fixture(1), fixture(2)
    a['result']['scores'] = SCORES + [SCORES[0], (9, 9, float('nan')), (8, 8, -1)]
    assert five_doubles(a, b) == five_doubles(fixture(1), b)
    a['result']['scores'] = [(1, 1, .8), (2, 2, .8)]
    assert five_doubles(a, b) == []


def test_same_fixture_and_cross_provider_duplicate_never_form_two_legs():
    a, b = fixture(1), fixture(1)
    assert five_doubles(a, b) == []
    b['fixture']['provider'] = 'football-data.org'
    b['fixture']['id'] = 999
    b['kickoff'] = '2026-11-15T14:00:00+02:00'
    assert five_doubles(a, b) == []


def test_same_number_on_different_providers_is_not_automatically_same_match():
    a, b = fixture(1), fixture(1, 'football-data.org')
    b['fixture']['homeTeam']['name'] = 'Different home'
    b['fixture']['awayTeam']['name'] = 'Different away'
    assert len(five_doubles(a, b)) == 5


def test_pair_day_compares_utc_not_date_substrings():
    a = fixture(1, kickoff='2026-11-15T23:00:00-02:00')
    b = fixture(2, kickoff='2026-11-16T12:00:00Z')
    assert len(rank_pairs([a, b])) == 1


def test_saves_are_prematch_immutable_and_per_model_version():
    conn = sqlite3.connect(':memory:')
    a, b = fixture(1), fixture(2)
    options = five_doubles(a, b)
    assert save(conn, a, b, options, now=NOW)
    original = saved_pairs(conn)[0]['predictions_json']
    assert not save(conn, a, b, list(reversed(options)), now=NOW)
    assert saved_pairs(conn)[0]['predictions_json'] == original
    a['result']['model_version'] = 'reference-informed-v8'
    assert save(conn, a, b, options, now=NOW)
    assert len(saved_pairs(conn)) == 2
    assert not save(conn, a, b, options, now='2026-11-16T00:00:00Z')


def test_legacy_none_provider_never_settles_from_assumed_provider():
    conn = sqlite3.connect(':memory:')
    conn.execute('''CREATE TABLE exact_score_doubles(snapshot_key TEXT PRIMARY KEY,saved_at TEXT,
        first_id TEXT,second_id TEXT,first_name TEXT,second_name TEXT,predictions_json TEXT)''')
    original = json.dumps([{'score1': '1-1', 'score2': '1-1', 'probability': .04}])
    conn.execute('INSERT INTO exact_score_doubles VALUES (?,?,?,?,?,?,?)',
                 ('2026-11-10:None:1:API-Football:2', NOW, '1', '2', 'A', 'B', original))
    conn.commit()
    result = settle(conn, {('football-data.org', '1'): '1-1', ('API-Football', '2'): '1-1'})
    assert result[0]['Status'] == 'Pending'
    assert saved_pairs(conn)[0]['predictions_json'] == original


def test_upcoming_revision_preserves_original_and_ignores_actual_results():
    original = pair()
    before = copy.deepcopy(original)
    result = revise_saved_pairs([original], revisions(), now=NOW, model_version='reference-informed-v7')
    revised = result['revisions'][0]
    assert original == before
    assert revised['mode'] == 'upcoming'
    assert revised['excluded_from_original_accuracy'] is True
    assert revised['original_five_doubles'] == original['original_five_doubles']
    updated = copy.deepcopy(revisions())
    updated[0]['actual_score'] = [0, 0]
    updated[1]['actual_score'] = [0, 0]
    assert revise_saved_pairs([original], updated, now=NOW)['revisions'][0]['updated_five_doubles'] == revised['updated_five_doubles']


def test_retrospective_rejects_post_capture_information():
    old = pair(kickoff='2026-11-11T12:00:00Z')
    late = revisions(mode='retrospective', cutoff='2026-11-11T11:00:00Z', kickoff=old['first_kickoff_utc'])
    result = revise_saved_pairs([old], late, now=NOW)
    assert not result['revisions']
    assert 'original pair cutoff' in result['skipped'][0]['reason']
    early = revisions(mode='retrospective', cutoff='2026-11-10T09:00:00Z', kickoff=old['first_kickoff_utc'])
    result = revise_saved_pairs([old], early, now=NOW)
    assert result['retrospective_pairs'] == 1
    assert result['revisions'][0]['excluded_from_original_accuracy']
    assert 'Retrospective' in result['revisions'][0]['label']


def test_wrong_provider_missing_leg_and_ambiguous_original_are_reported():
    old = pair()
    wrong = revisions()
    wrong[0]['provider'] = 'football-data.org'
    assert len(revise_saved_pairs([old], wrong, now=NOW)['skipped']) == 1
    old['first_provider'] = 'legacy-unknown'
    result = revise_saved_pairs([old], revisions(), now=NOW)
    assert 'provider unknown' in result['skipped'][0]['reason']


def test_model_fallback_remains_honestly_identified():
    rev = revisions()
    rev[0]['updated_model'] = 'v7-fallback-v6'
    result = revise_saved_pairs([pair()], rev, now=NOW, model_version='reference-informed-v7')
    assert result['revisions'][0]['updated_models'][0] == 'v7-fallback-v6'


def test_persistence_appends_and_does_not_rewrite_revision(tmp_path):
    report = revise_saved_pairs([pair()], revisions(), now=NOW)
    persist_double_revisions(report, tmp_path)
    record = report['revisions'][0]
    path = tmp_path/'revisions'/(record['revision_id']+'.json')
    before = path.read_bytes()
    changed = copy.deepcopy(report)
    changed['revisions'][0]['label'] = 'Changed caller record'
    with pytest.raises(ValueError, match='Conflicting immutable revision'):
        persist_double_revisions(changed, tmp_path)
    assert path.read_bytes() == before
    newer = copy.deepcopy(report)
    newer['rescored_at_utc'] = '2026-11-13T13:00:00Z'
    persist_double_revisions(newer, tmp_path)
    assert len(list((tmp_path/'runs').glob('*.json'))) == 2
    assert json.loads((tmp_path/'latest_report.json').read_text())['saved_pairs'] == 1


def test_load_sqlite_and_permanent_originals_does_not_mutate_files(tmp_path):
    conn = sqlite3.connect(':memory:')
    a, b = fixture(1), fixture(2)
    assert save(conn, a, b, five_doubles(a, b), now=NOW)
    archive = tmp_path/'pairs'
    archive.mkdir()
    raw = pair()
    raw['original'] = [{'Score 1': '1-1', 'Score 2': '2-1', 'Joint': .03}]
    (archive/'pair.json').write_text(json.dumps(raw))
    before = (archive/'pair.json').read_bytes()
    rows = load_saved_pairs(conn, archive, tmp_path/'absent')
    assert len(rows) == 2
    assert rows[1]['first_provider'] == 'API-Football'
    assert (archive/'pair.json').read_bytes() == before


def test_corrupt_sqlite_pair_is_reported_without_blocking_valid_pairs(tmp_path):
    conn = sqlite3.connect(':memory:')
    a, b = fixture(1), fixture(2)
    assert save(conn, a, b, five_doubles(a, b), now=NOW)
    conn.execute("UPDATE exact_score_doubles SET predictions_json='invalid'")
    conn.commit()
    loaded = load_saved_pairs(conn, tmp_path/'absent', tmp_path/'absent2')
    assert len(loaded) == 1 and loaded[0]['load_error']
    report = revise_saved_pairs(loaded, revisions(), now=NOW)
    assert len(report['skipped']) == 1


def test_retrospective_prefers_explicit_pair_context_revisions():
    original = pair(kickoff='2026-11-11T12:00:00Z')
    generic = revisions(mode='retrospective', cutoff='2026-11-10T09:00:00Z', kickoff=original['first_kickoff_utc'])
    contextual = copy.deepcopy(generic)
    for leg in contextual:
        leg['comparison_context'] = original['pair_key']
        leg['revision_id'] = 'pair-specific-'+str(leg['fixture_id'])
    report = revise_saved_pairs([original], generic+contextual, now=NOW)
    assert report['revisions'][0]['leg_revision_ids'] == ['pair-specific-1','pair-specific-2']


def test_failed_serialization_never_leaves_a_partial_immutable_record(tmp_path):
    from double_revisions import _write_once
    path = tmp_path/'revision.json'
    with pytest.raises(TypeError):
        _write_once(path, {'bad': object()})
    assert not path.exists()
    assert not list(tmp_path.glob('*.tmp'))
    assert _write_once(path, {'complete': True})
    assert json.loads(path.read_text()) == {'complete': True}
