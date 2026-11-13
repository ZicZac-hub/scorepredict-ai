"""Archive regression tests: provider IDs, migration, and fixture identity."""
import sqlite3
import tempfile
import threading
import unittest
from pathlib import Path

from history_store import _migrate, connect, identity_matches, load, reconcile, save_predictions


LEGACY_SCHEMA = '''CREATE TABLE predictions (
 fixture_id INTEGER PRIMARY KEY, saved_at TEXT NOT NULL, kickoff TEXT, league TEXT,
 home TEXT, away TEXT, predicted_score TEXT, home_prob REAL, draw_prob REAL,
 away_prob REAL, scores_json TEXT, actual_score TEXT)'''


def forecast(provider='API-Football', fixture_id=123, score=(1, 0)):
    return {'fixture': {'id': fixture_id, 'provider': provider,
                        'homeTeam': {'id': 10, 'name': 'Team A'},
                        'awayTeam': {'id': 20, 'name': 'Team B'}},
            'result': {'scores': [(score[0], score[1], .12)],
                       'home_win': .5, 'draw': .3, 'away_win': .2},
            'kickoff': '2100-01-01T12:00:00Z', 'league': 'Test'}


def final(provider='API-Football', fixture_id=123, score=(2, 0)):
    return {'id': fixture_id, 'provider': provider, 'status': 'FINISHED',
            'utcDate': '2100-01-01T12:00:00+00:00',
            'homeTeam': {'id': 10, 'name': 'Team A'},
            'awayTeam': {'id': 20, 'name': 'Team B'},
            'score': {'duration': 'REGULAR', 'fullTime': {'home': score[0], 'away': score[1]}}}


class ArchiveTests(unittest.TestCase):
    def setUp(self):
        self.conn = connect(':memory:')

    def tearDown(self):
        self.conn.close()

    def test_same_number_from_two_providers_is_not_a_collision(self):
        self.assertEqual(save_predictions(self.conn, [forecast(), forecast('football-data.org')]), 2)
        self.assertEqual(reconcile(self.conn, [final()]), 1)
        records = {row['provider']: row for row in load(self.conn)}
        self.assertEqual(records['API-Football']['actual_score'], '2-0')
        self.assertIsNone(records['football-data.org']['actual_score'])

    def test_first_forecast_is_immutable(self):
        self.assertEqual(save_predictions(self.conn, [forecast()]), 1)
        original = load(self.conn)[0]
        self.assertEqual(save_predictions(self.conn, [forecast(score=(3, 1))]), 0)
        self.assertEqual(load(self.conn)[0], original)

    def test_post_kickoff_snapshot_is_never_saved(self):
        row = forecast()
        row['kickoff'] = '2000-01-01T12:00:00Z'
        self.assertEqual(save_predictions(self.conn, [row]), 0)
        self.assertEqual(load(self.conn), [])

    def test_incomplete_team_identity_is_not_saved(self):
        row = forecast()
        del row['fixture']['awayTeam']['id']
        self.assertEqual(save_predictions(self.conn, [row]), 0)

    def test_wrong_provider_team_or_kickoff_cannot_settle(self):
        save_predictions(self.conn, [forecast('football-data.org')])
        candidates = [final(), final('football-data.org')]
        candidates[1]['homeTeam']['id'] = 99
        wrong_time = final('football-data.org')
        wrong_time['utcDate'] = '2100-01-01T13:00:00Z'
        candidates.append(wrong_time)
        incomplete = final('football-data.org')
        del incomplete['awayTeam']['id']
        candidates.append(incomplete)
        self.assertEqual(reconcile(self.conn, candidates), 0)
        self.assertIsNone(load(self.conn)[0]['actual_score'])
        self.assertEqual(reconcile(self.conn, [final('football-data.org')]), 1)

    def test_equivalent_utc_offsets_identify_same_fixture(self):
        save_predictions(self.conn, [forecast()])
        match = final()
        match['utcDate'] = '2100-01-01T23:00:00+11:00'
        self.assertTrue(identity_matches(load(self.conn)[0], match))
        self.assertEqual(reconcile(self.conn, [match]), 1)

    def test_non_regulation_and_invalid_scores_are_not_settled(self):
        save_predictions(self.conn, [forecast()])
        extra_time = final()
        extra_time['score']['duration'] = 'EXTRA_TIME'
        boolean = final(score=(True, 0))
        negative = final(score=(-1, 0))
        missing = final(score=(None, 0))
        self.assertEqual(reconcile(self.conn, [extra_time, boolean, negative, missing]), 0)

    def test_verified_score_cannot_be_rewritten(self):
        save_predictions(self.conn, [forecast()])
        self.assertEqual(reconcile(self.conn, [final()]), 1)
        self.assertEqual(reconcile(self.conn, [final(score=(4, 4))]), 0)
        self.assertEqual(load(self.conn)[0]['actual_score'], '2-0')
        self.assertEqual(load(self.conn)[0]['verification_status'], 'verified')

    def test_primary_native_schema_defaults_only_to_primary_provider(self):
        save_predictions(self.conn, [forecast('football-data.org')])
        match = final('football-data.org')
        del match['provider']
        self.assertEqual(reconcile(self.conn, [match]), 1)


class MigrationTests(unittest.TestCase):
    def create_legacy(self, path):
        conn = sqlite3.connect(path)
        conn.execute(LEGACY_SCHEMA)
        conn.execute('INSERT INTO predictions VALUES (?,?,?,?,?,?,?,?,?,?,?,?)',
                     (123, '2099-12-01T12:00:00Z', '2100-01-01T12:00:00Z', 'Test',
                      'Team A', 'Team B', '1-0', .5, .3, .2, '[[1,0,0.12]]', '7-6'))
        conn.commit()
        conn.close()

    def test_waiting_session_rechecks_schema_after_other_session_migrates(self):
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / 'archive.sqlite3')
            self.create_legacy(path)
            before_lock = threading.Event()
            allow_lock = threading.Event()
            errors = []

            class PausingConnection(sqlite3.Connection):
                def execute(self, sql, *args, **kwargs):
                    if sql == 'BEGIN IMMEDIATE':
                        before_lock.set()
                        if not allow_lock.wait(5):
                            raise TimeoutError('Test failed to release the waiting migration')
                    return super().execute(sql, *args, **kwargs)

            def waiting_session():
                conn = sqlite3.connect(path, factory=PausingConnection)
                try:
                    _migrate(conn)
                except Exception as exc:
                    errors.append(exc)
                finally:
                    conn.close()

            worker = threading.Thread(target=waiting_session)
            worker.start()
            conn = None
            try:
                self.assertTrue(before_lock.wait(5))
                # The worker has observed the old schema. This session now
                # migrates and verifies it before the worker gets its lock.
                conn = connect(path)
                self.assertEqual(reconcile(conn, [final()]), 1)
            finally:
                allow_lock.set()
                worker.join(5)
            self.assertFalse(worker.is_alive())
            self.assertEqual(errors, [])
            row = load(conn)[0]
            self.assertEqual(row['provider'], 'API-Football')
            self.assertEqual(row['verification_status'], 'verified')
            self.assertEqual(row['actual_score'], '2-0')
            tables = [row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")]
            self.assertEqual(sorted(tables), ['predictions', 'predictions_legacy_v1'])
            conn.close()

    def test_failed_migration_rolls_back_original_table(self):
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / 'archive.sqlite3')
            conn = sqlite3.connect(path)
            # An unexpected older shape must fail without partially renaming
            # or replacing the user's archive.
            conn.execute('CREATE TABLE predictions (fixture_id INTEGER PRIMARY KEY, actual_score TEXT)')
            conn.execute("INSERT INTO predictions VALUES (123, '7-6')")
            conn.commit()
            conn.close()
            with self.assertRaises(sqlite3.OperationalError):
                connect(path)
            conn = sqlite3.connect(path)
            self.assertEqual(conn.execute('SELECT actual_score FROM predictions').fetchone()[0], '7-6')
            self.assertIsNone(conn.execute("SELECT 1 FROM sqlite_master WHERE name='predictions_legacy_v1'").fetchone())
            conn.close()

    def test_legacy_results_are_quarantined_and_originals_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / 'archive.sqlite3')
            self.create_legacy(path)
            conn = connect(path)
            row = load(conn)[0]
            self.assertEqual(row['provider'], 'legacy-unknown')
            self.assertIsNone(row['actual_score'])
            self.assertEqual(row['legacy_actual_score'], '7-6')
            self.assertEqual(row['verification_status'], 'legacy_unverified')
            self.assertEqual(row['predicted_score'], '1-0')
            self.assertEqual(conn.execute('SELECT actual_score FROM predictions_legacy_v1').fetchone()[0], '7-6')
            conn.close()
            conn = connect(path)
            self.assertEqual(len(load(conn)), 1)
            self.assertIsNone(load(conn)[0]['actual_score'])
            conn.close()

    def test_legacy_needs_team_names_and_kickoff_before_reidentification(self):
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / 'archive.sqlite3')
            self.create_legacy(path)
            conn = connect(path)
            unrelated = final()
            unrelated['homeTeam']['name'] = 'Other Team'
            self.assertEqual(reconcile(conn, [unrelated]), 0)
            self.assertEqual(load(conn)[0]['provider'], 'legacy-unknown')
            self.assertEqual(reconcile(conn, [final()]), 1)
            row = load(conn)[0]
            self.assertEqual(row['provider'], 'API-Football')
            self.assertEqual(row['actual_score'], '2-0')
            self.assertEqual(row['home_id'], '10')
            self.assertEqual(row['legacy_actual_score'], '7-6')
            self.assertEqual(row['predicted_score'], '1-0')
            conn.close()

    def test_new_save_reidentifies_legacy_without_replacing_original_forecast(self):
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / 'archive.sqlite3')
            self.create_legacy(path)
            conn = connect(path)
            original = load(conn)[0]
            self.assertEqual(save_predictions(conn, [forecast(score=(3, 1))]), 0)
            rows = load(conn)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]['provider'], 'API-Football')
            self.assertEqual(rows[0]['predicted_score'], original['predicted_score'])
            self.assertEqual(rows[0]['saved_at'], original['saved_at'])
            self.assertEqual(rows[0]['verification_status'], 'legacy_unverified')
            self.assertEqual(rows[0]['legacy_actual_score'], '7-6')
            self.assertIsNone(rows[0]['actual_score'])
            self.assertEqual(reconcile(conn, [final()]), 1)
            self.assertEqual(load(conn)[0]['predicted_score'], '1-0')
            conn.close()

    def test_different_match_sharing_legacy_id_is_saved_separately(self):
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / 'archive.sqlite3')
            self.create_legacy(path)
            conn = connect(path)
            unrelated = forecast(score=(3, 1))
            unrelated['fixture']['homeTeam'] = {'id': 99, 'name': 'Other Team'}
            self.assertEqual(save_predictions(conn, [unrelated]), 1)
            rows = {row['provider']: row for row in load(conn)}
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows['legacy-unknown']['predicted_score'], '1-0')
            self.assertEqual(rows['API-Football']['predicted_score'], '3-1')
            conn.close()


if __name__ == '__main__':
    unittest.main()
