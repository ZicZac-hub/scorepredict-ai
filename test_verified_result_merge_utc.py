"""Regression tests for verified final-score identity merging."""
import unittest

from reconcile_capture_conflicts import merge_verified


class VerifiedResultMergeTests(unittest.TestCase):
    def setUp(self):
        self.original = {
            "provider": "API-Football", "fixture_id": 1492450,
            "kickoff_utc": "2026-11-19T00:30:00+00:00",
            "home_team_id": 131, "away_team_id": 120,
            "score_basis": "regulation", "home_goals": 4, "away_goals": 0,
            "evidence_version": 2, "verified_at_utc": "2026-11-19T04:25:07+00:00",
        }

    def test_equivalent_utc_spelling_preserves_first_published_proof(self):
        alternate = dict(self.original, kickoff_utc="2026-11-19T00:30:00Z",
                         verified_at_utc="2026-11-19T02:51:18+00:00")
        result = merge_verified({"1492450": self.original}, {"1492450": alternate})
        self.assertEqual(result["1492450"], self.original)

    def test_equivalent_offset_preserves_first_published_proof(self):
        alternate = dict(self.original, kickoff_utc="2026-11-19T11:30:00+11:00")
        self.assertEqual(merge_verified({"1492450": self.original},
                                        {"1492450": alternate})["1492450"], self.original)

    def test_different_kickoff_is_rejected(self):
        alternate = dict(self.original, kickoff_utc="2026-11-19T00:31:00Z")
        with self.assertRaisesRegex(ValueError, "Conflicting final-score identity"):
            merge_verified({"1492450": self.original}, {"1492450": alternate})

    def test_changed_score_is_rejected(self):
        alternate = dict(self.original, kickoff_utc="2026-11-19T00:30:00Z",
                         home_goals=3)
        with self.assertRaisesRegex(ValueError, "Conflicting final-score identity"):
            merge_verified({"1492450": self.original}, {"1492450": alternate})

    def test_naive_kickoff_is_rejected(self):
        alternate = dict(self.original, kickoff_utc="2026-11-19T00:30:00")
        with self.assertRaisesRegex(ValueError, "Conflicting final-score identity"):
            merge_verified({"1492450": self.original}, {"1492450": alternate})


if __name__ == "__main__":
    unittest.main()
