"""V32 rank-lift checks: tied/partial days never become accuracy evidence."""
import unittest

from v32_rank_discrimination import _auc, _contrast, _sign_test, summarize


def candidate(rank, hit):
    return {
        "fixture_keys": [["API-Football", str(rank)], ["API-Football", str(rank + 100)]],
        "quality_rank": rank,
        "robustness_rank": 5 - rank,
        "robust_five_hit": hit,
    }


def source(*, complete=True, records=None):
    records = records if records is not None else [
        candidate(1, True), candidate(2, True),
        candidate(3, False), candidate(4, False)]
    return {
        "overlapping_pairs_are_not_independent_samples": True,
        "original_frozen_days": 1,
        "frozen_candidate_pair_rows": len(records),
        "invalid_or_tampered_frozen_days_excluded": {},
        "invalid_independent_final_sources": {},
        "days": [{
            "sydney_day": "2026-11-22",
            "complete_slate": complete,
            "original_policy": "FROZEN",
            "candidate_count": len(records),
            "candidate_pairs_pending": 0 if complete else len(records),
            "candidate_rows": records if complete else [],
            "original_candidate_catalog_sha256": "immutable-synthetic-digest",
            "unique_fixtures_verified": 8 if complete else 0,
        }],
    }


class RankResearchTests(unittest.TestCase):
    def test_incomplete_day_remains_pending_not_loss(self):
        x = summarize(source(complete=False))
        self.assertEqual(x["completed_independently_verified_days"], 0)
        self.assertEqual(x["pending_or_incomplete_days"], 1)
        self.assertIsNone(x["within_day_rank_lift"]["quality_top_bottom_1"]
                          ["unweighted_mean_day_level_pair_fraction_difference"])
        self.assertFalse(x["initial_review_ready"])

    def test_true_quality_rank_lift_and_inverse_control(self):
        x = summarize(source())
        self.assertEqual(x["within_day_rank_lift"]["quality_top_bottom_1"]
                         ["unweighted_mean_day_level_pair_fraction_difference"], 1)
        self.assertEqual(x["within_day_rank_lift"]["robustness_top_bottom_1"]
                         ["unweighted_mean_day_level_pair_fraction_difference"], -1)
        self.assertEqual(x["days"][0]["policies"]["quality"]
                         ["within_day_hit_vs_miss_rank_concordance"], 1)
        self.assertEqual(x["days"][0]["policies"]["robustness"]
                         ["within_day_hit_vs_miss_rank_concordance"], 0)
        self.assertIsNone(x["within_day_rank_lift"]["quality_top_bottom_1"]
                          ["paired_day_sign_test_two_sided_exploratory"])

    def test_equal_nonoverlapping_tickets_with_small_slate(self):
        r = [candidate(1, True), candidate(2, False), candidate(3, False)]
        x = _contrast(r, "quality_rank", 10)
        self.assertEqual(x["actual_pairs_per_arm"], 1)
        self.assertEqual(x["tickets_per_arm"], 5)
        self.assertTrue(x["top_any_hit"])
        self.assertFalse(x["bottom_any_hit"])

    def test_null_rank_concordance_if_no_win_or_no_loss(self):
        self.assertIsNone(_auc([candidate(1, False), candidate(2, False)],
                               "quality_rank"))
        self.assertIsNone(_auc([candidate(1, True), candidate(2, True)],
                               "quality_rank"))

    def test_reject_tampered_or_invalid_ranks(self):
        rows = [candidate(1, True), candidate(1, False)]
        with self.assertRaisesRegex(ValueError, "invalid_original_rank_permutation"):
            summarize(source(records=rows))

    def test_sign_test_counts_days_not_correlated_pairs(self):
        self.assertIsNone(_sign_test(0, 0))
        self.assertEqual(_sign_test(10, 0), 2 / 1024)
        self.assertEqual(_sign_test(1, 1), 1)


if __name__ == "__main__":
    unittest.main()
