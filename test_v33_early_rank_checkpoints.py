"""Synthetic V33 checkpoint tests: no hindsight, no partial-result losses."""
import unittest
from v33_early_rank_checkpoints import summarize, _checkpoint


def c(rank):
    return {"fixture_keys": [["API-Football", str(rank)], ["API-Football", str(rank + 100)]],
            "quality_rank_score": 10 - rank,
            "robustness_rank_score": rank}


def report(verified_rows):
    return {"overlapping_pairs_are_not_independent_samples": True,
            "original_frozen_days": 1, "frozen_candidate_pair_rows": 4,
            "fully_verified_slate_days": int(len(verified_rows) == 4),
            "invalid_or_tampered_frozen_days_excluded": {},
            "invalid_independent_final_sources": {},
            "days": [{
                "sydney_day": "2026-11-23",
                "original_policy": "FROZEN",
                "candidate_count": 4,
                "candidate_pairs_verified": len(verified_rows),
                "candidate_pairs_pending": 4-len(verified_rows),
                "candidate_rows": verified_rows,
                "complete_slate": len(verified_rows) == 4,
                "original_candidate_catalog_sha256": "frozen-test-hash",
            }]}


def observed(rank, hit):
    return {"fixture_keys": c(rank)["fixture_keys"], "quality_rank": rank,
            "robustness_rank": 5-rank, "robust_five_hit": hit}


class EarlyCheckpointTests(unittest.TestCase):
    def test_early_settlement_before_full_slate(self):
        x = summarize(report([observed(1, True), observed(4, False)]),
                      {"2026-11-23": [c(i) for i in range(1,5)]})
        quality = x["original_rank_checkpoint_results"]["quality_top_bottom_1"]
        self.assertEqual(quality["verified_day_units"], 1)
        self.assertEqual(quality["unweighted_mean_day_level_pair_fraction_difference"], 1)
        self.assertEqual(x["fully_verified_daily_slates_v32_basis"], 0)
        self.assertFalse(quality["statistical_review_ready"])
        self.assertEqual(x["original_rank_checkpoint_results"]["quality_top_bottom_5"]["verified_day_units"], 0)

    def test_missing_bottom_cannot_be_scored_as_miss(self):
        x = summarize(report([observed(1, True)]),
                      {"2026-11-23": [c(i) for i in range(1,5)]})
        stat = x["original_rank_checkpoint_results"]["quality_top_bottom_1"]
        self.assertEqual(stat["pending_day_units"], 1)
        self.assertIsNone(stat["unweighted_mean_day_level_pair_fraction_difference"])

    def test_original_full_slate_rank_even_if_only_middle_settled(self):
        x = summarize(report([observed(2, True), observed(3, False)]),
                      {"2026-11-23": [c(i) for i in range(1,5)]})
        checkpoint = x["days"][0]["checkpoints"]["quality"]["1"]
        self.assertEqual(checkpoint["status"], "pending")
        self.assertEqual(checkpoint["pending_top_pair_results"], 1)
        self.assertEqual(checkpoint["pending_bottom_pair_results"], 1)

    def test_arms_nonoverlap_and_equal_budget(self):
        r = _checkpoint([c(i) for i in range(1,5)], {}, "quality_rank_score", 10)
        self.assertEqual(r["predeclared_pairs_per_arm"], 2)
        self.assertEqual(r["tickets_per_arm"], 10)
        self.assertEqual(r["pending_top_pair_results"], 2)
        self.assertEqual(r["pending_bottom_pair_results"], 2)

    def test_reverse_robustness_rank_after_full_settlement(self):
        x = summarize(report([observed(i, i < 3) for i in range(1,5)]),
                      {"2026-11-23": [c(i) for i in range(1,5)]})
        self.assertEqual(x["original_rank_checkpoint_results"]["quality_top_bottom_1"]
                         ["unweighted_mean_day_level_pair_fraction_difference"], 1)
        self.assertEqual(x["original_rank_checkpoint_results"]["robustness_top_bottom_1"]
                         ["unweighted_mean_day_level_pair_fraction_difference"], -1)

    def test_reject_tampered_or_mismatched_candidate_rank(self):
        bad = observed(1, True)
        bad["quality_rank"] = 99
        with self.assertRaisesRegex(ValueError, "verified_row_not_at_original_rank"):
            summarize(report([bad]), {"2026-11-23": [c(i) for i in range(1,5)]})

    def test_reject_duplicate_frozen_candidates(self):
        with self.assertRaisesRegex(ValueError, "duplicate_original_frozen_pairs"):
            summarize(report([]), {"2026-11-23": [c(1)]*4})

    def test_no_completed_day_gives_no_illusory_gain(self):
        x = summarize(report([]), {"2026-11-23": [c(i) for i in range(1,5)]})
        for metric in x["original_rank_checkpoint_results"].values():
            self.assertEqual(metric["verified_day_units"], 0)
            self.assertIsNone(metric["unweighted_mean_day_level_pair_fraction_difference"])


if __name__ == "__main__":
    unittest.main()
