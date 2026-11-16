"""Unit tests for honest model comparison."""
import unittest
from model_comparison_audit import benchmark, paired_benchmark

def row():
    return {"fixture_id":"api:12","kickoff":"2026-11-20T19:00:00Z",
            "actual":[1,0],"forecasts":{"baseline":{
                "created_at":"2026-11-20T12:00:00Z",
                "scores":[[1,0,.4],[0,0,.3],[0,1,.2],[2,0,.1]]}}}
class BenchmarkTests(unittest.TestCase):
    def test_correct_exact_and_top5(self):
        x=benchmark([row()])["models"]["baseline"]
        self.assertEqual(x["exact_hit_rate"],1)
        self.assertEqual(x["top5_hit_rate"],1)
        self.assertAlmostEqual(x["mean_top5_coverage"],1)
    def test_late_prediction_rejected(self):
        r=row();r["forecasts"]["baseline"]["created_at"]=r["kickoff"]
        self.assertEqual(benchmark([r])["models"],{})
    def test_duplicate_fixture_not_double_counted(self):
        self.assertEqual(benchmark([row(),row()])["models"]["baseline"]["fixtures"],1)
    def test_invalid_mass_rejected(self):
        r=row();r["forecasts"]["baseline"]["scores"][0][2]=.9
        self.assertEqual(benchmark([r])["models"],{})
    def test_no_automatic_promotion(self):
        self.assertFalse(benchmark([row()])["models"]["baseline"]["eligible_for_promotion"])
    def test_missing_actual_not_counted(self):
        r=row();r["actual"]=None
        self.assertEqual(benchmark([r])["models"],{})
    def test_truncated_distribution_not_brier(self):
        r=row();r["forecasts"]["baseline"]["scores"]=[[1,0,.3],[0,0,.2]]
        self.assertIsNone(benchmark([r])["models"]["baseline"]["multiclass_brier"])
    def test_paired_cohort_excludes_duplicate_fixtures(self):
        r=row()
        r["forecasts"]["challenger"]=dict(r["forecasts"]["baseline"])
        self.assertEqual(paired_benchmark([r,r])["shared_fixtures"],0)
        self.assertEqual(paired_benchmark([r,r])["duplicate_fixtures_excluded"],1)
    def test_paired_cohort_same_match(self):
        r=row()
        r["forecasts"]["challenger"]=dict(r["forecasts"]["baseline"])
        result=paired_benchmark([r])
        self.assertEqual(result["shared_fixtures"],1)
        self.assertEqual(result["models"]["baseline"]["fixtures"],1)
        self.assertEqual(result["models"]["challenger"]["fixtures"],1)
if __name__=="__main__":
    unittest.main()
