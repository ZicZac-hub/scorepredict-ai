"""Regression checks for immutable prospective five-double audit."""
import unittest
from five_double_audit import evaluate

def sample():
    return {"id":"pair-a","predicted_at":"2026-11-01T08:00:00Z",
            "kickoff_1":"2026-11-01T10:00:00Z","kickoff_2":"2026-11-01T12:00:00Z",
            "actual_1":"1-0","actual_2":"0-0",
            "doubles":[{"Score 1":a,"Score 2":b,"Joint":p}
                       for a,b,p in [("1-0","0-0",.05),("0-0","0-0",.04),
                                     ("2-0","0-0",.03),("1-1","0-0",.02),
                                     ("1-0","1-1",.01)]]}
class AuditTests(unittest.TestCase):
    def test_verified_hit(self):
        x=evaluate([sample()])
        self.assertEqual((x["settled"],x["hits"]), (1,1))
        self.assertAlmostEqual(x["mean_model_coverage"],.15)
    def test_late_prediction_excluded(self):
        r=sample();r["predicted_at"]=r["kickoff_1"]
        self.assertEqual(evaluate([r])["settled"],0)
    def test_unsettled_excluded(self):
        r=sample();r["actual_2"]=None
        self.assertEqual(evaluate([r])["excluded"]["unsettled"],1)
    def test_duplicate_excluded(self):
        self.assertEqual(evaluate([sample(),sample()])["settled"],1)
    def test_duplicate_score_pair_excluded(self):
        r=sample();r["doubles"][1]=dict(r["doubles"][0])
        self.assertEqual(evaluate([r])["settled"],0)
    def test_missing_time_excluded(self):
        r=sample();r["predicted_at"]="2026-11-01"
        self.assertEqual(evaluate([r])["settled"],0)
    def test_miss_not_fabricated_hit(self):
        r=sample();r["actual_1"]="4-4"
        self.assertEqual(evaluate([r])["observed_hit_rate"],0)
if __name__=="__main__":
    unittest.main()
