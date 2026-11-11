import unittest
from double_score_selection import select_diversified,score_coverage

class TestDoubleScoreSelection(unittest.TestCase):
    def test_diversifies_both_legs(self):
        options=[{"Score 1":a,"Score 2":b,"Joint":pa*pb}
                 for a,pa in [("1-1",.35),("2-1",.25),("1-0",.20)]
                 for b,pb in [("1-1",.40),("1-0",.30),("0-1",.15)]]
        picks=select_diversified(options)
        self.assertEqual(len(picks),5)
        self.assertEqual((picks[0]["Score 1"],picks[0]["Score 2"]),("1-1","1-1"))
        self.assertGreaterEqual(score_coverage(picks)["match_1_scores"],3)
        self.assertGreaterEqual(score_coverage(picks)["match_2_scores"],3)
        self.assertEqual(len({(p["Score 1"],p["Score 2"]) for p in picks}),5)

    def test_fewer_than_five(self):
        options=[{"Score 1":"1-0","Score 2":"0-0","Joint":.1}]
        self.assertEqual(select_diversified(options),options)
        self.assertEqual(select_diversified([],5),[])

if __name__=="__main__":unittest.main()
