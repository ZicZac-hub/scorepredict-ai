"""Regression checks for five-double ranking and honest coverage accounting.

Run: python -m unittest test_smart_double_regression.py
"""
import unittest
from smart_double_ai import five_doubles, rank_pairs

def fixture(fid, home, away, scores, kickoff="2026-12-01T10:00:00Z"):
    return {"fixture":{"provider":"API-Football","id":fid,
            "homeTeam":{"name":home},"awayTeam":{"name":away},
            "utcDate":kickoff},"kickoff":kickoff,
            "result":{"scores":scores}}

class FiveDoubleRegression(unittest.TestCase):
    def setUp(self):
        self.a=fixture(1,"Alpha","Beta",[(1,0,.21),(1,1,.19),(2,0,.12)])
        self.b=fixture(2,"Gamma","Delta",[(1,0,.20),(0,0,.16),(1,1,.14)])

    def test_five_distinct_and_ordered(self):
        picks=five_doubles(self.a,self.b)
        self.assertEqual(len(picks),5)
        self.assertEqual(len({(p["Score 1"],p["Score 2"]) for p in picks}),5)
        self.assertEqual([p["Joint"] for p in picks],
                         sorted((p["Joint"] for p in picks),reverse=True))

    def test_coverage_is_not_claimed_win_rate(self):
        picks=five_doubles(self.a,self.b)
        self.assertAlmostEqual(sum(p["Joint"] for p in picks),
                               sum(sorted((x[2]*y[2] for x in self.a["result"]["scores"]
                                           for y in self.b["result"]["scores"]),reverse=True)[:5]))
        self.assertLess(sum(p["Joint"] for p in picks),1)

    def test_same_fixture_and_duplicate_provider_blocked(self):
        self.assertEqual(five_doubles(self.a,self.a),[])
        duplicate=fixture(999,"Alpha","Beta",self.a["result"]["scores"])
        self.assertEqual(five_doubles(self.a,duplicate),[])

    def test_invalid_probabilities_do_not_create_predictions(self):
        invalid=fixture(3,"E","F",[(1,0,float("nan")),(2,0,-.1),(0,0,2)])
        self.assertEqual(five_doubles(self.a,invalid),[])

    def test_pair_rejects_same_team(self):
        other=fixture(4,"Alpha","Zeta",self.b["result"]["scores"])
        self.assertEqual(rank_pairs([self.a,other]),[])

    def test_future_only_when_now_supplied(self):
        self.assertEqual(rank_pairs([self.a,self.b],now="2026-12-01T09:00:00Z")[0]["coverage"],
                         sum(p["Joint"] for p in five_doubles(self.a,self.b)))
        self.assertEqual(rank_pairs([self.a,self.b],now="2026-12-01T10:00:00Z"),[])

if __name__=="__main__":
    unittest.main()
