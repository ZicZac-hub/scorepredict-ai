"""Regression tests for duplicate fixture audit integrity."""
import unittest
from model_comparison_audit import benchmark, paired_benchmark

class DuplicateFixtureTests(unittest.TestCase):
    def test_both_duplicate_rows_excluded(self):
        row={"fixture_id":"api:1","kickoff":"2026-11-20T20:00:00Z","actual":[1,0],"forecasts":{"baseline":{"created_at":"2026-11-20T12:00:00Z","scores":[[1,0,0.4]]}}}
        result=benchmark([row,dict(row,actual=[0,1])])
        self.assertEqual(result["models"],{})
        self.assertEqual(result["rejected"]["duplicate_fixture"],2)

    def test_unique_fixture_retained(self):
        row={"fixture_id":"api:2","kickoff":"2026-11-20T20:00:00Z","actual":[1,0],"forecasts":{"baseline":{"created_at":"2026-11-20T12:00:00Z","scores":[[1,0,0.4]]}}}
        self.assertEqual(benchmark([row])["models"]["baseline"]["fixtures"],1)

if __name__ == "__main__":
    unittest.main()
