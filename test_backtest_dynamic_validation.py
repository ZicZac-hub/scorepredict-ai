"""Regression checks for the matched dynamic-model validation report."""
import unittest

from backtest_dynamic_dixon_coles import evaluate, paired_comparison, wilson


class DynamicValidationTests(unittest.TestCase):
    def test_exact_paired_test(self):
        self.assertEqual(paired_comparison(0, 0), 1.0)
        self.assertAlmostEqual(paired_comparison(5, 0), 0.0625)
        self.assertAlmostEqual(paired_comparison(0, 5), 0.0625)
        self.assertEqual(paired_comparison(5, 5), 1.0)

    def test_wilson_intervals(self):
        self.assertIsNone(wilson(0, 0))
        low, high = wilson(0, 10)
        self.assertAlmostEqual(low, 0)
        self.assertAlmostEqual(high, 0.2775328)
        low, high = wilson(10, 10)
        self.assertAlmostEqual(low, 0.7224672)
        self.assertAlmostEqual(high, 1)

    def test_empty_data_never_reports_verified_accuracy(self):
        report = evaluate([])
        self.assertEqual(report['matched_pairs'], 0)
        self.assertEqual(report['excluded_pairs'], 0)
        for model in report['models'].values():
            self.assertIsNone(model['hit_rate'])
            self.assertIsNone(model['hit_rate_wilson_95'])
        for comparison in report['paired_comparisons'].values():
            self.assertIsNone(comparison['exact_p_value'])
            self.assertIsNone(comparison['hit_rate_difference'])


if __name__ == '__main__':
    unittest.main()
