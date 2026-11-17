"""Regression tests for the read-only Advanced Quality Report."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from advanced_quality_ui import _local_time, render, summary


class AdvancedQualityReportTests(unittest.TestCase):
    def test_summary_uses_recorded_values_without_inventing_coverage(self):
        report = {
            "checked_at_utc": "2026-11-17T03:53:07+00:00",
            "source_completeness": {
                "packets_checked": 20,
                "with_historical_xg": 0,
                "with_6_per_team": 0,
                "with_both_starting_XIs": 0,
                "with_2_market_types": 20,
            },
            "frozen_v22_vs_v8": {"v22_research": {"verified": 0, "hits": 0}},
        }
        market = {"coverage_health": {
            "eligible_frozen_forecasts": 208, "matched_market_forecasts": 6
        }}
        result = summary(report, market)
        self.assertEqual(result["source_packets"], 20)
        self.assertEqual(result["historical_xg"], 0)
        self.assertEqual(result["two_market_types"], 20)
        self.assertEqual(result["market_matched"], 6)
        self.assertAlmostEqual(result["market_percent"], 6 / 208 * 100)
        self.assertIsNone(result["source_capture_enabled"])
        self.assertEqual(result["v22_verified_pairs"], 0)

    def test_missing_data_is_not_called_disabled_or_verified(self):
        result = summary({"source_completeness": {}}, {})
        self.assertEqual(result["source_packets"], 0)
        self.assertIsNone(result["market_percent"])
        self.assertEqual(result["v22_verified_pairs"], 0)

    def test_timestamp_is_sydney_not_utc(self):
        self.assertEqual(_local_time("2026-11-17T03:53:07+00:00"),
                         "17 Nov 2026, 02:53 PM Sydney")
        self.assertEqual(_local_time(None), "Not recorded")

    def test_missing_report_displays_warning_instead_of_crashing(self):
        with tempfile.TemporaryDirectory() as folder:
            st = MagicMock()
            render(st, Path(folder))
            st.warning.assert_called_once()
            st.link_button.assert_called_once()
            st.metric.assert_not_called()

    def test_saved_report_displays_observed_coverage_and_unknown_collector(self):
        with tempfile.TemporaryDirectory() as folder:
            data = Path(folder)
            (data / "v22_prospective").mkdir()
            (data / "v24_market_catalog").mkdir()
            (data / "v22_prospective" / "quality_report.json").write_text(
                json.dumps({
                    "checked_at_utc": "2026-11-17T03:53:07+00:00",
                    "source_completeness": {
                        "packets_checked": 20,
                        "with_historical_xg": 0,
                        "with_6_per_team": 0,
                        "with_both_starting_XIs": 0,
                        "with_2_market_types": 20,
                    },
                    "frozen_v22_vs_v8": {"v22_research": {"verified": 0, "hits": 0}},
                }), encoding="utf-8")
            (data / "v24_market_catalog" / "status.json").write_text(
                json.dumps({"coverage_health": {
                    "eligible_frozen_forecasts": 208,
                    "matched_market_forecasts": 6}}), encoding="utf-8")
            st = MagicMock()
            st.columns.return_value = [MagicMock() for _ in range(4)]
            render(st, data)
            st.dataframe.assert_called_once()
            st.warning.assert_called_once()
            self.assertGreaterEqual(st.info.call_count,1)
            st.link_button.assert_any_call("Open saved quality report",
                                           "https://github.com/ZicZac-hub/scorepredict-ai/blob/main/data/v22_prospective/quality_report.json")


if __name__ == "__main__":
    unittest.main()
