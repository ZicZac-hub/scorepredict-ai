"""Regression tests for original ZicZac prospective archive conversion."""
import json
import tempfile
import unittest
from pathlib import Path
from benchmark_archive_loader import load_archive
from model_comparison_audit import benchmark

class ArchiveAdapterTests(unittest.TestCase):
    def test_verified_frozen_matrix_is_converted_without_mutation(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            results=root/"data/v10_prospective/verified_results.json"
            results.parent.mkdir(parents=True)
            results.write_text(json.dumps({"123":{"fixture_id":123,"provider":"API-Football",
                "status":"FT","score_basis":"regulation","home_goals":1,
                "away_goals":0,"kickoff_utc":"2026-11-15T12:00:00Z"}}))
            folder=root/"data/v10_prospective/forecasts/API-Football/123"
            folder.mkdir(parents=True)
            frozen={"fixture_id":"123","provider":"API-Football",
                "mode":"prospective","model_version":"v8",
                "captured_at_utc":"2026-11-14T12:00:00Z",
                "kickoff_utc":"2026-11-15T12:00:00Z",
                "score_matrix":[[.2,.1],[.6,.1]]}
            path=folder/"abc.json"
            original=json.dumps(frozen)
            path.write_text(original)
            rows,diagnostics=load_archive(temp)
            self.assertEqual(diagnostics["compatible_records"],1)
            self.assertEqual(diagnostics["frozen_forecasts"],1)
            self.assertEqual(path.read_text(),original)
            report=benchmark(rows)
            self.assertEqual(report["models"]["v8"]["exact_hit_rate"],1)
            frozen["captured_at_utc"]="2026-11-16T12:00:00Z"
            (folder/"late.json").write_text(json.dumps(frozen))
            rows,diagnostics=load_archive(temp)
            self.assertEqual(diagnostics["compatible_records"],1)
            self.assertEqual(diagnostics["skipped_records"],1)
if __name__=="__main__":
    unittest.main()
