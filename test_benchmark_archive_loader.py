"""Regression tests for the automatic multi-archive verified benchmark."""
import json
import tempfile
import unittest
from pathlib import Path

from benchmark_archive_loader import load_archive, write_summary
from model_comparison_audit import benchmark, paired_benchmark


class ArchiveAdapterTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def write(self, path, value):
        destination = self.root / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(value), encoding="utf-8")
        return destination

    def result(self, fid=123, score=(1, 0)):
        return {
            "fixture_id": fid, "provider": "API-Football",
            "home_team_id": 11, "away_team_id": 22, "status": "FT",
            "score_basis": "regulation", "evidence_version": 2,
            "home_goals": score[0], "away_goals": score[1],
            "kickoff_utc": "2026-11-15T12:00:00Z",
            "verified_at_utc": "2026-11-15T16:00:00Z",
        }

    def forecast(self, model="v8", fid=123, matrix=None):
        return {
            "fixture_id": str(fid), "provider": "API-Football",
            "home_team_id": "11", "away_team_id": "22",
            "mode": "prospective", "score_basis": "regulation",
            "model_version": model,
            "captured_at_utc": "2026-11-14T12:00:00Z",
            "evidence_as_of_utc": "2026-11-14T11:00:00Z",
            "kickoff_utc": "2026-11-15T12:00:00Z",
            "score_matrix": matrix or [[.2, .1], [.6, .1]],
        }

    def archive(self, folder, filename, row, fid=123):
        return self.write(
            "data/" + folder + "/forecasts/API-Football/" + str(fid) + "/" + filename + ".json",
            row)

    def test_reads_both_frozen_ledgers_and_duplicate_verified_result_sources(self):
        result = self.result()
        self.write("data/v10_prospective/verified_results.json", {"123": result})
        self.write("data/prematch/verified_results.json", {"123": result})
        original = self.archive("v10_prospective", "original", self.forecast("v8"))
        self.archive("prospective_ledger", "other", self.forecast("v7"))
        before = original.read_bytes()
        rows, diagnostics = load_archive(self.root)
        self.assertEqual(diagnostics["verified_fixtures"], 1)
        self.assertEqual(diagnostics["compatible_records"], 1)
        self.assertEqual(diagnostics["frozen_forecasts"], 2)
        self.assertEqual(diagnostics["awaiting_verified_results"], 0)
        self.assertEqual(set(rows[0]["forecasts"]), {"v8", "v7"})
        self.assertEqual(paired_benchmark(rows)["shared_fixtures"], 1)
        self.assertEqual(benchmark(rows)["models"]["v8"]["exact_hit_rate"], 1.0)
        self.assertEqual(original.read_bytes(), before)

    def test_conflicting_verified_final_scores_excluded(self):
        self.write("data/v10_prospective/verified_results.json", {"123": self.result()})
        self.write("data/prematch/verified_results.json", {"123": self.result(score=(2, 1))})
        self.archive("v10_prospective", "a", self.forecast())
        rows, diagnostics = load_archive(self.root)
        self.assertEqual(rows, [])
        self.assertEqual(diagnostics["verified_fixtures"], 0)
        self.assertEqual(diagnostics["conflicting_result_fixtures"], 1)
        self.assertEqual(diagnostics["awaiting_verified_results"], 1)

    def test_point_only_baselines_excluded_from_probability_metrics(self):
        self.write("data/v10_prospective/verified_results.json", {
            "123": self.result(score=(1, 1)),
        })
        self.archive("v10_prospective", "v8", self.forecast())
        for name in ("fixed-always-1-1-v1", "marked-point-model"):
            row = self.forecast(model=name, matrix=[[0, 0], [0, 1]])
            if name == "marked-point-model":
                row["probability_metrics_eligible"] = False
            self.archive("prospective_ledger", name, row)

        report = write_summary(self.root)
        for cohort in (report["models"], report["paired"]["models"]):
            for name in ("fixed-always-1-1-v1", "marked-point-model"):
                stats = cohort[name]
                self.assertEqual(stats["exact_hit_rate"], 1)
                self.assertEqual(stats["probability_forecast_count"], 0)
                self.assertEqual(stats["full_distribution_count"], 0)
                self.assertIsNone(stats["top5_hit_rate"])
                self.assertIsNone(stats["mean_top5_coverage"])
                self.assertIsNone(stats["multiclass_brier"])
            self.assertEqual(cohort["v8"]["probability_forecast_count"], 1)
            self.assertEqual(cohort["v8"]["full_distribution_count"], 1)

    def test_requires_original_evidence_and_strict_identity(self):
        self.write("data/v10_prospective/verified_results.json", {"123": self.result()})
        original = self.forecast()
        self.archive("v10_prospective", "ok", original)
        altered = dict(original, home_team_id="99")
        self.archive("v10_prospective", "wrong-team", altered)
        late = dict(original, captured_at_utc="2026-11-16T12:00:00Z")
        self.archive("v10_prospective", "late", late)
        stale = dict(original, evidence_as_of_utc="2026-11-15T13:00:00Z")
        self.archive("v10_prospective", "future-evidence", stale)
        invalid = dict(original, score_matrix=[[.1, -.1], [.9, .1]])
        self.archive("v10_prospective", "invalid-grid", invalid)
        excluded = dict(original, eligible_for_comparison=False)
        self.archive("prospective_ledger", "fallback", excluded)
        rows, diagnostics = load_archive(self.root)
        self.assertEqual(len(rows), 1)
        self.assertEqual(diagnostics["skipped_records"], 5)
        self.assertEqual(diagnostics["frozen_forecasts"], 1)

    def test_unfinished_forecast_does_not_imply_a_result(self):
        self.archive("v10_prospective", "a", self.forecast())
        rows, diagnostics = load_archive(self.root)
        self.assertEqual(rows, [])
        self.assertEqual(diagnostics["archived_fixture_ids"], 1)
        self.assertEqual(diagnostics["awaiting_verified_results"], 1)
        self.assertEqual(diagnostics["verified_fixtures"], 0)

    def test_settlement_ledger_is_valid_fallback_and_report_is_deterministic(self):
        self.write("data/prospective_ledger/settlements/original.json", self.result())
        original = self.archive("prospective_ledger", "f", self.forecast())
        content = original.read_bytes()
        report1 = write_summary(self.root)
        path = self.root / "data/benchmark/summary.json"
        saved = path.read_bytes()
        report2 = write_summary(self.root)
        self.assertEqual(report1, report2)
        self.assertEqual(saved, path.read_bytes())
        self.assertEqual(content, original.read_bytes())
        self.assertEqual(report1["diagnostics"]["verified_fixtures"], 1)

    def test_tampered_verification_time_is_rejected(self):
        invalid = self.result()
        invalid["verified_at_utc"] = "2026-11-14T12:00:00Z"
        self.write("data/v10_prospective/verified_results.json", {"123": invalid})
        self.archive("v10_prospective", "f", self.forecast())
        rows, diagnostics = load_archive(self.root)
        self.assertEqual(rows, [])
        self.assertEqual(diagnostics["rejected_results"], 1)


if __name__ == "__main__":
    unittest.main()
