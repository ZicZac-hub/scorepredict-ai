import json

from v43_catalog import build_catalog


def _write(root, name, payload):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _final(fixture_id, *, home_goals=1, verified_at="2026-11-15T01:00:00Z"):
    return {
        "provider": "API-Football",
        "fixture_id": fixture_id,
        "home_team_id": 1,
        "away_team_id": 2,
        "kickoff_utc": "2026-11-14T21:30:00Z",
        "verified_at_utc": verified_at,
        "score_basis": "regulation",
        "evidence_version": 2,
        "status": "FT",
        "home_goals": home_goals,
        "away_goals": 0,
    }


def test_catalog_merges_verified_sources_and_quarantines_disagreements(tmp_path):
    v10 = "data/v10_prospective/verified_results.json"
    prematch = "data/prematch/verified_results.json"
    _write(tmp_path, v10, {"101": _final(101), "102": _final(102)})
    _write(tmp_path, prematch, {
        "101": _final(101, verified_at="2026-11-15T02:00:00Z"),
        "102": _final(102, home_goals=2),
        "103": _final(103, verified_at="2026-11-17T02:00:00Z"),
    })
    _write(tmp_path, "data/prediction_evidence_index.json", {"paths": [v10, prematch, prematch]})

    catalog = build_catalog(tmp_path, as_of="2026-11-16T00:00:00Z")
    summary = catalog["summary"]
    assert summary["verified_fixtures"] == 1
    assert summary["verified_fixtures_by_provider"] == {"API-Football": 1}
    assert summary["verified_result_conflicts"] == 1
    assert summary["verified_result_rejections"] == {"invalid_or_unavailable_verified_final": 1}
    assert summary["verified_fixture_start_utc"] == "2026-11-14T21:30:00+00:00"
    assert summary["indexed_files"] == summary["archive_files"] == 2


def test_catalog_keeps_unavailable_versions_and_unmatched_statistics_explicit(tmp_path):
    _write(tmp_path, "data/accuracy_diagnosis/historical_replay.json", {
        "matched_fixtures": 4,
        "single_match_metrics": {
            "same_fixture_cohort": {"evaluated": 4, "exact_hits": 1},
            "partial_fixture_cohort": {"evaluated": 2, "exact_hits": 1},
        },
    })
    (tmp_path / "engine.py").touch()

    catalog = build_catalog(tmp_path)
    assert [row["version"] for row in catalog["versions"]] == list(range(1, 43))
    assert catalog["summary"]["unavailable_version_labels"] == [1, 2, 3, 4, 5]
    assert catalog["versions"][5]["status"] == "Documented"
    assert catalog["versions"][-1]["status"] == "Artifact unavailable"
    assert len(catalog["benchmarks"]) == 1
    assert [row["method"] for row in catalog["metrics"]] == ["same_fixture_cohort"]
