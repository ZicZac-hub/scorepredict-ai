from datetime import datetime, timedelta, timezone
import hashlib
import json

import numpy as np

from forecast_ledger import _forecast, append_file_record
from v21_capture_sources import source_packet_path
import v45_source_impact as impact


KICKOFF = datetime(2025, 8, 10, 18, tzinfo=timezone.utc)
NOW = KICKOFF + timedelta(hours=3)


def model(minutes=90, horizon="rolling"):
    form = [.38, .32, .17, .08, .04, .01]
    capture = KICKOFF - timedelta(minutes=minutes)
    return _forecast({"mode": "prospective", "provider": "API-Football", "fixture_id": 123,
                      "model_key": "v8", "model_version": "v8-test", "horizon": horizon,
                      "input_digest": "saved-history", "score_basis": "regulation",
                      "home_team_id": 4, "away_team_id": 5, "home_team": "A", "away_team": "B",
                      "evidence_as_of_utc": capture.isoformat(), "captured_at_utc": capture.isoformat(),
                      "kickoff_utc": KICKOFF.isoformat(), "score_matrix": np.outer(form, form).tolist()})


def save_market(root, forecast, *, minutes):
    fields = ("provider", "fixture_id", "forecast_id", "input_digest", "home_team_id", "away_team_id", "kickoff_utc")
    capture = (KICKOFF - timedelta(minutes=minutes)).isoformat()
    source = {key: forecast[key] for key in fields}
    snapshot = {**source, "status": "captured_pre_kickoff", "captured_at_utc": capture,
                "multi_market": {
                    "1x2": {"last_source_update_utc": capture,
                            "probabilities": {"home": .6, "draw": .2, "away": .2}},
                    "over_2_5": {"last_source_update_utc": capture,
                                 "probabilities": {"yes": .35, "no": .65}}}}
    snapshot["snapshot_sha256"] = hashlib.sha256(json.dumps(snapshot, sort_keys=True,
                  separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    path = source_packet_path(root, forecast)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"source_forecast": source, "provider_snapshot": snapshot}))


def test_market_within_fixed_cutoff_is_scored_and_later_market_is_excluded(tmp_path):
    source = tmp_path / "source"
    provider = tmp_path / "providers"
    baseline = model()
    append_file_record(baseline, root=source)
    (source / "verified_results.json").write_text(json.dumps({"result": {
        **{k: baseline[k] for k in ("provider", "fixture_id", "home_team_id", "away_team_id", "kickoff_utc")},
        "evidence_version": 2, "score_basis": "regulation", "status": "FT",
        "home_goals": 1, "away_goals": 0, "verified_at_utc": NOW.isoformat()}}))
    later_forecast = model(minutes=10, horizon="later")
    append_file_record(later_forecast, root=source)
    save_market(provider, later_forecast, minutes=7)

    def evaluate():
        return impact.evaluate(source_root=source, prematch_root=tmp_path, provider_root=provider,
                               early_root=tmp_path, late_root=tmp_path, pair_root=tmp_path, now=NOW)

    before = evaluate()["comparisons"]["market"]["single_fixtures"]
    assert before["opportunities"] == 1
    assert before["status"] == "no_qualified_source_evidence"
    assert before["models"]["baseline"]["hit_rate"] is None

    # A valid packet digest can still be too late for the fixed 15 minute cutoff.
    save_market(provider, baseline, minutes=10)
    assert evaluate()["comparisons"]["market"]["single_fixtures"]["matched"] == 0

    save_market(provider, baseline, minutes=30)
    after = evaluate()["comparisons"]["market"]["single_fixtures"]
    assert after["qualified_inputs"] == after["matched"] == 1
    assert after["status"] == "insufficient_matched_results"
    assert after["models"]["baseline"]["exact_log_loss"] != after["models"]["with_source"]["exact_log_loss"]


def test_doubles_require_both_sources_and_both_verified_finals():
    source = {"xg": {"value": {"ready": True, "expected_goal_indicators": [1.8, .9]}, "reference": {}}}
    one = model()
    two = {**one, "fixture_id": "456"}
    comparisons = impact._empty_comparisons()
    impact._record([one, two], [source, {}], [(1, 0), (0, 0)], comparisons)
    impact._record([one, two], [source, source], [(1, 0), None], comparisons)
    impact._record([one, two], [source, source], [(1, 0), (0, 0)], comparisons)
    summary = impact._comparison(*comparisons["historical_xg"], pairs=True)
    assert summary["opportunities"] == 3
    assert summary["missing_source_evidence"] == summary["missing_verified_final"] == summary["matched"] == 1
    assert summary["models"]["baseline"]["score_log_loss"] != summary["models"]["with_source"]["score_log_loss"]
    assert impact._comparison(*comparisons["market"], pairs=True)["matched"] == 0
