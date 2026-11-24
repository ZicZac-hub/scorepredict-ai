"""Prospective, source-authenticated ablations of the same immutable pair."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
import math
from pathlib import Path

import numpy as np
import pytest

from forecast_ledger import append_file_record, read_file_records
import professional_double_shadow as baseline_policy
import professional_paired_study as study
from v21_capture_sources import source_packet_path

NOW = datetime(2035, 1, 1, tzinfo=timezone.utc)


def sealed(value, field="snapshot_sha256"):
    output = {k: v for k, v in deepcopy(value).items() if k != field}
    output[field] = study._hash(output)
    return output


def write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value))


def setup(tmp_path, *, available=True, observed=None):
    original, pair_root, provider_root, root = [tmp_path/name for name in ("original", "pairs", "provider", "study")]
    marginal = np.array([.30, .35, .20, .10, .04, .01])
    grid = np.outer(marginal, marginal)
    scores = sorted([[h, a, float(p)] for h, row in enumerate(grid) for a, p in enumerate(row)],
                    key=lambda r: (-r[2], r[0], r[1]))[:10]
    for i in (1, 2):
        model = {"provider": "API-Football", "fixture_id": str(i), "model_key": "v8",
                 "model_version": "original-v8", "input_digest": "original-digest-"+str(i),
                 "home_team_id": str(2*i), "away_team_id": str(2*i+1),
                 "home_team": "Home "+str(i), "away_team": "Away "+str(i),
                 "fixture_name": "Home "+str(i)+" vs Away "+str(i), "competition": "PL",
                 "captured_at_utc": (NOW-timedelta(hours=2)).isoformat(),
                 "evidence_as_of_utc": (NOW-timedelta(hours=2, minutes=5)).isoformat(),
                 "kickoff_utc": (NOW+timedelta(minutes=45+20*(i-1))).isoformat(),
                 "score_matrix": grid.tolist(), "top10_scores": scores}
        assert append_file_record(model, root=original)
    assert baseline_policy.freeze(root=pair_root, source_root=original,
                                  now=NOW-timedelta(hours=1), clock=lambda: NOW-timedelta(hours=1))["new_sets"]
    models = read_file_records(root=original)
    if available:
        for model in models:
            packet = source(model, observed=observed)
            write(source_packet_path(provider_root, model), packet)
    return {"root": root, "pair_root": pair_root, "source_root": original,
            "provider_root": provider_root}, models


def source(model, *, observed=None, markets=True, xg=True, lineup=True):
    captured = observed or NOW-timedelta(minutes=10)
    packet = {"status": "captured_pre_kickoff", "provider": model["provider"],
              "fixture_id": model["fixture_id"], "home_team_id": model["home_team_id"],
              "away_team_id": model["away_team_id"], "kickoff_utc": model["kickoff_utc"],
              "captured_at_utc": captured.isoformat(), "verified_historical_fixture_xg": [],
              "multi_market": {}, "confirmed_starting_xi": []}
    if markets:
        for kind, probabilities in (("1x2", {"home": .61, "draw": .24, "away": .15}),
                                    ("over_2_5", {"yes": .62, "no": .38})):
            packet["multi_market"][kind] = {"last_source_update_utc": (captured-timedelta(minutes=5)).isoformat(),
                                           "probabilities": probabilities}
    if xg:
        for team, venue in ((model["home_team_id"], "home"), (model["away_team_id"], "away")):
            for i in range(6):
                home, away = (team, "other-"+str(i)) if venue == "home" else ("other-"+str(i), team)
                packet["verified_historical_fixture_xg"].append({"fixture_id": "past-"+team+"-"+str(i),
                    "home_team_id": home, "away_team_id": away,
                    "kickoff_utc": (NOW-timedelta(days=i+1)).isoformat(), "home_xg": 2.1, "away_xg": 1.1})
    if lineup:
        for team in (model["home_team_id"], model["away_team_id"]):
            packet["confirmed_starting_xi"].append({"team_id": team, "formation": "4-3-3",
                                                   "starting_xi": ["Player "+team+"-"+str(i) for i in range(11)]})
    return {"source_forecast": {k: model[k] for k in ("provider", "fixture_id", "forecast_id", "input_digest",
                "home_team_id", "away_team_id", "kickoff_utc")}, "provider_snapshot": sealed(packet)}


def frozen(tmp_path, **kwargs):
    paths, models = setup(tmp_path, **kwargs)
    status = study.freeze(**paths, now=NOW, clock=lambda: NOW)
    assert status["new_trials"] == ["2035-01-01"]
    path = paths["root"]/"trials"/"2035-01-01.json"
    return paths, models, path, json.loads(path.read_text())


def settled(paths, models, *, goals=(1, 0), verified_at=None):
    finals = {}
    for model in models:
        row = {k: model[k] for k in ("provider", "fixture_id", "home_team_id", "away_team_id", "kickoff_utc")}
        row.update({"evidence_version": 2, "status": "FT", "score_basis": "regulation",
                    "home_goals": goals[0], "away_goals": goals[1],
                    "verified_at_utc": (verified_at or NOW+timedelta(hours=4)).isoformat()})
        finals[model["fixture_id"]] = row
    write(paths["source_root"]/"verified_results.json", finals)
    return finals


def evaluate(paths, *, now=None):
    return study.evaluate(**paths, prematch_root=paths["root"]/"missing", now=now or NOW+timedelta(days=1))


def test_same_baseline_pair_and_full_grid_arms_frozen_once_without_api_calls(tmp_path, monkeypatch):
    import requests
    monkeypatch.setattr(requests, "get", lambda *a, **k: pytest.fail("No network requests permitted"))
    paths, models, path, record = frozen(tmp_path)
    original_files = {p: p.read_bytes() for p in paths["source_root"].rglob("*.json")}
    saved = path.read_bytes()
    assert record["arms"]["baseline_v8"]["full_score_grids"] == [m["score_matrix"] for m in models]
    assert all(arm["status"] == "issued" for arm in record["arms"].values())
    assert len(record["arms"]["xg_market"]["five_doubles"]) == 5
    assert record["lineup_context"][0]["both_confirmed_xis"]
    assert record["lineup_context"][0]["numerical_lineup_weight"] == 0
    later = study.freeze(**paths, now=NOW+timedelta(minutes=1), clock=lambda: NOW+timedelta(minutes=1))
    assert later["existing_trials_preserved"] == ["2035-01-01"]
    assert path.read_bytes() == saved
    assert all(p.read_bytes() == data for p, data in original_files.items())
    report = evaluate(paths)
    assert report["opportunities"] == report["pending_opportunities"] == 1
    assert all(row["verified"] == 0 and row["pending"] == 1 and row["hit_rate"] is None
               for row in report["methods"].values())


def test_missing_sources_preserve_control_and_reveal_all_arm_abstentions(tmp_path):
    paths, models, _, record = frozen(tmp_path, available=False)
    assert record["arms"]["baseline_v8"]["status"] == "issued"
    assert all(record["arms"][m]["status"] == "abstained" for m in study.METHODS[1:])
    settled(paths, models)
    report = evaluate(paths)
    assert report["verified_opportunities"] == 1
    assert report["methods"]["baseline_v8"]["verified"] == 1
    for method in study.METHODS[1:]:
        assert report["methods"][method]["opportunities"] == report["methods"][method]["abstained"] == 1
        assert report["methods"][method]["hit_rate"] is None
        assert report["paired_comparisons"][method]["verified_common_pairs"] == 0


def test_verified_metrics_use_same_actual_scores_and_explicit_common_pair_sample(tmp_path):
    paths, models, _, record = frozen(tmp_path)
    settled(paths, models)
    report = evaluate(paths)
    assert report["verified_opportunities"] == 1
    for method in study.METHODS:
        metrics = report["methods"][method]
        grids = record["arms"][method]["full_score_grids"]
        assert metrics["verified"] == 1 and metrics["single_fixtures_evaluated"] == 2
        expected = sum(-math.log(g[1][0]) for g in grids)/2
        assert metrics["single_score_log_loss"] == pytest.approx(expected)
        assert 0 <= metrics["outcome_brier"] <= 2
    for method in study.METHODS[1:]:
        paired = report["paired_comparisons"][method]
        assert paired["verified_common_pairs"] == 1
        assert paired["proper_same_event_score_log_loss_advantage"]["mean"] == pytest.approx(
            report["methods"]["baseline_v8"]["single_score_log_loss"]-report["methods"][method]["single_score_log_loss"])
        assert paired["proper_same_event_score_log_loss_advantage"]["normal_95_interval"] is None
    assert report["target_70_percent_proven"] is False and report["realized_roi"] is None


@pytest.mark.parametrize("changes", [{"verified_at_utc": (NOW+timedelta(days=2)).isoformat()},
    {"home_team_id": "wrong-team"}, {"score_basis": "extra_time"}, {"home_goals": True}])
def test_wrong_identity_or_future_result_does_not_become_loss(tmp_path, changes):
    paths, models, _, _ = frozen(tmp_path)
    finals = settled(paths, models)
    next(iter(finals.values())).update(changes)
    write(paths["source_root"]/"verified_results.json", finals)
    report = evaluate(paths)
    assert report["pending_opportunities"] == 1 and report["verified_opportunities"] == 0
    assert all(row["hits"] == row["verified"] == 0 for row in report["methods"].values())


def test_stale_future_wrong_digest_sources_cannot_become_features(tmp_path):
    paths, models = setup(tmp_path, available=False)
    for model in models:
        for i, value in enumerate((source(model, observed=NOW-timedelta(hours=4)),
                                   source(model, observed=NOW+timedelta(minutes=1)), source(model))):
            if i == 2:
                value["source_forecast"]["input_digest"] = "wrong"
            base = source_packet_path(paths["provider_root"], model)
            write(base if i == 0 else base.parent/"versions"/(base.stem+"_"+str(i)+".json"), value)
    assert study.freeze(**paths, now=NOW, clock=lambda: NOW)["new_trials"]
    record = json.loads(next((paths["root"]/"trials").glob("*.json")).read_text())
    assert all(record["arms"][method]["status"] == "abstained" for method in study.METHODS[1:])


def test_separate_fresh_odds_packet_cannot_hide_real_xg_or_backdate_future_features(tmp_path):
    paths, models = setup(tmp_path)
    for model in models:
        base = source_packet_path(paths["provider_root"], model)
        write(base.parent/"versions"/(base.stem+"_fresh_odds.json"),
              source(model, observed=NOW-timedelta(minutes=1), xg=False, lineup=False))
    study.freeze(**paths, now=NOW, clock=lambda: NOW)
    path = next((paths["root"]/"trials").glob("*.json"))
    original = path.read_bytes()
    row = json.loads(original)
    assert all(arm["status"] == "issued" for arm in row["arms"].values())
    assert row["component_sources"][0]["market"]["snapshot_sha256"] != row["component_sources"][0]["xg"]["snapshot_sha256"]
    for model in models:
        base = source_packet_path(paths["provider_root"], model)
        write(base.parent/"versions"/(base.stem+"_future.json"), source(model, observed=NOW+timedelta(minutes=1)))
    assert evaluate(paths)["valid_opportunities"] == 1
    assert path.read_bytes() == original


@pytest.mark.parametrize("field", ["arms", "component_sources", "baseline_pair"])
def test_resealed_mutation_does_not_replace_original_sources(tmp_path, field):
    paths, _, path, record = frozen(tmp_path)
    if field == "arms":
        record[field]["market_only"]["full_score_grids"][0][0][0] += .1
    elif field == "component_sources":
        record[field][0]["market"]["captured_at_utc"] = (NOW-timedelta(minutes=1)).isoformat()
    else:
        record[field]["record_sha256"] = "not-original"
    write(path, sealed(record, "record_sha256"))
    report = evaluate(paths)
    assert report["opportunities"] == report["invalid_opportunities"] == 1
    assert report["valid_opportunities"] == report["methods"]["baseline_v8"]["verified"] == 0
    assert report["source_rejections"]


def test_missing_protocol_is_rejected_without_recreation_and_implementation_changes_fail_closed(tmp_path, monkeypatch):
    paths, _, _, _ = frozen(tmp_path)
    protocol = paths["root"]/"protocol.json"
    data = protocol.read_bytes()
    protocol.unlink()
    assert evaluate(paths)["valid_opportunities"] == 0
    assert not protocol.exists()
    protocol.write_bytes(data)
    monkeypatch.setattr(study, "_implementation", lambda: {"changed": "new-hash"})
    assert evaluate(paths)["valid_opportunities"] == 0
    with pytest.raises(ValueError, match="fixed_protocol"):
        study.freeze(**paths, now=NOW)
    assert protocol.read_bytes() == data


def test_decision_window_and_slow_calculation_never_backdate(tmp_path, monkeypatch):
    paths, _ = setup(tmp_path)
    before = NOW-timedelta(hours=1)
    assert study.freeze(**paths, now=before, clock=lambda: before)["new_trials"] == []
    # Reuse protocol registered in the earlier pass; no late write is allowed.
    current = [NOW]
    original = study._arms
    def slow(*args):
        result = original(*args)
        current[0] = NOW+timedelta(minutes=42)
        return result
    monkeypatch.setattr(study, "_arms", slow)
    result = study.freeze(**paths, now=NOW, clock=lambda: current[0])
    assert result["new_trials"] == []
    assert result["rejections"] == {"calculation_finished_outside_pre_match_window": 1}
    assert not list((paths["root"]/"trials").glob("*.json"))


def test_live_historical_xg_and_wrong_source_quotes_never_inform_arm(tmp_path):
    paths, models = setup(tmp_path)
    for model in models:
        packet = source(model)
        for market in packet["provider_snapshot"]["multi_market"].values():
            market["last_source_update_utc"] = (NOW+timedelta(minutes=1)).isoformat()
        packet["provider_snapshot"]["verified_historical_fixture_xg"][0]["kickoff_utc"] = (NOW-timedelta(hours=1)).isoformat()
        packet["provider_snapshot"] = sealed(packet["provider_snapshot"])
        write(source_packet_path(paths["provider_root"], model), packet)
    study.freeze(**paths, now=NOW, clock=lambda: NOW)
    record = json.loads(next((paths["root"]/"trials").glob("*.json")).read_text())
    assert all(record["arms"][method]["status"] == "abstained" for method in study.METHODS[1:])


def test_machine_rounding_tolerance_preserves_authentic_result_but_rejects_probability_change(tmp_path):
    paths, _, path, record = frozen(tmp_path)
    record["arms"]["market_only"]["full_score_grids"][0][0][0] += 2e-16
    write(path, sealed(record, "record_sha256"))
    assert evaluate(paths)["valid_opportunities"] == 1
    record["arms"]["market_only"]["full_score_grids"][0][0][0] += 1e-6
    write(path, sealed(record, "record_sha256"))
    assert evaluate(paths)["valid_opportunities"] == 0
    assert not study._same_calculation(True, 1)
    assert not study._same_calculation(float("nan"), float("nan"))


def test_v3_registration_keeps_previous_version_files_immutable(tmp_path):
    assert study.ROOT.name == "professional_paired_study_v3"
    assert study.VERSION == "professional-fixed-pair-ablation-v3"
    previous = tmp_path/"professional_paired_study_v2"
    (previous/"trials").mkdir(parents=True)
    old_protocol = previous/"protocol.json"
    old_trial = previous/"trials"/"2026-11-23.json"
    old_protocol.write_bytes(b'{"version":"v2"}')
    old_trial.write_bytes(b'{"original":true}')
    active = tmp_path/study.ROOT.name
    record = study.register_protocol(root=active)
    assert record["protocol"]["version"] == study.VERSION
    assert old_protocol.read_bytes() == b'{"version":"v2"}'
    assert old_trial.read_bytes() == b'{"original":true}'
    assert not (active/"trials").exists()
