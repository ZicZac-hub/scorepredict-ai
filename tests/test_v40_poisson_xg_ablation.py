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
import v40_poisson_xg_ablation as study
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




def test_v40_frozen_arms_and_incremental_matched_comparison(tmp_path):
    paths, models, path, record = frozen(tmp_path)
    assert set(record["arms"]) == set(study.METHODS)
    for arm in record["arms"].values():
        assert arm["status"] == "issued"
        assert len(arm["five_doubles"]) == 5
        for grid in arm["full_score_grids"]:
            assert np.isclose(np.asarray(grid).sum(),1)
    original = path.read_bytes()
    settled(paths, models)
    report = evaluate(paths)
    assert report["incremental_xg_comparison"]["verified_common_pairs"] == 1
    assert report["target_70_percent_proven"] is False
    assert report["model_promoted"] is False
    assert path.read_bytes() == original


def test_v40_missing_xg_is_abstention_not_loss(tmp_path):
    paths, models, path, record = frozen(tmp_path,available=False)
    assert record["arms"]["poisson_marginal"]["status"] == "issued"
    assert record["arms"]["conditional_margin_125"]["status"] == "issued"
    assert record["arms"]["poisson_xg"]["status"] == "abstained"
    assert record["arms"]["xg_only"]["status"] == "abstained"
    settled(paths, models)
    report = evaluate(paths)
    assert report["incremental_xg_comparison"]["verified_common_pairs"] == 0
    assert report["methods"]["poisson_xg"]["hit_rate"] is None


def test_v40_archive_is_independent():
    assert study.ROOT.name == "v40_poisson_xg_ablation"
    assert study.PROTOCOL["xg_blend_weight"] == .25
    assert study.PROTOCOL["numerical_lineup_weight"] == 0
    assert study.PROTOCOL["market_numerical_weight"] == 0
    assert study.PROTOCOL["automatic_model_promotion"] is False
