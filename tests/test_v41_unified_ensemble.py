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
import v41_unified_trial as study
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





def test_core_is_normalized_and_never_mutates_source():
    from v41_unified_ensemble import core, WEIGHTS
    p = np.outer([.30,.35,.20,.10,.04,.01],[.30,.35,.20,.10,.04,.01])
    before=p.copy()
    q=core(p)
    assert np.array_equal(p,before)
    assert np.isclose(sum(WEIGHTS.values()),1)
    assert np.isclose(q.sum(),1) and np.all(q>=0) and q.shape==p.shape
    assert np.array_equal(q,core(p))


def test_one_final_output_frozen_and_verified(tmp_path):
    paths, models, path, record = frozen(tmp_path)
    final=record["arms"]["unified_final"]
    assert final["status"]=="issued"
    assert final["single_final_five_double_output"] is True
    assert final["paired_xg_used"] is True
    assert final["paired_market_used"] is True
    assert len({(p["score1"],p["score2"]) for p in final["five_doubles"]})==5
    assert np.isclose(final["estimated_coverage"],sum(p["joint_probability"] for p in final["five_doubles"]))
    original=path.read_bytes()
    settled(paths,models)
    report=evaluate(paths)
    assert report["single_final_arm"]=="unified_final"
    assert report["methods"]["unified_final"]["verified"]==1
    assert report["paired_comparisons"]["unified_final"]["verified_common_pairs"]==1
    assert report["target_70_percent_proven"] is False and report["model_promoted"] is False
    assert path.read_bytes()==original


def test_partial_or_missing_sources_do_not_invent_evidence(tmp_path):
    paths, models=setup(tmp_path)
    source_packet_path(paths["provider_root"],models[0]).unlink()
    result=study.freeze(**paths,now=NOW,clock=lambda:NOW)
    assert result["new_trials"]==["2035-01-01"]
    record=json.loads((paths["root"]/"trials"/"2035-01-01.json").read_text())
    final=record["arms"]["unified_final"]
    assert final["paired_xg_used"] is False and final["paired_market_used"] is False
    assert np.allclose(final["full_score_grids"],record["arms"]["unified_core"]["full_score_grids"])


def test_v41_fixed_protocol_is_separate():
    assert study.ROOT.name=="v41_unified_ensemble"
    assert study.PROTOCOL["one_final_arm"]=="unified_final"
    assert study.PROTOCOL["minimum_review_verified_pairs"]==100
    assert study.PROTOCOL["automatic_model_promotion"] is False
    assert study.PROTOCOL["numerical_lineup_weight"]==0
    assert study.PROTOCOL["numerical_ai_weight"]==0
