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
import v41_upset_trial as study
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






def test_upset_correction_is_bounded_and_home_away_symmetric():
    from v41_upset_aware import adjust
    p=np.zeros((6,6))
    p[2,0]=.65
    p[0,1]=.15
    p[1,1]=.20
    before=p.copy()
    q,info=adjust(p)
    assert info["favorite"]=="home" and info["applied"]
    assert info["underdog_after"]>info["underdog_before"]
    assert info["draw_after"]>info["draw_before"]
    assert info["favorite_after"]<info["favorite_before"]
    assert info["total_variation"]<=.05+1e-12
    assert np.isclose(q.sum(),1) and np.all(q>=0)
    assert np.array_equal(p,before)
    swapped,meta=adjust(p.T)
    assert meta["favorite"]=="away"
    assert np.allclose(swapped,q.T)


def test_balanced_match_is_not_forced_into_upset():
    from v41_upset_aware import adjust
    p=np.outer([.3,.35,.2,.1,.04,.01],[.3,.35,.2,.1,.04,.01])
    q,info=adjust(p)
    assert info["applied"] is False and info["favorite"] is None
    assert np.allclose(q,p)


def test_zero_support_is_not_invented():
    from v41_upset_aware import adjust
    p=np.zeros((5,5)); p[2,0]=1
    q,_=adjust(p)
    assert np.array_equal(p,q)


def test_freeze_compare_same_pair_and_preserve_archive(tmp_path):
    paths,models,path,record=frozen(tmp_path)
    assert set(record["arms"])=={"unified_final","upset_aware"}
    for arm in record["arms"].values():
        assert len({(p["score1"],p["score2"]) for p in arm["five_doubles"]})==5
    before=path.read_bytes()
    settled(paths,models)
    report=evaluate(paths)
    assert report["comparison_control"]=="unified_final"
    assert report["paired_comparisons"]["upset_aware"]["verified_common_pairs"]==1
    assert report["model_promoted"] is False
    assert report["target_70_percent_proven"] is False
    assert path.read_bytes()==before
