"""V27 same-day full-slate selection: no hindsight, complete slate, honest grades."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json

import numpy as np
import pytest

from forecast_ledger import forecast_id
from professional_analyst_shadow_v25 import propose
from v27_pair_selection import (
    VERSION, build_slate, freeze, evaluate, candidate_entry,
)

SELECT = datetime(2035, 2, 20, 10, tzinfo=timezone.utc)
KICK = SELECT + timedelta(hours=6)
A = np.outer([.20, .32, .25, .15, .08],
             [.31, .37, .19, .10, .03]).tolist()


def model(fid, *, age_hours=3, capture=None, matrix=None):
    captured = capture if capture is not None else SELECT - timedelta(hours=age_hours)
    score = matrix if matrix is not None else A
    data = {
        "provider": "API-Football",
        "fixture_id": str(fid),
        "model_key": "v8",
        "model_version": "archived-v8",
        "mode": "prospective",
        "score_basis": "regulation",
        "home_team_id": str(fid * 2),
        "away_team_id": str(fid * 2 + 1),
        "home_team": f"Home {fid}",
        "away_team": f"Away {fid}",
        "kickoff_utc": (KICK+timedelta(minutes=(fid-1)*8)).isoformat(),
        "captured_at_utc": captured.isoformat(),
        "evidence_as_of_utc": (captured-timedelta(minutes=1)).isoformat(),
        "input_digest": f"immutable-examples-{fid}",
        "history_matches": 55,
        "score_matrix": score,
    }
    data["forecast_id"] = forecast_id(data)
    return data


def verified(row, goals):
    return {
        **{key: row[key] for key in (
            "provider", "fixture_id", "home_team_id", "away_team_id",
            "kickoff_utc")},
        "home_goals": goals[0], "away_goals": goals[1],
        "score_basis": "regulation", "status": "FT", "evidence_version": 2,
        "verified_at_utc": (
            datetime.fromisoformat(row["kickoff_utc"])+timedelta(hours=3)
        ).isoformat(),
    }


def setup(tmp, *, ages=(110, 105, 2, 1), mats=None):
    root = tmp/"root"
    prematch = tmp/"prematch"
    v25 = tmp/"v25"
    v27 = tmp/"v27"
    prematch.mkdir()
    originals = [model(i, age_hours=age, matrix=mats[i-1] if mats else None)
                 for i, age in enumerate(ages, 1)]
    for row in originals:
        file = root/"forecasts"/row["provider"]/row["fixture_id"]/(row["forecast_id"]+".json")
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text(json.dumps(row))
    first = propose(originals, SELECT, min_envelope=0)
    packet = next(iter(first["days"].values()))
    assert packet["decision"] == "FROZEN_RESEARCH_ONLY"
    v25.mkdir()
    saved = v25/(packet["sydney_date"]+".json")
    saved.write_text(json.dumps(packet))
    return root, prematch, v25, v27, saved, originals


def test_all_pairs_ranked_provenance_frozen_and_same_day_verified(tmp_path):
    root, prematch, sets, out, saved, originals = setup(tmp_path)
    first = saved.read_bytes()
    issue_time = SELECT+timedelta(minutes=1)
    frozen = freeze(root=root, sets=sets, out=out, now=issue_time)
    assert len(frozen["new_decisions"]) == 1
    assert frozen["new_decisions"][0]["decision"] == "V27_SELECTED"
    file = next((out/"decisions").glob("*.json"))
    packet = json.loads(file.read_text())
    assert packet["version"] == VERSION
    assert packet["qualified_four_team_pairs"] == 6
    assert packet["evaluated_pairs"] == 6
    assert len(packet["candidate_catalog"]) == 6
    assert packet["candidate_catalog"][0] == packet["selected_pair"]
    assert len(packet["selected_pair"]["five_robust"]) == 5
    assert len({(r["score1"], r["score2"]) for r in
                packet["selected_pair"]["five_robust"]}) == 5
    assert packet["source_receipts_sha256"]
    assert packet["production_model_promoted"] is False
    assert saved.read_bytes() == first
    original_snapshot = deepcopy(originals)
    untouched = file.read_bytes()
    assert freeze(root=root, sets=sets, out=out, now=issue_time)[
        "new_decisions"] == []
    assert file.read_bytes() == untouched

    pending = evaluate(root=root, prematch=prematch, sets=sets, out=out, now=issue_time)
    assert pending["both_policies_verified_same_days"] == 0
    assert pending["v27_pending_pair_results"] == 1
    assert pending["rows"][0]["status"] == "pending"

    first_pair = packet["selected_pair"]["fixture_keys"]
    winning = packet["selected_pair"]["five_robust"][0]
    result = {r["fixture_id"]: verified(r, (0, 0)) for r in originals}
    for (provider, fid), score in zip(
        first_pair, (winning["score1"], winning["score2"])):
        row = next(r for r in originals if r["fixture_id"] == fid)
        result[fid] = verified(row, tuple(map(int, score.split("-"))))
    (prematch/"verified_results.json").write_text(json.dumps(result))
    after = KICK+timedelta(hours=6)
    done = evaluate(root=root, prematch=prematch, sets=sets, out=out, now=after)
    assert done["invalid_or_tampered_decisions_excluded"] == {}
    assert done["v27_verified_pair_results"] == 1
    assert done["both_policies_verified_same_days"] == 1
    assert done["wins_on_same_completed_days"]["v27_selected_robust"] == 1
    assert done["rows"][0]["model_mass_is_not_measured_accuracy"] is True
    assert done["70_percent_five_double_accuracy_proven"] is False
    assert saved.read_bytes() == first
    assert file.read_bytes() == untouched
    assert originals == original_snapshot


def test_freshness_ranking_does_not_change_five_score_probabilities(tmp_path):
    _, _, sets, _, saved, originals = setup(tmp_path)
    # Same model grids means V25's deterministic tie-break is the oldest
    # pair, whereas the predeclared V27 freshness factor favours fresh pair.
    day = json.loads(saved.read_text())["sydney_date"]
    slate = build_slate(originals, SELECT, day)
    assert slate["decision"] == "V27_SELECTED"
    chosen = [p[1] for p in slate["selected_pair"]["fixture_keys"]]
    assert chosen == ["3", "4"]
    assert slate["selected_pair"]["freshness_multiplier"] == 1.0
    assert slate["candidate_catalog"][0]["rank_quality"] > (
        slate["candidate_catalog"][-1]["rank_quality"])
    assert slate["selected_pair"]["original_v8_top_five_model_mass"] < 0.70


def test_single_candidate_pair_abstains_even_if_v25_selected(tmp_path):
    root, prematch, sets, out, _, originals = setup(tmp_path, ages=(2, 1))
    frozen = freeze(root=root, sets=sets, out=out,
                    now=SELECT+timedelta(minutes=1))
    assert frozen["new_decisions"][0]["decision"] == "V27_ABSTAIN"
    report = evaluate(root=root, prematch=prematch, sets=sets, out=out,
                      now=SELECT+timedelta(minutes=1))
    assert report["abstained_days"] == 1
    assert report["frozen_decision_days"] == 1
    assert report["both_policies_verified_same_days"] == 0
    assert report["rows"][0]["reason"] == "fewer_than_two_distinct_eligible_pairs"


def test_future_captured_source_not_eligible_for_old_cutoff(tmp_path):
    _, _, _, _, packet, originals = setup(tmp_path)
    day = json.loads(packet.read_text())["sydney_date"]
    extra = model(99, capture=SELECT+timedelta(seconds=15))
    original = build_slate(originals, SELECT, day)
    after = build_slate(originals+[extra], SELECT, day)
    assert after["qualified_four_team_pairs"] == original["qualified_four_team_pairs"]
    assert after["candidate_catalog_sha256"] == original["candidate_catalog_sha256"]
    assert after["excluded_original_sources"]["invalid_source_time"] >= 1


def test_large_slate_abstains_and_does_not_cherry_pick(tmp_path):
    _, _, _, _, packet, originals = setup(tmp_path)
    day = json.loads(packet.read_text())["sydney_date"]
    limited = build_slate(originals, SELECT, day, max_pairs=2)
    assert limited["qualified_four_team_pairs"] == 6
    assert limited["candidate_catalog"] == []
    assert limited["evaluated_pairs"] == 0
    assert limited["decision"] == "V27_ABSTAIN"
    assert limited["reason"] == "too_many_pairs_to_compare_without_truncation"


def test_old_v25_packet_never_receives_retroactive_v27_predictions(tmp_path):
    root, _, sets, out, _, _ = setup(tmp_path)
    result = freeze(root=root, sets=sets, out=out,
                    now=SELECT+timedelta(minutes=21))
    assert result["new_decisions"] == []
    assert result["not_created"]["not_a_new_v25_freeze"] == 1
    assert not (out/"decisions").exists()


def test_tampered_selected_policy_and_immutable_source_are_excluded(tmp_path):
    root, prematch, sets, out, _, _ = setup(tmp_path)
    issued = SELECT+timedelta(minutes=1)
    freeze(root=root, sets=sets, out=out, now=issued)
    file = next((out/"decisions").glob("*.json"))
    original = file.read_bytes()
    data = json.loads(original)
    data["selected_pair"]["rank_quality"] = 0.99
    file.write_text(json.dumps(data))
    invalid = evaluate(root=root, prematch=prematch, sets=sets, out=out, now=issued)
    assert invalid["frozen_decision_days"] == 0
    assert "candidate_catalog_or_rank_changed_selected_pair" in (
        invalid["invalid_or_tampered_decisions_excluded"])
    file.write_bytes(original)
    source = root/"forecasts"/"API-Football"/"3"/(
        next(r for r in data["eligible_original_source_receipts"]
             if r["fixture_id"] == "3")["forecast_id"]+".json")
    model_json = json.loads(source.read_text())
    model_json["input_digest"] = "post-hoc-tampering"
    source.write_text(json.dumps(model_json))
    rejected = evaluate(root=root, prematch=prematch, sets=sets, out=out, now=issued)
    assert rejected["frozen_decision_days"] == 0
    assert rejected["invalid_or_tampered_decisions_excluded"]


def test_candidate_threshold_is_research_only_not_win_probability(tmp_path):
    _, _, _, _, packet, originals = setup(tmp_path)
    day = json.loads(packet.read_text())["sydney_date"]
    slate = build_slate(originals, SELECT, day)
    selected = slate["selected_pair"]
    assert selected["rank_quality"] < 1
    assert selected["rank_quality"] > 0
    assert selected["v25_lower_envelope_is_uncalibrated"] is True
