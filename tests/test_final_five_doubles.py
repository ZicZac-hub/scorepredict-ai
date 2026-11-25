"""Exercise the prospective five-double issue and independent result gates."""

from datetime import date, datetime, timedelta, timezone
import hashlib
from itertools import combinations
import json
import math

import numpy as np
import pytest

import final_five_doubles as final
from forecast_ledger import forecast_id
from v21_capture_sources import source_packet_path
from v28_independent_pair_trial import eligible_slate, freeze as freeze_original
from v41_unified_ensemble import final_pair


CUTOFF = datetime(2035, 2, 21, 8, tzinfo=timezone.utc)
GRIDS = (
    np.outer([.20, .30, .25, .17, .08], [.29, .37, .20, .10, .04]),
    np.outer([.31, .30, .22, .12, .05], [.40, .38, .14, .06, .02]),
    np.outer([.19, .29, .29, .16, .07], [.25, .34, .24, .12, .05]),
    np.outer([.40, .38, .14, .06, .02], [.45, .36, .13, .04, .02]),
)


def model(number, *, captured=None):
    captured = captured or CUTOFF - timedelta(hours=2)
    row = {
        "provider": "API-Football",
        "fixture_id": str(number),
        "model_key": "v8",
        "model_version": "original-v8",
        "mode": "prospective",
        "score_basis": "regulation",
        "home_team_id": str(2 * number),
        "away_team_id": str(2 * number + 1),
        "home_team": f"Home {number}",
        "away_team": f"Away {number}",
        "kickoff_utc": (CUTOFF + timedelta(hours=4, minutes=number)).isoformat(),
        "captured_at_utc": captured.isoformat(),
        "evidence_as_of_utc": (captured - timedelta(minutes=2)).isoformat(),
        "history_matches": 55,
        "input_digest": f"immutable-test-source-{number}",
        "score_matrix": GRIDS[(number - 1) % len(GRIDS)].tolist(),
    }
    row["forecast_id"] = forecast_id(row)
    return row


def write(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record), encoding="utf-8")


def verified_result(row, score, *, verified_at):
    keys = "provider", "fixture_id", "home_team_id", "away_team_id", "kickoff_utc"
    return {
        **{key: row[key] for key in keys},
        "status": "FT",
        "score_basis": "regulation",
        "evidence_version": 2,
        "home_goals": score[0],
        "away_goals": score[1],
        "verified_at_utc": verified_at.isoformat(),
    }


def write_markets(root, model, captured):
    identity = {key: model[key] for key in ("provider", "fixture_id", "home_team_id",
                                             "away_team_id", "kickoff_utc")}
    probabilities = {"1x2": {"home": .7, "draw": .2, "away": .1},
                     "over_2_5": {"yes": .4, "no": .6}}
    snapshot = {**identity, "status": "captured_pre_kickoff",
                "captured_at_utc": captured.isoformat(), "multi_market": {
                    market: {"probabilities": values, "last_source_update_utc": captured.isoformat()}
                    for market, values in probabilities.items()}}
    canonical = json.dumps(snapshot, sort_keys=True, separators=(",", ":")).encode()
    snapshot["snapshot_sha256"] = hashlib.sha256(canonical).hexdigest()
    write(source_packet_path(root, model), {"provider_snapshot": snapshot, "source_forecast": {
        **identity, "forecast_id": model["forecast_id"], "input_digest": model["input_digest"]}})


@pytest.fixture
def frozen_slate(tmp_path):
    source = tmp_path / "source"
    v28 = tmp_path / "v28"
    models = [model(i) for i in range(1, 5)]
    for row in models:
        path = source / "forecasts" / row["provider"] / row["fixture_id"]
        write(path / (row["forecast_id"] + ".json"), row)
    assert len(freeze_original(root=source, out=v28, now=CUTOFF)["created"]) == 1
    return source, v28, models


def brute_force_mass(first, second):
    cells = sorted((float(a * b) for a in first.flat for b in second.flat), reverse=True)
    return math.fsum(cells[:5])


def study_paths(tmp_path, frozen_slate):
    source, v28, _ = frozen_slate
    return {"root": tmp_path / "final", "v28_out": v28, "source_root": source,
            "provider_root": tmp_path / "provider"}


def historical_rows(wins, opportunities=None):
    count = opportunities if opportunities is not None else final.MINIMUM_VERIFIED
    first = date(2034, 1, 1)
    return [{"sydney_day": (first + timedelta(days=i)).isoformat(),
             "status": "verified", "hit": i < wins} for i in range(count)]


def test_complete_slate_matches_brute_force_for_each_pair(tmp_path):
    groups, _ = eligible_slate([model(i) for i in range(1, 5)], CUTOFF)
    items = next(iter(groups.values()))
    settings = {"provider_root": tmp_path, "cutoff": CUTOFF}
    comparison = final.compare_slate(items, **settings)
    expected = {}
    for left, right in combinations(items, 2):
        grids, _ = final_pair([left["grid"], right["grid"]], [{}, {}])
        expected[(left["key"], right["key"])] = brute_force_mass(*grids)

    assert comparison["status"] == "compared"
    assert comparison["compared_pairs"] == comparison["eligible_pairs"] == 6
    for row in comparison["candidate_catalog"]:
        keys = tuple(map(tuple, row["fixture_keys"]))
        assert row["model_mass"]["unified_final"] == pytest.approx(expected[keys])
    selection = comparison["selection"]
    assert selection["estimate"] == pytest.approx(max(expected.values()))
    assert len({(row["score1"], row["score2"]) for row in selection["five_doubles"]}) == 5
    assert final.compare_slate(reversed(items), **settings) == comparison


def test_comparison_budget_abstains_without_selecting_a_partial_slate(tmp_path):
    groups, _ = eligible_slate([model(i) for i in range(1, 5)], CUTOFF)
    comparison = final.compare_slate(next(iter(groups.values())), provider_root=tmp_path,
                                     cutoff=CUTOFF, max_pairs=5)
    assert comparison["status"] == "abstained"
    assert comparison["eligible_pairs"] == 6
    assert comparison["compared_pairs"] == 0
    assert comparison["candidate_catalog"] == []
    assert comparison["selection"] is None


def test_market_evidence_requires_two_qualified_legs_before_cutoff(tmp_path):
    models = [model(i) for i in range(1, 4)]
    for row in models[:2]:
        write_markets(tmp_path, row, CUTOFF - timedelta(minutes=5))
    write_markets(tmp_path, models[2], CUTOFF + timedelta(minutes=5))
    groups, _ = eligible_slate(models, CUTOFF)
    output = final.compare_slate(next(iter(groups.values())), provider_root=tmp_path, cutoff=CUTOFF)
    assert output["evidence_counts"]["market"] == 1
    for pair in output["candidate_catalog"]:
        affected = tuple(key[1] for key in pair["fixture_keys"]) == ("1", "2")
        assert pair["paired_evidence"]["market"] is affected
        changed = pair["model_mass"]["unified_core"] != pytest.approx(pair["model_mass"]["unified_final"])
        assert changed is affected


def test_later_candidate_failure_prevents_issuing_a_partial_winner(tmp_path, frozen_slate, monkeypatch):
    settings = study_paths(tmp_path, frozen_slate)
    original = final._choices
    evaluated = 0

    def fail_on_second_pair(*grids):
        nonlocal evaluated
        evaluated += 1
        if evaluated == 4:
            raise ValueError("invalid score model for later candidate")
        return original(*grids)

    monkeypatch.setattr(final, "_choices", fail_on_second_pair)
    assert final.freeze(**settings, now=CUTOFF)
    comparison = json.loads(next((settings["root"] / "decisions").glob("*.json")).read_text())["comparison"]
    assert comparison["status"] == "abstained"
    assert comparison["selection"] is None and comparison["candidate_catalog"] == []


def test_validation_requires_a_supported_rate_and_reports_pending_separately():
    n = final.MINIMUM_VERIFIED
    assert not final.validation_summary(historical_rows(n - 1, n - 1))["ready"]
    observed_at_target = historical_rows(math.ceil(final.TARGET * n))
    assert not final.validation_summary(observed_at_target)["ready"]
    supported = historical_rows(n)
    assert final.validation_summary(supported)["ready"]
    with_pending = supported + [{"sydney_day": "2034-12-30", "status": "pending", "hit": None}]
    result = final.validation_summary(with_pending)
    assert result["wins"] == result["verified"] == n
    assert result["pending"] == 1
    assert result["observed_hit_rate"] == 1

    early_pending = [{**supported[0], "status": "pending", "hit": None},
                     *supported[1:], {"sydney_day": "2034-12-31", "status": "verified", "hit": True}]
    assert final.validation_summary(early_pending)["verified"] == n
    assert not final.validation_summary(early_pending)["ready"]


def test_frozen_selection_is_immutable_and_settles_only_with_two_available_finals(tmp_path, frozen_slate):
    settings = study_paths(tmp_path, frozen_slate)
    assert final.freeze(**settings, now=CUTOFF)
    packet_path = next((settings["root"] / "decisions").glob("*.json"))
    assert not final.freeze(**settings, now=CUTOFF)
    original = packet_path.read_bytes()
    packet = json.loads(original)
    selection = packet["comparison"]["selection"]
    models = {row["fixture_id"]: row for row in frozen_slate[2]}
    score = selection["five_doubles"][0]
    observed = CUTOFF + timedelta(hours=9)
    results = {}
    for (_, fixture_id), option, verified_at in zip(selection["fixture_keys"],
            (score["score1"], score["score2"]), (observed, observed + timedelta(hours=1))):
        results[fixture_id] = verified_result(models[fixture_id], tuple(map(int, option.split("-"))),
                                              verified_at=verified_at)
    prematch = tmp_path / "prematch"
    write(prematch / "verified_results.json", results)

    pending = final.evaluate(**settings, prematch=prematch, now=observed)
    assert pending["rows"][0]["status"] == "pending"
    completed = final.evaluate(**settings, prematch=prematch, now=observed + timedelta(hours=1))
    assert completed["rows"][0]["status"] == "verified"
    assert completed["rows"][0]["hit"] is True
    assert packet_path.read_bytes() == original

    altered = {key: value for key, value in packet.items() if key != "sha256"}
    altered["comparison"]["selection"]["estimate"] = .99
    write(packet_path, final._signed(altered))
    reviewed = final.evaluate(**settings, prematch=prematch, now=observed + timedelta(hours=1))
    assert reviewed["rows"][0]["status"] == "invalid"


def test_old_slate_cannot_be_frozen_after_the_original_decision_window(tmp_path, frozen_slate):
    settings = study_paths(tmp_path, frozen_slate)
    assert final.freeze(**settings, now=CUTOFF + timedelta(minutes=21)) == []
    assert list((settings["root"] / "decisions").glob("*.json")) == []
    assert final.evaluate(**settings, now=CUTOFF + timedelta(days=1))["rows"] == []


def test_missed_post_activation_slate_is_an_abstention_without_a_ticket(tmp_path, frozen_slate):
    settings = study_paths(tmp_path, frozen_slate)
    write(settings["root"] / "activation.json", {
        "version": final.VERSION, "first_window_utc": (CUTOFF - timedelta(hours=1)).isoformat(),
    })
    later = CUTOFF + timedelta(days=1)
    assert final.freeze(**settings, now=later) == []
    report = final.evaluate(**settings, now=later)
    assert len(report["rows"]) == 1
    assert report["rows"][0]["status"] == "abstained"
    assert report["rows"][0]["reason"] == "scheduled_freeze_window_missed"
    assert report["rows"][0]["compared_pairs"] == 0
    assert report["validation"]["abstained"] == 1
    assert report["validation"]["verified"] == 0
    assert not list((settings["root"] / "decisions").glob("*.json"))


def test_public_view_requires_validation_and_expires_with_kickoff(tmp_path, frozen_slate, monkeypatch):
    settings = study_paths(tmp_path, frozen_slate)
    assert final.freeze(**settings, now=CUTOFF, clock=lambda: CUTOFF)
    report = final.evaluate(**settings, prematch=tmp_path / "prematch", now=CUTOFF)
    root = settings["root"]
    write(root / "status.json", report)
    waiting = final.presentation(root=root, now=CUTOFF)
    assert waiting["state"] == "awaiting_validation"
    assert "names" not in waiting["decision"] and "five_doubles" not in waiting["decision"]

    report["rows"] = historical_rows(final.MINIMUM_VERIFIED) + report["rows"]
    report["validation"] = final.validation_summary(report["rows"][:-1])
    write(root / "status.json", report)
    # Fabricated status rows alone must not unlock the UI.
    forged = final.presentation(**settings, prematch=tmp_path / "prematch", now=CUTOFF)
    assert forged["state"] == "evidence_unavailable"
    assert "names" not in forged["decision"] and "five_doubles" not in forged["decision"]
    # Positive UI-path test: simulate the independent evaluator authenticating
    # the full historical cohort and current original decision.
    monkeypatch.setattr(final, "evaluate", lambda **kwargs: report)
    released = final.presentation(root=root, now=CUTOFF)
    assert released["state"] == "available"
    assert len(released["decision"]["names"]) == 2
    assert len(released["decision"]["five_doubles"]) == 5

    kickoff = datetime.fromisoformat(report["rows"][-1]["first_kickoff_utc"])
    expired = final.presentation(root=root, now=kickoff)
    assert expired["state"] == "expired"
    assert "names" not in expired["decision"] and "five_doubles" not in expired["decision"]
