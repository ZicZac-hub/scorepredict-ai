"""Use real frozen-pair, odds capture, and settlement receipts for V45."""
from datetime import timedelta
import json

from forecast_ledger import read_file_records
from test_v43_prospective import DAY, NOW, setup_original_pair
import v45_market_capture as capture
import v45_market_trial as trial


KICKOFF = NOW + timedelta(hours=2)
EARLY = KICKOFF - timedelta(minutes=30)
DECISION = KICKOFF - timedelta(minutes=15)
LATER = NOW + timedelta(hours=7)


def _odds(model, at, *, home="1.70"):
    bets = [
        {"name": "Match Winner", "values": [
            {"value": "Home", "odd": home}, {"value": "Draw", "odd": "4.25"},
            {"value": "Away", "odd": "4.40"}]},
        {"name": "Goals Over/Under", "values": [
            {"value": "Over 2.5", "odd": "1.85"}, {"value": "Under 2.5", "odd": "1.90"}]},
    ]
    return {
        "metadata": {"provider": "API-Football", "stale": False, "fetched_at_utc": at.isoformat()},
        "data": [{"fixture": {"id": int(model["fixture_id"]), "date": model["kickoff_utc"]},
                  "update": (at - timedelta(minutes=1)).isoformat(),
                  "bookmakers": [{"name": "Book One", "bets": bets},
                                 {"name": "Book Two", "bets": bets}]}],
    }


def _capture(settings, at, *, include=(1, 2), home="1.70"):
    models = {int(model["fixture_id"]): model for model in read_file_records(settings["source_root"])}

    def request(_key, endpoint, params, **_kwargs):
        assert endpoint == "/odds"
        ident = params["fixture"]
        response = _odds(models[ident], at, home=home)
        if ident not in include:
            response["data"] = []
        return response

    return capture.capture(now=at, clock=lambda: at, api_key="test-key", environ={},
                           root=settings["provider_root"], pair_root=settings["pair_root"],
                           source_root=settings["source_root"], broad_root=settings["root"] / "quota",
                           request=request)


def _finals(settings, receipt, count, dest=None):
    raw = {}
    models = sorted(read_file_records(settings["source_root"]), key=lambda model: model["fixture_id"])
    choice = receipt["arms"]["with_market"]["five_doubles"][0]
    for index, model in enumerate(models[:count], 1):
        home, away = map(int, choice[f"score{index}"].split("-"))
        raw[model["fixture_id"]] = {
            **{key: model[key] for key in ("provider", "fixture_id", "home_team_id", "away_team_id", "kickoff_utc")},
            "home_goals": home, "away_goals": away, "score_basis": "regulation", "status": "FT",
            "evidence_version": 2, "verified_at_utc": (LATER - timedelta(hours=1)).isoformat(),
        }
    root = settings["source_root"] if dest is None else dest
    root.mkdir(parents=True, exist_ok=True)
    (root / "verified_results.json").write_text(json.dumps(raw), encoding="utf-8")
    return raw


def test_refreshes_both_markets_and_grades_only_a_complete_matched_pair(tmp_path):
    settings = setup_original_pair(tmp_path)
    assert len(_capture(settings, EARLY)["receipts"]) == 2
    assert trial.freeze(**settings, now=EARLY, clock=lambda: EARLY)["issued"] == []
    assert len(_capture(settings, DECISION, home="1.65")["receipts"]) == 2
    assert trial.freeze(**settings, now=DECISION, clock=lambda: DECISION)["issued"] == [DAY]

    path = settings["root"] / "trials" / f"{DAY}.json"
    original = path.read_bytes()
    receipt = json.loads(original)
    assert [ref["captured_at_utc"] for ref in receipt["market_sources"]] == [DECISION.isoformat()] * 2
    control, adjusted = receipt["arms"]["baseline"], receipt["arms"]["with_market"]
    assert len(control["five_doubles"]) == len(adjusted["five_doubles"]) == 5
    assert control["full_score_grids"] != adjusted["full_score_grids"]

    initial = trial.evaluate(**settings, now=DECISION)
    assert initial["paired_market_coverage"] == 1
    assert initial["pending_kickoff_pairs"] == 1
    assert initial["metrics"]["with_market"]["hit_rate"] is None
    assert trial.freeze(**settings, now=DECISION, clock=lambda: DECISION)["preserved"] == [DAY]

    _finals(settings, receipt, 1)
    pending = trial.evaluate(**settings, now=LATER)
    assert pending["matched_verified_pairs"] == 0 and pending["pending_verification_pairs"] == 1

    _finals(settings, receipt, 2)
    report = trial.evaluate(**settings, now=LATER)
    assert report["matched_verified_pairs"] == 1
    assert report["metrics"]["baseline"]["verified_pairs"] == report["metrics"]["with_market"]["hits"] == 1
    assert report["metrics"]["market_minus_baseline"]["score_log_loss"]["sample_pairs"] == 1
    assert report["pending_verification_pairs"] == report["invalid_receipts"] == 0
    assert path.read_bytes() == original


def test_missing_one_market_records_a_skip_and_never_fills_it_later(tmp_path):
    settings = setup_original_pair(tmp_path)
    report = _capture(settings, DECISION, include=(1,))
    assert len(report["receipts"]) == 1

    assert trial.freeze(**settings, now=DECISION, clock=lambda: DECISION)["abstained"] == [DAY]
    path = settings["root"] / "trials" / f"{DAY}.json"
    original = path.read_bytes()
    receipt = json.loads(original)
    assert receipt["reason"] == "market_missing_on_one_leg"

    after = DECISION + timedelta(minutes=1)
    model = next(model for model in read_file_records(settings["source_root"]) if model["fixture_id"] == "2")
    packet = capture._capture_packet(_odds(model, after), model, observed=after, kickoff=KICKOFF,
                                     phase="late", pair_sha256=receipt["original_pair_sha256"])
    odds_path = capture._source_path(settings["provider_root"], model, packet["provider_snapshot"]["snapshot_sha256"])
    odds_path.write_text(json.dumps(packet), encoding="utf-8")
    assert trial.freeze(**settings, now=after, clock=lambda: after)["preserved"] == [DAY]
    report = trial.evaluate(**settings, now=LATER)
    assert report["paired_market_coverage"] == report["matched_verified_pairs"] == 0
    assert report["skip_rate"] == 1 and report["abstained_pairs"] == 1
    assert path.read_bytes() == original


def test_closed_windows_are_counted_without_historical_freezing(tmp_path):
    settings = setup_original_pair(tmp_path)
    trial.freeze(**settings, now=EARLY, clock=lambda: EARLY)
    close = KICKOFF - timedelta(minutes=5)
    finishing = iter([DECISION, DECISION, close])
    attempt = trial.freeze(**settings, clock=lambda: next(finishing))
    assert attempt["rejections"] == {"decision_window_closed_during_calculation": 1}

    assert trial.freeze(**settings, now=close, clock=lambda: close)["issued"] == []
    report = trial.evaluate(**settings, now=close)
    assert report["missed_windows"] == report["skipped_pairs"] == 1
    assert report["paired_market_coverage"] == 0
    assert not list((settings["root"] / "trials").glob("*.json"))


def test_reproduction_rejects_resigned_tampering_and_conflicting_finals(tmp_path):
    settings = setup_original_pair(tmp_path)
    assert len(_capture(settings, DECISION)["receipts"]) == 2
    assert trial.freeze(**settings, now=DECISION, clock=lambda: DECISION)["issued"] == [DAY]
    path = settings["root"] / "trials" / f"{DAY}.json"
    receipt = json.loads(path.read_text(encoding="utf-8"))
    results = _finals(settings, receipt, 2)
    results["1"]["home_goals"] += 1
    prematch = tmp_path / "prematch"
    prematch.mkdir()
    (prematch / "verified_results.json").write_text(json.dumps(results), encoding="utf-8")

    report = trial.evaluate(**settings, now=LATER, prematch_root=prematch)
    assert report["matched_verified_pairs"] == 0 and report["pending_verification_pairs"] == 1
    assert report["rows"][0]["settlement_conflict"] is True

    receipt["arms"]["with_market"]["five_doubles"][0]["joint_probability"] += .01
    receipt["record_sha256"] = trial._digest(receipt, "record_sha256")
    path.write_text(json.dumps(receipt), encoding="utf-8")
    report = trial.evaluate(**settings, now=LATER)
    assert report["matched_verified_pairs"] == 0
    assert report["receipt_rejections"] == {"market_comparison_does_not_reproduce": 1}
