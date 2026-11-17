"""Read-only, historical V25 goal-model reality check, not model fitting.

Sources come from an existing time-aware retrospective replay. These are
development data, not prospective wins; no goal weights are changed here.
"""
import json
from pathlib import Path

BACKTEST = Path("data/accuracy_diagnosis/historical_replay.json")


def historical_reality_check(path=BACKTEST):
    try:
        report=json.loads(Path(path).read_text(encoding="utf-8"))
        if report.get("kind") != "retrospective_walk_forward_replay_not_genuine_future_accuracy":
            return {"status":"unavailable","reason":"not_a_verified_historical_replay"}
        pair=report.get("same_pair_exact_five") or {}
        baseline=(report.get("paired_v8_v20_same_fixtures") or {}).get(
            "original_v8_on_eligible_fixtures") or {}
        methods=pair.get("methods") or {}
        v8=methods.get("baseline_v8") or {}
        if (type(baseline.get("evaluated")) is not int or baseline["evaluated"]<=0 or
                type(v8.get("issued")) is not int or v8["issued"]<=0):
            return {"status":"unavailable","reason":"no_complete_historical_sample"}
        actual=float(baseline["actual_high_score_rate"])
        predicted=float(baseline["predicted_high_score_mass"])
        if not all(0<=x<=1 for x in (actual,predicted)):
            return {"status":"unavailable","reason":"invalid_probability_statistics"}
        return {
            "status":"ready",
            "retrospective_single_fixtures":baseline["evaluated"],
            "high_goal_definition":"at least 4 combined regulation-time goals",
            "observed_high_goal_share":actual,
            "predicted_high_goal_mass":predicted,
            "high_goal_calibration_gap_observed_minus_predicted":actual-predicted,
            "baseline_v8_exact_single_hits":int(baseline["exact_hits"]),
            "baseline_v8_exact_single_tested":baseline["evaluated"],
            "retrospective_five_double_issued":v8["issued"],
            "retrospective_five_double_wins":int(v8["wins"]),
            "other_model_replay":{name:{"issued":int(m["issued"]), "wins":int(m["wins"])}
                                  for name,m in methods.items()
                                  if isinstance(m,dict) and type(m.get("issued")) is int
                                  and type(m.get("wins")) is int},
            "development_data_only":True,
            "safe_to_claim_new_model_is_better":False,
            "high_score_multiplier_applied":False,
            "future_paired_validation_required":True,
        }
    except (OSError, ValueError, TypeError, KeyError, OverflowError):
        return {"status":"unavailable","reason":"historical_replay_missing_or_corrupt"}
