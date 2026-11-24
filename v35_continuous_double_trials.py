"""V35: continuously freeze & independently grade new five-double score methods.

Only a genuinely new V28 original pair can supply candidate fixtures. Five
methods use identical pre-match source grids, same pairs and fixed prespecified
parameters. Original V8/V25 scores and all result evidence are immutable.
No hindsight backfill, betting claims, provider calls, or automatic promotion.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
from math import comb, fsum
import json
import math
from pathlib import Path

import numpy as np

from five_double_accuracy import joint_geometry, utc, wilson
from v25_score_accuracy_research import describe_score
from v25_anchored_challenger import (
    _anchored_grid, _frozen_eligible, MAX_DECISION_AGE,
)
from v26_temporal_score import _write_once
from v28_independent_pair_trial import (
    ORIGINAL_ROOT, OUT as V28_ROOT, _key, verify_decision,
)
from professional_analyst_shadow_v25 import MIN_LEAD
from v30_full_slate_audit import analyze as audit_original_full_slates

VERSION = "v35-fixed-forward-score-challenger-research-v1"
ROOT = Path("data/v35_continuous_challengers")
METHODS = (
    "v8_original",
    "v25_anchored_75_25",
    "temperature_flatten_1_15",
    "temperature_sharpen_0_85",
    "high_goal_tilt_0_15",
    "low_goal_tilt_0_15",
)
MIN_REVIEW_DAYS = 100
MIN_REVIEW_UNIQUE_FIXTURES = 100
MIN_WILSON_LOWER_HIT_RATE = 0.70


def score_grid(item, method):
    """All methods and constants fixed here, before any prospectively graded day."""
    base = np.asarray(item["grid"], dtype=float)
    if (base.ndim != 2 or min(base.shape) < 5 or
            not np.isfinite(base).all() or np.any(base < 0) or
            not np.isclose(base.sum(), 1.0, atol=1e-6)):
        raise ValueError("invalid_original_immutable_score_grid")
    if method == "v8_original":
        candidate = base.copy()
    elif method == "v25_anchored_75_25":
        candidate = np.asarray(_anchored_grid(item), dtype=float)
    elif method in ("temperature_flatten_1_15", "temperature_sharpen_0_85"):
        t = 1.15 if method == "temperature_flatten_1_15" else 0.85
        candidate = np.power(base, 1.0/t)
    elif method in ("high_goal_tilt_0_15", "low_goal_tilt_0_15"):
        h, a = np.indices(base.shape)
        excess = np.clip(h + a - 3, -3, 5)
        beta = 0.15 if method == "high_goal_tilt_0_15" else -0.15
        candidate = base * np.exp(beta * excess)
    else:
        raise ValueError("unregistered_forecasting_method")
    if (candidate.shape != base.shape or not np.isfinite(candidate).all() or
            np.any(candidate < 0) or candidate.sum() <= 0):
        raise ValueError("invalid_challenger_distribution")
    candidate /= float(candidate.sum())
    return candidate.tolist()


def original_portfolio(original, items, issued):
    """Preserve original V28 QUALITY pair and issue all methods simultaneously."""
    if original.get("policy") != "FROZEN" or not _frozen_eligible(original, issued):
        raise ValueError("outside_original_20_minute_pre_kickoff_issue_window")
    selection = original["quality_selection"]
    pair = selection["fixture_keys"]
    if len(pair) != 2 or len({tuple(x) for x in pair}) != 2:
        raise ValueError("invalid_two_independent_fixture_keys")
    source = {_key(item): item for item in items}
    if any(tuple(k) not in source for k in pair):
        raise ValueError("original_quality_pair_source_missing")
    legs = [source[tuple(k)] for k in pair]
    if any(issued > x["kickoff"] - MIN_LEAD for x in legs):
        raise ValueError("candidate_not_at_least_45_minutes_before_both_kickoffs")
    if len(legs[0]["teams"] | legs[1]["teams"]) != 4:
        raise ValueError("not_four_distinct_teams")
    policies = {}
    for method in METHODS:
        grids = [score_grid(item, method) for item in legs]
        five = joint_geometry(*grids)["optimal_five"]
        if len(five) != 5 or len({(x["score1"], x["score2"]) for x in five}) != 5:
            raise ValueError("not_five_distinct_joint_score_options")
        policies[method] = {
            "score_grids": grids,
            "five_joint_scores": five,
        }
    return {
        "version": VERSION,
        "sydney_day": original["sydney_day"],
        "frozen_at_utc": utc(issued).isoformat(),
        "original_v28_issued_at_utc": utc(original["actually_issued_at_utc"]).isoformat(),
        "original_catalog_sha256": original["candidate_catalog_sha256"],
        "original_selected_fixture_keys": pair,
        "original_source_forecast_ids": selection["source_forecast_ids"],
        "original_v25_robust_five": selection["five_original_v25_robust_scores"],
        "original_v8_five": selection["five_original_v8_optimal_scores"],
        "predeclared_methods": list(METHODS),
        "policies": policies,
        "same_original_fixture_pair": True,
        "no_live_model_change": True,
        "no_external_provider_calls": True,
        "not_hindsight_selected": True,
    }


def freeze(*, root=ORIGINAL_ROOT, v28_root=V28_ROOT, out=ROOT, now=None):
    clock = utc(now or datetime.now(timezone.utc))
    made, reasons = [], Counter()
    for file in sorted((Path(v28_root)/"decisions").glob("*.json")):
        dest = Path(out)/"sets"/file.name
        if dest.exists():
            reasons["already_frozen"] += 1
            continue
        try:
            original, items = verify_decision(file, root=root, clock=clock)
            if not _frozen_eligible(original, clock):
                reasons["not_new_original_v28_decision"] += 1
                continue
            issued = clock if now is not None else utc(datetime.now(timezone.utc))
            if not _frozen_eligible(original, issued):
                reasons["generation_exceeded_original_20_minute_window"] += 1
                continue
            packet = original_portfolio(original, items, issued)
            if _write_once(dest, packet):
                made.append(file.stem)
        except (OSError, KeyError, ValueError, IndexError, TypeError, OverflowError) as exc:
            reasons[str(exc) if isinstance(exc, ValueError) else type(exc).__name__] += 1
    return {"created_days": made, "skipped": dict(reasons)}



def _same_portfolio_with_float_tolerance(reproduced, saved):
    """Strict source/ticket equality with 1e-12 absolute numeric tolerance.

    Only cross-environment NumPy rounding may differ. Original V28 receipts,
    method names, every selected scoreline and their ordering MUST match.
    No frozen file is rewritten and no changed ticket can become verified.
    """
    if not isinstance(saved,dict) or set(saved)!=set(reproduced):
        return False
    if any(saved[k]!=v for k,v in reproduced.items() if k!="policies"):
        return False
    if not isinstance(saved.get("policies"),dict) or (
            set(saved["policies"])!=set(reproduced["policies"])):
        return False
    for method, expected in reproduced["policies"].items():
        candidate=saved["policies"].get(method)
        if not isinstance(candidate,dict) or set(candidate)!=set(expected):
            return False
        frozen_grids=candidate["score_grids"]
        expected_grids=expected["score_grids"]
        if not isinstance(frozen_grids,list) or len(frozen_grids)!=2:
            return False
        for frozen, regenerated in zip(frozen_grids,expected_grids):
            try:
                if any(type(v) not in (float,int) for row in frozen for v in row):
                    return False
                x=np.asarray(frozen,dtype=float)
                y=np.asarray(regenerated,dtype=float)
                if (x.ndim!=2 or x.shape!=y.shape or min(x.shape)<5 or
                        not np.isfinite(x).all() or np.any(x<0) or
                        abs(float(x.sum())-1)>1e-6 or
                        not np.allclose(x,y,rtol=0,atol=1e-12)):
                    return False
            except (TypeError,ValueError,OverflowError):
                return False
        frozen_picks=candidate["five_joint_scores"]
        expected_picks=expected["five_joint_scores"]
        if not isinstance(frozen_picks,list) or len(frozen_picks)!=5 or len(expected_picks)!=5:
            return False
        if len({(p.get("score1"),p.get("score2")) for p in frozen_picks
                if isinstance(p,dict)})!=5:
            return False
        for old,new in zip(frozen_picks,expected_picks):
            if (not isinstance(old,dict) or set(old)!=set(new) or
                    old.get("score1")!=new["score1"] or
                    old.get("score2")!=new["score2"]):
                return False
            p=old["joint_probability"]
            q=new["joint_probability"]
            if (type(p) not in (float,int) or type(q) not in (float,int)
                    or not math.isfinite(p) or not math.isfinite(q)
                    or p<0 or abs(p-q)>1e-12):
                return False
    return True

def verify_packet(file, *, root, v28_root, clock):
    packet = json.loads(Path(file).read_text(encoding="utf-8"))
    if packet.get("version") != VERSION or packet.get("sydney_day") != Path(file).stem:
        raise ValueError("bad_frozen_packet_identity")
    original, items = verify_decision(
        Path(v28_root)/"decisions"/Path(file).name, root=root, clock=clock)
    issued = utc(packet["frozen_at_utc"])
    if issued > clock or not _frozen_eligible(original, issued):
        raise ValueError("late_or_future_frozen_challenger")
    reproduced = original_portfolio(original, items, issued)
    if not _same_portfolio_with_float_tolerance(reproduced,packet):
        raise ValueError("modified_original_forecast_or_challenger_tickets")
    return packet


def _five_hit(picks, actual):
    return any([p["score1"], p["score2"]] == actual for p in picks)


def _paired_exact_sign_test(only_candidate, only_baseline):
    """Day is one observation, not each of five tickets or overlapping pairs."""
    n = only_candidate + only_baseline
    if not n:
        return None
    return min(1.0, 2 * fsum(comb(n, k) for k in range(min(only_candidate, only_baseline) + 1)) / (2**n))


def summarize(rows, *, now):
    finished = [r for r in rows if r["status"] == "verified"]
    policy_stats = {}
    n = len(finished)
    original_fixture_keys = {}
    for row in finished:
        for i, key in enumerate(row["original_selected_fixture_keys"]):
            signature = tuple(key)
            value = tuple(row["actual_scores"][i].split("-"))
            if signature in original_fixture_keys and original_fixture_keys[signature] != value:
                raise ValueError("conflicting_verified_same_physical_fixture")
            original_fixture_keys[signature] = value
    for name in METHODS:
        hits = sum(int(r["hits"][name]) for r in finished)
        baseline_hits = sum(int(r["hits"]["v8_original"]) for r in finished)
        only_new = sum(r["hits"][name] and not r["hits"]["v8_original"] for r in finished)
        only_old = sum(r["hits"]["v8_original"] and not r["hits"][name] for r in finished)
        seen = set()
        legs = {"exact_score_log_loss": [], "multiclass_score_brier": [],
                "over_3_5_brier": [], "under_1_5_brier": []}
        for row in finished:
            for i, key in enumerate(row["original_selected_fixture_keys"]):
                if tuple(key) in seen:
                    continue
                seen.add(tuple(key))
                for metric in legs:
                    legs[metric].append(row["proper_scores"][name][i][metric])
        avg = {m: fsum(v)/len(v) if v else None for m,v in legs.items()}
        lower = wilson(hits, n)[0] if n else None
        exact_p = _paired_exact_sign_test(only_new, only_old) if n >= MIN_REVIEW_DAYS else None
        # Values below are descriptive; a true causal promotion needs prospective
        # independent matched wins and score calibration on the same fixtures.
        candidate_ready = (
            name != "v8_original" and
            n >= MIN_REVIEW_DAYS and len(seen) >= MIN_REVIEW_UNIQUE_FIXTURES and
            lower is not None and lower >= MIN_WILSON_LOWER_HIT_RATE and
            hits > baseline_hits and only_new > only_old and
            exact_p is not None and exact_p < 0.05
        )
        policy_stats[name] = {
            "verified_same_pair_days": n,
            "hits": hits,
            "observed_hit_rate": hits/n if n else None,
            "wilson95": wilson(hits, n),
            "candidate_only_wins_vs_v8": only_new,
            "v8_only_wins": only_old,
            "same_day_exact_sign_p_if_review_eligible": exact_p,
            "unique_verified_physical_fixtures": len(seen),
            "mean_proper_scores": avg,
            "review_gate_passed_on_hit_rate_only": candidate_ready,
            "not_auto_promoted": True,
        }
    for name in METHODS:
        if name == "v8_original":
            continue
        if policy_stats[name]["review_gate_passed_on_hit_rate_only"]:
            baseline_scores = policy_stats["v8_original"]["mean_proper_scores"]
            candidate_scores = policy_stats[name]["mean_proper_scores"]
            if (candidate_scores["exact_score_log_loss"] is None or
                    candidate_scores["exact_score_log_loss"] >= baseline_scores["exact_score_log_loss"] or
                    candidate_scores["multiclass_score_brier"] >= baseline_scores["multiclass_score_brier"]):
                policy_stats[name]["review_gate_passed_on_hit_rate_only"] = False
    return {
        "version": VERSION,
        "evaluated_at_utc": utc(now).isoformat(),
        "frozen_prospective_days": len(rows),
        "verified_same_pair_days": n,
        "pending_same_pair_days": len(rows) - n,
        "unique_verifiable_match_count": len(original_fixture_keys),
        "initial_review_verified_day_gate": MIN_REVIEW_DAYS,
        "methods": policy_stats,
        "days": rows,
        "ready_for_separate_production_review": [x for x in METHODS
            if policy_stats[x]["review_gate_passed_on_hit_rate_only"]],
        "research_only_methods_not_live_promoted": True,
        "independent_same_fixture_comparisons_only": True,
        "late_and_missing_results_remain_pending": True,
        "70_percent_hit_rate_not_claimed_without_verified_confidence_gate": True,
    }


def evaluate(*, root=ORIGINAL_ROOT, prematch=Path("data/prematch"),
             v28_root=V28_ROOT, out=ROOT, now=None):
    clock = utc(now or datetime.now(timezone.utc))
    audit = audit_original_full_slates(
        root=root, prematch=prematch, v28_out=v28_root, now=clock)
    by_day = {x["sydney_day"]:x for x in audit["days"] if x["original_policy"] == "FROZEN"}
    rows, invalid = [], Counter()
    for file in sorted((Path(out)/"sets").glob("*.json")):
        try:
            packet = verify_packet(file, root=root, v28_root=v28_root, clock=clock)
            day = by_day.get(packet["sydney_day"])
            if day is None:
                raise ValueError("original_day_not_in_v30_verified_catalog")
            matches = [x for x in day["candidate_rows"]
                       if x["fixture_keys"] == packet["original_selected_fixture_keys"]]
            if len(matches) > 1:
                raise ValueError("duplicate_original_candidate_row")
            row = {
                "sydney_day": packet["sydney_day"],
                "original_selected_fixture_keys": packet["original_selected_fixture_keys"],
                "frozen_at_utc": packet["frozen_at_utc"],
                "status": "pending",
            }
            if matches:
                actual = matches[0]["actual_scores"]
                goals = [tuple(map(int, score.split("-"))) for score in actual]
                proper = {name: [describe_score(grid, goal)
                        for grid, goal in zip(packet["policies"][name]["score_grids"], goals)]
                    for name in METHODS}
                row.update({
                    "status": "verified", "actual_scores": actual,
                    "hits": {name:_five_hit(packet["policies"][name]["five_joint_scores"],actual)
                             for name in METHODS},
                    "proper_scores": proper,
                })
            rows.append(row)
        except (OSError, TypeError, ValueError, KeyError, IndexError, OverflowError) as exc:
            invalid[str(exc) if isinstance(exc, ValueError) else type(exc).__name__] += 1
    report = summarize(rows, now=clock)
    report["original_v28_rejected_or_tampered"] = audit["invalid_or_tampered_frozen_days_excluded"]
    report["verified_result_conflicts_excluded"] = audit["verified_result_conflicts_excluded"]
    report["invalid_or_altered_v35_sets_excluded"] = dict(invalid)
    if invalid:
        report["ready_for_separate_production_review"] = []
    return report


def run(*, root=ORIGINAL_ROOT, prematch=Path("data/prematch"),
        v28_root=V28_ROOT, out=ROOT, now=None, verify_only=False):
    capture = {"created_days": [], "skipped": {"verify_only": 1}}
    if not verify_only:
        capture = freeze(root=root, v28_root=v28_root, out=out, now=now)
    report = evaluate(root=root, prematch=prematch, v28_root=v28_root, out=out, now=now)
    report["capture"] = capture
    destination = Path(out)/"status.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if not destination.exists() or destination.read_text(encoding="utf-8") != content:
        destination.write_text(content, encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    result = run(verify_only=args.verify_only)
    print(json.dumps({
        "frozen": result["frozen_prospective_days"],
        "verified": result["verified_same_pair_days"],
        "pending": result["pending_same_pair_days"],
        "methods": list(METHODS),
        "promotion_ready": result["ready_for_separate_production_review"],
    }, sort_keys=True))
