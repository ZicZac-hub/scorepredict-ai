"""V25 anchored-score challenger: true future-only, matched five-double trial.

The existing V25 robust lower envelope is uncalibrated and does not maximize
probability mass. This challenger fixes a 75% original V8 / 25% V25 sensitivity
mixture BEFORE outcomes; it competes with the original V25 robust five and
V8 top-five ON EXACTLY THE SAME TWO MATCHES and same Sydney day.

V28 supplies write-once source-backed full-slate pair selection. Never create
predictions for an old V28 day, even if its matches have not kicked off.
No source API calls, retrospective selections, changed V25 archive, or live
model promotion. Real improvement must be demonstrated prospectively.
"""
from collections import Counter
from datetime import datetime, timedelta, timezone
from math import fsum
import json
from pathlib import Path

import numpy as np

from five_double_accuracy import joint_geometry, utc, wilson
from professional_analyst_shadow_v25 import VARIANTS, MIN_LEAD
from v10_prospective import ROOT as ORIGINAL_ROOT
from v28_independent_pair_trial import OUT as V28_ROOT, _key, verify_decision
from v30_full_slate_audit import analyze as audit_original_full_slates
from v26_temporal_score import _write_once

VERSION = "v25-anchored-probability-75-25-future-only-shadow-v1"
OUT = Path("data/v25_anchored_challenger")
MAX_DECISION_AGE = timedelta(minutes=20)
BASELINE_WEIGHT = 0.75
SENSITIVITY_WEIGHT = 0.25
EVALUATION_MIN_PAIRS = 100


def _frozen_eligible(original, now):
    issued = utc(original["actually_issued_at_utc"])
    clock = utc(now)
    if not (issued <= clock <= issued + MAX_DECISION_AGE):
        return False
    return original.get("policy") == "FROZEN"


def _anchored_grid(item):
    """Fixed, nonfitted shrinkage to the actually frozen original V8 grid."""
    v8 = np.asarray(item["grid"], dtype=float)
    if (v8.ndim != 2 or not np.isfinite(v8).all() or
            np.any(v8 < 0) or not np.isclose(v8.sum(), 1.0, atol=1e-6)):
        raise ValueError("invalid_frozen_original_v8_grid")
    sensitivity = np.zeros_like(v8)
    if not all(name in item["alternatives"] for name in VARIANTS[1:]):
        raise ValueError("missing_source_based_sensitivity")
    for name in VARIANTS[1:]:
        alternative = np.asarray(item["alternatives"][name], dtype=float)
        if (alternative.ndim != 2 or
                any(alternative.shape[d] > v8.shape[d] for d in (0, 1)) or
                not np.isfinite(alternative).all() or np.any(alternative < 0) or
                not np.isclose(alternative.sum(), 1.0, atol=1e-6)):
            raise ValueError("invalid_sensitivity_grid")
        sensitivity[:alternative.shape[0], :alternative.shape[1]] += (
            alternative / len(VARIANTS[1:]))
    blend = BASELINE_WEIGHT*v8 + SENSITIVITY_WEIGHT*sensitivity
    blend /= float(blend.sum())
    return blend.tolist()


def _picks_hit(picks, scores):
    return any((p["score1"], p["score2"]) == tuple(scores) for p in picks)


def _portfolio(original, items, clock):
    """Original pair choice and all source grids reverified before scoring."""
    if original["policy"] != "FROZEN":
        raise ValueError("not_an_original_frozen_day")
    selected = original["quality_selection"]
    keys = selected["fixture_keys"]
    if len(keys) != 2 or len({tuple(key) for key in keys}) != 2:
        raise ValueError("invalid_original_selected_pair")
    lookup = {_key(x):x for x in items}
    if any(tuple(key) not in lookup for key in keys):
        raise ValueError("original_selected_source_missing")
    legs = [lookup[tuple(key)] for key in keys]
    if any(clock > x["kickoff"] - MIN_LEAD for x in legs):
        raise ValueError("not_at_least_45_minutes_before_both_kickoffs")
    if len(legs[0]["teams"] | legs[1]["teams"]) != 4:
        raise ValueError("not_four_distinct_teams")
    grids = [_anchored_grid(x) for x in legs]
    choices = joint_geometry(*grids)["optimal_five"]
    if len(choices) != 5 or len({(x["score1"], x["score2"]) for x in choices}) != 5:
        raise ValueError("invalid_anchored_five")
    return {
        "version": VERSION,
        "sydney_day": original["sydney_day"],
        "frozen_at_utc": utc(clock).isoformat(),
        "original_v28_decision_issued_at_utc": utc(original["actually_issued_at_utc"]).isoformat(),
        "original_v28_cutoff_utc": utc(original["policy_cutoff_utc"]).isoformat(),
        "original_candidate_catalog_sha256": original["candidate_catalog_sha256"],
        "original_selected_fixture_keys": keys,
        "original_source_forecast_ids": selected["source_forecast_ids"],
        "v8_anchor_weight": BASELINE_WEIGHT,
        "v25_sensitivity_weight": SENSITIVITY_WEIGHT,
        "frozen_anchored_score_grids": grids,
        "five_anchored_v25": choices,
        "original_v25_robust_five": selected["five_original_v25_robust_scores"],
        "original_v8_optimal_five": selected["five_original_v8_optimal_scores"],
        "same_original_pair_for_all_three_policies": True,
        "no_new_source_requests": True,
        "no_backfilled_five_doubles": True,
        "live_model_unchanged": True,
    }


def freeze(*, root=ORIGINAL_ROOT, v28_out=V28_ROOT, out=OUT, now=None):
    clock = utc(now or datetime.now(timezone.utc))
    created, excluded = [], Counter()
    for file in sorted((Path(v28_out)/"decisions").glob("*.json")):
        target = Path(out)/"sets"/file.name
        if target.exists():
            excluded["already_frozen"] += 1
            continue
        try:
            original, items = verify_decision(file, root=root, clock=clock)
            if not _frozen_eligible(original, clock):
                excluded["not_new_original_v28_decision"] += 1
                continue
            # Use actual wall clock after the expensive original source check;
            # never give a late generated prediction an earlier issue time.
            issued = clock if now is not None else utc(datetime.now(timezone.utc))
            if not _frozen_eligible(original, issued):
                excluded["generation_exceeded_20_minute_window"] += 1
                continue
            packet = _portfolio(original, items, issued)
            # A valid snapshot must be a truly NEW pre-kickoff action.
            if _write_once(target, packet):
                created.append(original["sydney_day"])
        except (OSError, ValueError, KeyError, TypeError, OverflowError, IndexError) as exc:
            excluded[str(exc) if isinstance(exc, ValueError) else type(exc).__name__] += 1
    return {"created": created, "skipped": dict(excluded)}


def _verify_frozen(file, *, root, v28_out, clock):
    packet = json.loads(file.read_text(encoding="utf-8"))
    day = file.stem
    if not isinstance(packet, dict) or packet.get("version") != VERSION or packet.get("sydney_day") != day:
        raise ValueError("invalid_anchored_archive_identity")
    original_file = Path(v28_out)/"decisions"/file.name
    original, items = verify_decision(original_file, root=root, clock=clock)
    frozen_at = utc(packet["frozen_at_utc"])
    if frozen_at > clock or not _frozen_eligible(original, frozen_at):
        raise ValueError("challenger_was_not_frozen_in_new_original_window")
    expected = _portfolio(original, items, frozen_at)
    if packet != expected:
        raise ValueError("altered_anchored_forecast_or_source_provenance")
    return packet


def evaluate(*, root=ORIGINAL_ROOT, prematch=Path("data/prematch"),
             v28_out=V28_ROOT, out=OUT, now=None):
    clock = utc(now or datetime.now(timezone.utc))
    audit = audit_original_full_slates(root=root, prematch=prematch,
                                      v28_out=v28_out, now=clock)
    days = {row["sydney_day"]: row for row in audit["days"] if row["original_policy"] == "FROZEN"}
    results, errors = [], Counter()
    for file in sorted((Path(out)/"sets").glob("*.json")):
        try:
            packet = _verify_frozen(file, root=root, v28_out=v28_out, clock=clock)
            day = days.get(packet["sydney_day"])
            if day is None:
                raise ValueError("original_verified_day_not_in_audit")
            matches = [r for r in day["candidate_rows"]
                       if r["fixture_keys"] == packet["original_selected_fixture_keys"]]
            if len(matches) > 1:
                raise ValueError("duplicate_verified_original_pair")
            row = {"sydney_day":packet["sydney_day"],
                   "status": "pending",
                   "same_pair_comparison": True,
                   "original_candidate_catalog_sha256":packet["original_candidate_catalog_sha256"]}
            if matches:
                actual = matches[0]["actual_scores"]
                row.update({
                    "status":"verified", "actual_scores": actual,
                    "v25_robust_hit": _picks_hit(packet["original_v25_robust_five"],actual),
                    "v8_optimal_hit": _picks_hit(packet["original_v8_optimal_five"],actual),
                    "anchored_v25_hit": _picks_hit(packet["five_anchored_v25"],actual),
                    "frozen_anchored_actual_joint_probability":
                        _joint_mass_at_actual(packet["frozen_anchored_score_grids"],actual),
                })
            results.append(row)
        except (OSError, ValueError, KeyError, TypeError, IndexError, OverflowError) as exc:
            errors[str(exc) if isinstance(exc, ValueError) else type(exc).__name__] += 1
    verified = [r for r in results if r["status"] == "verified"]
    counts = {policy:sum(bool(row[policy+"_hit"]) for row in verified)
              for policy in ("v8_optimal","v25_robust","anchored_v25")}
    anchor_only = sum(r["anchored_v25_hit"] and not r["v25_robust_hit"] for r in verified)
    original_only = sum(r["v25_robust_hit"] and not r["anchored_v25_hit"] for r in verified)
    n = len(verified)
    return {
        "version":VERSION, "evaluated_at_utc":clock.isoformat(),
        "frozen_prospective_same_pair_days":len(results),
        "verified_matched_pair_days":n,
        "pending_pair_days":len(results)-n,
        "invalid_or_tampered_challengers_excluded":dict(errors),
        "v30_original_invalid_days_excluded":audit["invalid_or_tampered_frozen_days_excluded"],
        "v30_invalid_verified_finals":audit["invalid_independent_final_sources"],
        "same_original_pair_matched_policy_wins":counts,
        "same_original_pair_matched_policy_hit_rates":
            {name:counts[name]/n if n else None for name in counts},
        "anchored_v25_wilson95":wilson(counts["anchored_v25"],n),
        "anchored_only_wins_over_robust":anchor_only,
        "original_robust_only_wins_over_anchor":original_only,
        "initial_review_verified_pairs_target":EVALUATION_MIN_PAIRS,
        "review_ready": n >= EVALUATION_MIN_PAIRS and not errors,
        "report_rows":results,
        "pending_is_not_a_miss":True,
        "same_fixture_pairs_are_required_for_every_comparison":True,
        "estimated_model_mass_is_not_observed_accuracy":True,
        "new_anchored_model_not_promoted_to_live":True,
        "seventy_percent_five_double_hit_rate_unproven":True,
    }


def _joint_mass_at_actual(grids, scores):
    coords = []
    for score in scores:
        parts = score.split("-")
        if len(parts) != 2:
            raise ValueError("invalid_verified_score")
        h,a=map(int,parts)
        if h < 0 or a < 0:
            raise ValueError("invalid_verified_score")
        coords.append((h,a))
    p=1.0
    for grid,(h,a) in zip(grids,coords):
        p *= grid[h][a] if h < len(grid) and a < len(grid[0]) else 0.
    return float(p)


def run(*,root=ORIGINAL_ROOT,prematch=Path("data/prematch"),
        v28_out=V28_ROOT,out=OUT,now=None,verify_only=False):
    frozen={"created":[],"skipped":{}} if verify_only else freeze(
        root=root,v28_out=v28_out,out=out,now=now)
    clock=utc(now or datetime.now(timezone.utc))
    report=evaluate(root=root,prematch=prematch,v28_out=v28_out,out=out,now=clock)
    report["capture"]=frozen
    destination=Path(out)/"status.json"
    destination.parent.mkdir(parents=True,exist_ok=True)
    content=json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+"\n"
    if not destination.exists() or destination.read_text(encoding="utf-8") != content:
        destination.write_text(content,encoding="utf-8")
    return report


if __name__ == "__main__":
    import argparse
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--verify-only", action="store_true",
                     help="CI read-only mode; never mint past picks")
    a=cli.parse_args()
    result=run(verify_only=a.verify_only)
    print(json.dumps({
        "created":result["capture"]["created"],
        "verified":result["verified_matched_pair_days"],
        "pending":result["pending_pair_days"],
        "v25_robust_wins":result["same_original_pair_matched_policy_wins"]["v25_robust"],
        "v25_anchored_wins":result["same_original_pair_matched_policy_wins"]["anchored_v25"],
        "production_model_promoted":False,
    },sort_keys=True))
