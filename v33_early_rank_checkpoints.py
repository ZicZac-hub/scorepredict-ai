"""V33: evaluate pre-frozen rank checkpoints as soon as THEIR legs settle.

Every arm is selected from the ENTIRE original V28 candidate catalog, before
examining any outcome. Unlike the V32 complete-slate analysis, a checkpoint
can settle when just its chosen top/bottom legs have verified final results.
Days remain the unit; pairs share fixtures and are never independent trials.
All ranking and forecast data is immutable. Research only.
"""
from collections import Counter
from datetime import datetime, timezone
from math import fsum
import json
from pathlib import Path

from five_double_accuracy import utc
from v28_independent_pair_trial import ORIGINAL_ROOT, OUT as V28_OUT, verify_decision
from v30_full_slate_audit import analyze as audit_original_full_slates
from v32_rank_discrimination import CHECKPOINTS, MIN_COMPLETE_DAYS, POLICIES, _sign_test

OUT = Path("data/v33_early_rank_checkpoints/status.json")
VERSION = "v33-frozen-rank-checkpoints-early-verified-v1-research-only"


def _key(row):
    return tuple(tuple(str(v) for v in x) for x in row["fixture_keys"])


def _ranked(catalog, score):
    return sorted(catalog, key=lambda x: (-float(x[score]), x["fixture_keys"]))


def _checkpoint(catalog, verified, score, requested):
    """Determine both arms BEFORE reading wins, using the full frozen slate."""
    n = min(requested, len(catalog) // 2)
    if n == 0:
        return {"status": "ineligible_small_original_slate"}
    ranked = _ranked(catalog, score)
    top = [_key(row) for row in ranked[:n]]
    bottom = [_key(row) for row in ranked[-n:]]
    if len(set(top + bottom)) != 2 * n:
        raise ValueError("duplicate_or_overlapping_original_candidate_pair")
    still_pending_top = sum(k not in verified for k in top)
    still_pending_bottom = sum(k not in verified for k in bottom)
    base = {
        "status": "pending",
        "requested_pairs_per_arm": requested,
        "predeclared_pairs_per_arm": n,
        "tickets_per_arm": n * 5,
        "original_top_pairs": [list(map(list, k)) for k in top],
        "original_bottom_pairs": [list(map(list, k)) for k in bottom],
        "pending_top_pair_results": still_pending_top,
        "pending_bottom_pair_results": still_pending_bottom,
    }
    if still_pending_top or still_pending_bottom:
        # An incomplete arm must NEVER be graded as a losing arm.
        return base
    hits_top = sum(verified[k] for k in top)
    hits_bottom = sum(verified[k] for k in bottom)
    return {
        **base, "status": "fully_verified_original_checkpoint",
        "top_hit_pairs_correlated": hits_top,
        "bottom_hit_pairs_correlated": hits_bottom,
        "top_any_hit": hits_top > 0,
        "bottom_any_hit": hits_bottom > 0,
        "top_minus_bottom_pair_fraction_descriptive": (hits_top - hits_bottom) / n,
    }


def summarize(v30, catalogs):
    """Consumes only V30's source-reverified rows and original frozen catalogs."""
    if not v30["overlapping_pairs_are_not_independent_samples"]:
        raise ValueError("missing_correlated_candidate_guardrail")
    days = []
    if len({d["sydney_day"] for d in v30["days"]}) != len(v30["days"]):
        raise ValueError("duplicate_sydney_day_in_audit")
    for day in v30["days"]:
        if day["original_policy"] != "FROZEN":
            continue
        date = day["sydney_day"]
        catalog = catalogs.get(date)
        if not isinstance(catalog, list) or len(catalog) != day["candidate_count"]:
            raise ValueError("missing_or_incomplete_frozen_candidate_catalog")
        if day.get("candidate_pairs_verified", 0) != len(day["candidate_rows"]):
            raise ValueError("verified_pair_count_mismatch")
        if day.get("candidate_pairs_pending", 0) + len(day["candidate_rows"]) != len(catalog):
            raise ValueError("original_pair_count_mismatch")
        lookup = {_key(row): row for row in catalog}
        if len(lookup) != len(catalog):
            raise ValueError("duplicate_original_frozen_pairs")
        original_ranks = {
            policy: {_key(row): i for i, row in enumerate(_ranked(catalog, score), 1)}
            for policy, score in (
                ("quality", "quality_rank_score"),
                ("robustness", "robustness_rank_score"),
            )
        }
        verified = {}
        for row in day["candidate_rows"]:
            key = _key(row)
            if key not in lookup or key in verified or type(row.get("robust_five_hit")) is not bool:
                raise ValueError("unexpected_verified_candidate_row")
            if any(row.get(policy + "_rank") != original_ranks[policy][key]
                   for policy in POLICIES):
                raise ValueError("verified_row_not_at_original_rank")
            verified[key] = int(row["robust_five_hit"])
        checkpoints = {}
        for policy, score in (
            ("quality", "quality_rank_score"),
            ("robustness", "robustness_rank_score"),
        ):
            checkpoints[policy] = {
                str(k): _checkpoint(catalog, verified, score, k) for k in CHECKPOINTS
            }
        days.append({
            "sydney_day": date,
            "original_candidate_catalog_sha256": day["original_candidate_catalog_sha256"],
            "original_candidate_pairs": len(catalog),
            "verified_candidate_pair_results_correlated": len(verified),
            "full_slate_verified": day["complete_slate"],
            "checkpoints": checkpoints,
        })
    aggregates = {}
    for policy in POLICIES:
        for k in CHECKPOINTS:
            rows = [day["checkpoints"][policy][str(k)] for day in days]
            scored = [row for row in rows if row["status"] == "fully_verified_original_checkpoint"]
            diffs = [row["top_minus_bottom_pair_fraction_descriptive"] for row in scored]
            better, worse = sum(d > 0 for d in diffs), sum(d < 0 for d in diffs)
            aggregates[f"{policy}_top_bottom_{k}"] = {
                "verified_day_units": len(scored),
                "pending_day_units": sum(row["status"] == "pending" for row in rows),
                "days_top_outperformed_bottom": better,
                "days_bottom_outperformed_top": worse,
                "days_tied": len(scored) - better - worse,
                "unweighted_mean_day_level_pair_fraction_difference":
                    fsum(diffs) / len(diffs) if diffs else None,
                "paired_day_sign_test_two_sided_exploratory":
                    _sign_test(better, worse) if len(scored) >= MIN_COMPLETE_DAYS else None,
                "statistical_review_ready": len(scored) >= MIN_COMPLETE_DAYS,
                "overlapping_candidate_pairs_are_not_iid_samples": True,
            }
    return {
        "version": VERSION,
        "source": "original_v28_frozen_catalog_plus_v30_independently_verified_results",
        "original_frozen_days": v30["original_frozen_days"],
        "original_candidate_pair_rows": v30["frozen_candidate_pair_rows"],
        "fully_verified_daily_slates_v32_basis": v30["fully_verified_slate_days"],
        "initial_review_verified_day_target": MIN_COMPLETE_DAYS,
        "days": days,
        "original_rank_checkpoint_results": aggregates,
        "source_invalid_original_days_excluded": v30["invalid_or_tampered_frozen_days_excluded"],
        "source_invalid_finals": v30["invalid_independent_final_sources"],
        "all_checkpoint_arm_memberships_fixed_before_reading_results": True,
        "partial_checkpoint_never_scored_as_loss": True,
        "rank_lift_is_not_a_single_pair_live_policy_hit_rate": True,
        "research_only_no_new_five_double_predictions": True,
        "live_model_changed": False,
        "seventy_percent_five_double_hit_rate_not_proven": True,
    }


def run(*, out=OUT, now=None):
    clock = utc(now or datetime.now(timezone.utc))
    v30 = audit_original_full_slates(now=clock)
    catalogs = {}
    for row in v30["days"]:
        if row["original_policy"] != "FROZEN":
            continue
        date = row["sydney_day"]
        file = Path(V28_OUT) / "decisions" / (date + ".json")
        original, _ = verify_decision(file, root=ORIGINAL_ROOT, clock=clock)
        if (original.get("candidate_catalog_sha256") !=
                row["original_candidate_catalog_sha256"]):
            raise ValueError("original_catalog_digest_disagrees_with_v30")
        catalogs[date] = original["candidate_catalog"]
    report = summarize(v30, catalogs)
    report["evaluated_at_utc"] = clock.isoformat()
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if not out.exists() or out.read_text(encoding="utf-8") != content:
        out.write_text(content, encoding="utf-8")
    return report


if __name__ == "__main__":
    report = run()
    print(json.dumps({
        "frozen_days": report["original_frozen_days"],
        "verified_top_1_quality_days":
            report["original_rank_checkpoint_results"]["quality_top_bottom_1"]["verified_day_units"],
        "live_model_changed": False,
        "seventy_percent_proven": False,
    }, sort_keys=True))
