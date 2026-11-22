"""V32 read-only, *day-level* V30 rank discrimination research.

Only independently verified, complete V28-original slates can enter this audit.
Comparisons use the same exact-score five-ticket policy, equal numbers of
nonoverlapping top and bottom pairs per day, and ranks frozen before kickoff.
Overlapping candidate pairs are NOT independent experiments. Nothing here
creates score predictions or changes the live model.
"""
from collections import Counter
from datetime import datetime, timezone
from math import comb, fsum
import json
from pathlib import Path

from v30_full_slate_audit import analyze as audit_original_full_slates

OUT = Path("data/v32_rank_discrimination/status.json")
VERSION = "v32-original-full-slate-ranked-lift-v1-research-only"
CHECKPOINTS = (1, 5, 10)
POLICIES = {"quality": "quality_rank", "robustness": "robustness_rank"}
MIN_COMPLETE_DAYS = 30


def _auc(rows, rank_key):
    """Descriptive within-day concordance; NEVER a binomial/iid test."""
    hits = [int(r[rank_key]) for r in rows if r["robust_five_hit"]]
    misses = [int(r[rank_key]) for r in rows if not r["robust_five_hit"]]
    if not hits or not misses:
        return None
    concordant = sum(1 if a < b else 0.5 if a == b else 0
                     for a in hits for b in misses)
    return concordant / (len(hits) * len(misses))


def _contrast(rows, rank_key, requested):
    """Disjoint, equal-ticket top and bottom sets from ONE complete day."""
    count = min(requested, len(rows) // 2)
    if count < 1:
        return None
    ranked = sorted(rows, key=lambda r: (int(r[rank_key]), r["fixture_keys"]))
    top, bottom = ranked[:count], ranked[-count:]
    assert not {tuple(tuple(k) for k in r["fixture_keys"]) for r in top} & {
        tuple(tuple(k) for k in r["fixture_keys"]) for r in bottom
    }, "top/bottom comparison must contain distinct original candidate pairs"
    top_hits = sum(bool(r["robust_five_hit"]) for r in top)
    bottom_hits = sum(bool(r["robust_five_hit"]) for r in bottom)
    return {
        "requested_pairs_per_arm": requested,
        "actual_pairs_per_arm": count,
        "tickets_per_arm": 5 * count,
        "top_hits": top_hits,
        "bottom_hits": bottom_hits,
        "top_hit_pair_fraction_correlated": top_hits / count,
        "bottom_hit_pair_fraction_correlated": bottom_hits / count,
        "top_minus_bottom_pair_fraction_descriptive": (top_hits - bottom_hits) / count,
        "top_any_hit": top_hits > 0,
        "bottom_any_hit": bottom_hits > 0,
        "different_underlying_fixtures_not_guaranteed": True,
    }


def _sign_test(a, b):
    """Exact sign test across *days*, excluding tied day-level contrasts."""
    n = a + b
    if not n:
        return None
    return min(1.0, 2 * fsum(comb(n, j) for j in range(min(a, b) + 1)) / 2**n)


def summarize(v30):
    """No partial slates; a full Sydney day is one comparison unit."""
    if not v30["overlapping_pairs_are_not_independent_samples"]:
        raise ValueError("V30 independence guardrail missing")
    complete = []
    seen_days = set()
    for day in v30["days"]:
        if not day.get("complete_slate") or day.get("original_policy") != "FROZEN":
            continue
        day_id = day["sydney_day"]
        if day_id in seen_days:
            raise ValueError("duplicate_complete_sydney_day")
        seen_days.add(day_id)
        rows = day["candidate_rows"]
        if (day.get("candidate_pairs_pending") != 0 or
                len(rows) != day["candidate_count"] or
                len(rows) < 2 or
                not all("robust_five_hit" in r for r in rows)):
            raise ValueError("inconsistent_complete_original_slate")
        policies = {}
        for name, rank_key in POLICIES.items():
            ranks = [int(row[rank_key]) for row in rows]
            if sorted(ranks) != list(range(1, len(rows) + 1)):
                raise ValueError("invalid_original_rank_permutation")
            policies[name] = {
                "within_day_hit_vs_miss_rank_concordance": _auc(rows, rank_key),
                "equal_ticket_top_vs_bottom": {
                    str(k): _contrast(rows, rank_key, k) for k in CHECKPOINTS
                },
            }
        complete.append({
            "sydney_day": day_id,
            "original_candidate_catalog_sha256": day["original_candidate_catalog_sha256"],
            "frozen_candidate_pairs": len(rows),
            "original_fixtures_verified": day["unique_fixtures_verified"],
            "policies": policies,
        })
    aggregates = {}
    for name in POLICIES:
        for k in CHECKPOINTS:
            observed = [
                d["policies"][name]["equal_ticket_top_vs_bottom"][str(k)]
                for d in complete
            ]
            observed = [x for x in observed if x]
            diffs = [x["top_minus_bottom_pair_fraction_descriptive"] for x in observed]
            above, below = sum(x > 0 for x in diffs), sum(x < 0 for x in diffs)
            aggregates[f"{name}_top_bottom_{k}"] = {
                "complete_day_units": len(diffs),
                "days_top_outperformed_bottom": above,
                "days_bottom_outperformed_top": below,
                "days_tied": len(diffs) - above - below,
                "unweighted_mean_day_level_pair_fraction_difference":
                    fsum(diffs) / len(diffs) if diffs else None,
                "paired_day_sign_test_two_sided_exploratory":
                    _sign_test(above, below) if len(diffs) >= MIN_COMPLETE_DAYS else None,
                "statistical_review_ready": len(diffs) >= MIN_COMPLETE_DAYS,
                "not_an_independent_pair_win_rate": True,
            }
    return {
        "version": VERSION,
        "source": "reverified_original_V28_sources_via_V30_full_slate_audit",
        "original_frozen_slate_days": v30["original_frozen_days"],
        "completed_independently_verified_days": len(complete),
        "pending_or_incomplete_days": v30["original_frozen_days"] - len(complete),
        "original_candidate_pair_rows": v30["frozen_candidate_pair_rows"],
        "source_invalid_original_days_excluded":
            v30["invalid_or_tampered_frozen_days_excluded"],
        "source_invalid_verified_finals": v30["invalid_independent_final_sources"],
        "initial_review_complete_day_target": MIN_COMPLETE_DAYS,
        "initial_review_ready": len(complete) >= MIN_COMPLETE_DAYS,
        "days": complete,
        "within_day_rank_lift": aggregates,
        "rank_concordance_is_descriptive_not_iid_inference": True,
        "top_vs_bottom_ticket_budgets_equal_per_day": True,
        "missing_results_are_pending_not_losses": True,
        "no_hindsight_selections_or_new_predictions": True,
        "live_model_unchanged": True,
        "seventy_percent_five_double_accuracy_not_proven": True,
    }


def run(*, out=OUT, now=None):
    v30 = audit_original_full_slates(now=now)
    report = summarize(v30)
    report["evaluated_at_utc"] = (now or datetime.now(timezone.utc)).isoformat()
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if not out.exists() or out.read_text(encoding="utf-8") != content:
        out.write_text(content, encoding="utf-8")
    return report


if __name__ == "__main__":
    report = run()
    print(json.dumps({
        "complete_day_units": report["completed_independently_verified_days"],
        "review_ready": report["initial_review_ready"],
        "live_model_changed": False,
        "seventy_percent_proven": False,
    }, sort_keys=True))
