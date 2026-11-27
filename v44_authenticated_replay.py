"""Read-only V44 matched historical replay, not a prospective accuracy claim.

All score-shape hypotheses here derive from the same archived V8 grid.
They are not independent information sources. Original V8 selections and
V43 replay are controls, never overwritten. No external API calls.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
from pathlib import Path

from five_double_accuracy import _final, _result_index, wilson
from professional_double_shadow import _validated
from v10_prospective import utc
import v42_research_trial as sources
from v41_unified_ensemble import V35_METHODS, V36_METHODS
from v35_continuous_double_trials import score_grid
from v36_tail_calibration_research import transform_score_grid as tail
from v38_margin_calibration import transform_score_grid as margin
from v43_prediction import combine_grids
from v44_family_pool_research import MIN_TRAIN_PAIRS, walk_forward


def _groups(grids):
    return {
        "anchor": {"v8": grids},
        "shape": {name: [score_grid({"grid": g}, name) for g in grids]
                  for name in V35_METHODS},
        "tail": {name: [tail(g, name) for g in grids] for name in V36_METHODS},
        "margin": {"conditional_margin_125": [margin(g, "conditional_margin_125") for g in grids]},
    }


def _event(model):
    return json.dumps([utc(model["kickoff_utc"]).isoformat(),
                       *(" ".join(model[key].casefold().split())
                         for key in ("home_team", "away_team"))],
                      separators=(",", ":"))


def _summarize(metrics):
    n = len(metrics)
    hits = sum(m["hit"] for m in metrics)
    return {"pairs": n, "hits": hits, "hit_rate": hits / n if n else None,
            "wilson_95": wilson(hits, n),
            "mean_log_loss": math.fsum(m["score_log_loss"] for m in metrics) / n if n else None}


def evaluate(*, pair_root=sources.PAIR_ROOT, source_root=sources.SOURCE_ROOT,
             prematch_root=sources.PREMATCH_ROOT, now=None):
    clock = utc(now or datetime.now(timezone.utc))
    finals, conflicts, result_rejections = _result_index(source_root, prematch_root)
    rejections, valid, pending = Counter(), [], 0
    paths = sorted((Path(pair_root) / "sets").glob("*.json"))
    for path in paths:
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
            models, choices, coverage = _validated(record, source_root=source_root, now=clock)
            if path.stem != record["sydney_date"]:
                raise ValueError("filename_day_mismatch")
            decision = utc(record["predicted_at_utc"])
            valid.append((decision, path.name, models, choices, coverage))
        except (ValueError, TypeError, KeyError, OSError, AttributeError, ArithmeticError) as exc:
            rejections[str(exc) if isinstance(exc, ValueError) else type(exc).__name__] += 1

    rows, controls, seen, seen_ids = [], {}, set(), set()
    for decision, _, models, choices, coverage in sorted(valid, key=lambda x: x[:2]):
        try:
            ids = tuple(_event(m) for m in models)
            provider_ids = {(str(m["provider"]), str(m["fixture_id"])) for m in models}
            if (len(set(ids)) != 2 or len(provider_ids) != 2
                    or seen.intersection(ids) or seen_ids.intersection(provider_ids)):
                raise ValueError("repeated_fixture")
            seen.update(ids)
            seen_ids.update(provider_ids)
            results = [finals.get((str(m["provider"]), str(m["fixture_id"]))) for m in models]
            actual = [_final(m, r, clock) for m, r in zip(models, results)]
            if any(s is None for s in actual):
                pending += 1
                continue
            grids = [m["score_matrix"] for m in models]
            v43 = combine_grids(grids, [{}, {}])
            if v43["status"] != "issued":
                raise ValueError("v43_control_abstained")
            v8 = {"status": "issued", "full_score_grids": grids, "five_doubles": choices,
                  "estimated_coverage": coverage}
            groups = _groups(grids)
            # Validate every grid before adding the pair to the common cohort.
            from v44_family_pool_research import family_pool
            family_pool(groups)
            controls[ids] = {"v8": sources._metric(v8, actual),
                             "v43_replay": sources._metric(v43, actual)}
            rows.append({"decision_utc": decision.isoformat(),
                         "earliest_kickoff_utc": min(utc(m["kickoff_utc"]) for m in models).isoformat(),
                         "finals_known_utc": max(utc(r["verified_at_utc"]) for r in results).isoformat(),
                         "fixture_ids": list(ids), "verified": True,
                         "actual": actual, "families": groups})
        except (ValueError, TypeError, KeyError, OSError, AttributeError, ArithmeticError) as exc:
            rejections[str(exc) if isinstance(exc, ValueError) else type(exc).__name__] += 1

    report = walk_forward(rows)
    comparable = [r for r in report["rows"] if r["training_pairs"] >= MIN_TRAIN_PAIRS]
    comparisons = {name: _summarize([r[name] for r in comparable]) for name in ("learned", "equal")}
    for name in ("v8", "v43_replay"):
        comparisons[name] = _summarize([controls[tuple(r["fixture_ids"])][name] for r in comparable])
    return {"study_type": "retrospective_matched_walk_forward",
            "evaluated_at_utc": clock.isoformat(),
            "archived_pairs": len(paths), "verified_common_pairs": len(rows),
            "pending_pairs": pending, "source_rejections": dict(rejections),
            "conflicting_result_identities": len(conflicts),
            "result_rejections": dict(result_rejections),
            "warmup_pairs_required": MIN_TRAIN_PAIRS,
            "eligible_comparison_pairs": len(comparable),
            "comparisons": comparisons,
            "rows": report["rows"],
            "external_requests": 0, "prospective_v44_evidence": False,
            "automatic_promotion": False,
            "limitations": [
                "All hypotheses here are correlated V8-derived score transforms.",
                "Original pair archive only; no verified lineup, xG or market adjustment is introduced.",
                "Later verified finals cannot train earlier decisions.",
                "Retrospective method design may still be influenced by earlier research results.",
                "No claim of improved accuracy without prospective evaluation."
            ]}


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    print(json.dumps(evaluate(), indent=2, allow_nan=False))
