"""Read-only, matched replay of V43 against authenticated pre-match archives."""
from collections import Counter
from datetime import datetime, timezone
from functools import partial
import json
import math
from pathlib import Path

from five_double_accuracy import _final, _result_index
from professional_double_shadow import _validated as validate_original_pair
from v10_prospective import utc
import v42_late_research_trial as late
import v42_research_trial as early
from v43_prediction import combine_grids


def _load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _original_pair(record, *, source_root, now):
    models, choices, coverage = validate_original_pair(record, source_root=source_root, now=now)
    grids = [model["score_matrix"] for model in models]
    original = {"status": "issued", "full_score_grids": grids,
                "five_doubles": choices, "estimated_coverage": coverage}
    return models, [{}, {}], {"v8_original": original}


def _frozen_pair(protocol, root, record, *, pair_root, source_root, provider_root, now):
    models = protocol._validated(record, root=root, pair_root=pair_root,
                                 source_root=source_root, provider_root=provider_root, now=now)
    decision = utc(record["decision_as_of_utc"])
    bundles = [protocol._components(model, provider_root=provider_root, cutoff=decision)
               for model in models]
    arms = record["arms"]
    return models, bundles, {"v8_original": arms["baseline_v8"],
                             "v41_control": arms["v41_control"], "v42_frozen": arms["unified_final"]}


def _methods(names, settled):
    count = len(settled)
    methods = {}
    for name in names:
        rows = [result[name] for result in settled]
        hits = sum(row["hit"] for row in rows)
        methods[name] = {
            "verified_pairs": count,
            "hits": hits,
            "hit_rate": hits / count if count else None,
            "score_log_loss": math.fsum(row["score_log_loss"] for row in rows) / count if count else None,
            "outcome_brier": math.fsum(row["outcome_brier"] for row in rows) / count if count else None,
            "mean_model_coverage": math.fsum(row["estimated_coverage"] for row in rows) / count if count else None,
        }
    return methods


def _cohort(paths, load_pair, *, names, finals, context):
    settled, rejections = [], Counter()
    first_day, last_day = None, None
    seen_ids, seen_events = set(), set()
    pending, abstentions, xg_pairs, market_pairs = 0, 0, 0, 0
    for path in paths:
        try:
            record = _load(path)
            if path.stem != record["sydney_date"]:
                raise ValueError("study_filename_day_mismatch")
            models, bundles, arms = load_pair(record, **context)
            identities = {(model["provider"], str(model["fixture_id"])) for model in models}
            events = {(utc(model["kickoff_utc"]), " ".join(model["home_team"].casefold().split()),
                       " ".join(model["away_team"].casefold().split())) for model in models}
            if len(identities) != 2 or len(events) != 2 or identities & seen_ids or events & seen_events:
                raise ValueError("duplicate_physical_fixture")
        except (AttributeError, KeyError, OSError, OverflowError, TypeError, ValueError) as exc:
            rejections[str(exc) if isinstance(exc, ValueError) else type(exc).__name__] += 1
            continue
        seen_ids.update(identities)
        seen_events.update(events)
        combined = combine_grids([model["score_matrix"] for model in models], bundles)
        arms["v43_replay"] = combined
        if any(arm["status"] != "issued" for arm in arms.values()):
            abstentions += 1
            continue
        actual = [_final(model, finals.get((model["provider"], str(model["fixture_id"]))), context["now"])
                  for model in models]
        if any(score is None for score in actual):
            pending += 1
            continue
        settled.append({name: early._metric(arm, actual) for name, arm in arms.items()})
        evidence_used = combined["lineage"]["evidence_used"]
        xg_pairs += evidence_used["xg"]
        market_pairs += evidence_used["market"]
        first_day = first_day or path.stem
        last_day = path.stem
    return {"archived_pairs": len(paths), "verified_pairs": len(settled),
            "verified_fixtures": 2 * len(settled), "pending_pairs": pending,
            "invalid_pairs": sum(rejections.values()), "abstained_pairs": abstentions,
            "first_verified_sydney_date": first_day, "last_verified_sydney_date": last_day,
            "matched_pairs_with_paired_xg": xg_pairs,
            "matched_pairs_with_paired_market": market_pairs, "source_rejections": dict(rejections),
            "methods": _methods(names, settled)}


def evaluate(now=None, *, early_root=early.ROOT, late_root=late.ROOT, pair_root=early.PAIR_ROOT,
             source_root=early.SOURCE_ROOT, provider_root=early.PROVIDER_ROOT,
             prematch_root=early.PREMATCH_ROOT):
    """Recalculate from original archives without API requests or writes.

    Fixed V42 early and late cohorts share events and stay separate. The older
    pair archive provides a broader statistical sensitivity comparison, using
    only original grids saved alongside the original five combinations.
    """
    clock = utc(now or datetime.now(timezone.utc))
    finals, conflicts, result_rejections = _result_index(source_root, prematch_root)
    context = {"pair_root": pair_root, "source_root": source_root,
               "provider_root": provider_root, "now": clock}
    original = _cohort(sorted((Path(pair_root) / "sets").glob("*.json")), _original_pair,
                       names=("v8_original", "v43_replay"), finals=finals,
                       context={"source_root": source_root, "now": clock})
    cohorts = {"original_pair_archive": original}
    for name, protocol, root in (("v42_early", early, early_root), ("v42_late", late, late_root)):
        load_pair = partial(_frozen_pair, protocol, root)
        cohorts[name] = _cohort(sorted((Path(root) / "trials").glob("*.json")), load_pair,
                                names=("v8_original", "v41_control", "v42_frozen", "v43_replay"),
                                finals=finals, context=context)
    return {
        "version": "v43-matched-replay-v1", "evaluated_at_utc": clock.isoformat(),
        "study_type": "retrospective_matched_replay", "cohorts": cohorts,
        "conflicting_result_identities": len(conflicts), "result_rejections": dict(result_rejections),
        "external_requests": 0, "prospective_v43_evidence": False, "model_promoted": False,
        "limitations": [
            "Each cohort compares the same matches and verified regulation-time results.",
            "Cohorts overlap. Their counts describe separate comparisons and cannot be added.",
            "The original pair cohort uses the frozen statistical grids alone. V42 cohorts use qualifying evidence at their original cutoffs.",
            "V43 was calculated after archived results. Five-double hits are descriptive and lower values are better for both proper scores.",
        ],
    }


if __name__ == "__main__":
    print(json.dumps(evaluate(), indent=2, allow_nan=False))
