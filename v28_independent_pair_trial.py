"""V28 independent prospective match-pair trial, with an equal-cutoff control.

Unlike V26/V27, V28 does NOT depend on a NEW V25 frozen pair. A single
pre-kickoff decision saves BOTH the quality policy and a V25-style robustness
control, from the same complete eligible slate and the same score-picking
method, before either match kicks off. Never backfill predictions after results.

All score probabilities originate in immutable V8 source files. Four model
sensitivities are transforms of those same goal estimates, NOT independent
model evidence. Neither rank is a calibrated win probability. No live promotion.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from itertools import combinations
import hashlib
import json
import math
from pathlib import Path

from five_double_accuracy import _final, utc, wilson
from professional_analyst_shadow_v25 import (
    _eligible, _pair, MIN_LEAD, MAX_LEAD, SYDNEY,
)
from v10_prospective import ROOT as ORIGINAL_ROOT, read_latest
from v8_v25_matched_audit import _source
from v27_pair_selection import (
    MAX_EVALUATED_PAIRS, MIN_V8_COVERAGE, _freshness_multiplier,
    canonical, digest, receipt,
)
from v26_temporal_score import _write_once
from verified_result_index import index_verified_results

VERSION = "v28-independent-paired-pair-policy-v1-research-only"
OUT = Path("data/v28_independent_pair_trial")
PREMATCH = Path("data/prematch")
MAX_EVALUATION_AGE = timedelta(minutes=20)
MIN_VALID_PAIRS = 2
MIN_PROSPECTIVE_REVIEW_DAYS = 100


def _key(item):
    return tuple(str(v) for v in item["key"])


def _fixture_pairs(a, b):
    return [[_key(a)[0], _key(a)[1]], [_key(b)[0], _key(b)[1]]]


def eligible_slate(source_rows, cutoff, *, day=None):
    """Take at most one pre-kickoff source per fixture, no result access."""
    clock = utc(cutoff)
    chosen = {}
    excluded = Counter()
    rows = source_rows.values() if isinstance(source_rows, dict) else source_rows
    for source in rows:
        item, why = _eligible(source, clock)
        if item is None:
            excluded[why] += 1
            continue
        if day is not None and item["day"] != day:
            continue
        old = chosen.get(_key(item))
        if old is None or item["captured"] > old["captured"]:
            chosen[_key(item)] = item
    groups = defaultdict(list)
    for key, item in sorted(chosen.items()):
        groups[item["day"]].append(item)
    return dict(groups), dict(excluded)


def candidate(a, b):
    """Identical five-score choice policy in both arms: original V25 robust."""
    ranked = _pair(a, b)
    v8 = float(ranked["v8_optimal_five_model_mass"])
    robust_v8 = float(ranked["robust_five_v8_model_mass"])
    freshness = float(_freshness_multiplier(a, b))
    quality = (0.60*v8 + 0.40*robust_v8)*freshness
    return {
        "fixture_keys": _fixture_pairs(a, b),
        "source_forecast_ids": [
            str(a["record"]["forecast_id"]), str(b["record"]["forecast_id"])],
        "quality_rank_score": round(quality, 12),
        "robustness_rank_score": round(
            float(ranked["robust_lower_envelope_model_mass"]), 12),
        "v8_optimal_model_mass": round(v8, 12),
        "robust_five_under_original_v8_mass": round(robust_v8, 12),
        "source_freshness_multiplier": round(freshness, 12),
        "five_original_v25_robust_scores": ranked["five_doubles"],
        "five_original_v8_optimal_scores": ranked["baseline_five"],
        "robustness_is_not_calibrated_probability": True,
    }


def plan(items, *, max_pairs=MAX_EVALUATED_PAIRS):
    """Full slate of eligible four-distinct-team pairings; no top-N censoring."""
    if type(max_pairs) is not int or max_pairs < 1:
        raise ValueError("invalid_pair_comparison_budget")
    sorted_items = sorted(items, key=_key)
    source_receipts = [receipt(row) for row in sorted_items]
    pairs = [(a,b) for a,b in combinations(sorted_items, 2)
             if len(a["teams"] | b["teams"]) == 4]
    base = {
        "eligible_fixture_count": len(sorted_items),
        "eligible_four_team_pair_count": len(pairs),
        "full_slate_source_receipts": source_receipts,
        "source_receipts_sha256": digest(source_receipts),
        "all_eligible_pair_candidates_exhaustively_checked": len(pairs) <= max_pairs,
    }
    if len(pairs) > max_pairs:
        return {**base, "policy": "ABSTAIN",
                "reason": "full_pair_slate_exceeds_budget",
                "candidate_catalog": [], "candidate_catalog_sha256": digest([]),
                "computed_pairs": 0}
    catalog = [candidate(a,b) for a,b in pairs]
    catalog.sort(key=lambda x:x["fixture_keys"])
    base.update({
        "candidate_catalog": catalog,
        "candidate_catalog_sha256": digest(catalog),
        "computed_pairs": len(catalog),
    })
    if len(catalog) < MIN_VALID_PAIRS:
        return {**base, "policy": "ABSTAIN",
                "reason": "fewer_than_two_valid_distinct_pair_options"}
    quality = sorted(catalog, key=lambda x: (
        -x["quality_rank_score"], x["fixture_keys"]))[0]
    robust = sorted(catalog, key=lambda x: (
        -x["robustness_rank_score"], x["fixture_keys"]))[0]
    if quality["v8_optimal_model_mass"] < MIN_V8_COVERAGE:
        return {**base, "policy": "ABSTAIN",
                "reason": "best_quality_pair_below_prespecified_mass_gate"}
    return {
        **base, "policy": "FROZEN",
        "quality_selection": quality,
        "robustness_control": robust,
        "arms_use_identical_v25_robust_score_choice_policy": True,
        "different_pair_selection_policies_not_different_score_models": True,
    }


def _late_arm(source_map, choice, issued):
    return any(
        issued > source_map[tuple(key)]["kickoff"] - MIN_LEAD
        for key in choice["fixture_keys"]
    )


def freeze(*, root=ORIGINAL_ROOT, out=OUT, now=None, originals=None):
    """Make NEW shadow decisions independently for each eligible Sydney day.

    A day without enough original fixtures is NOT frozen; it can qualify at
    a later genuine pre-kickoff time when further authenticated data arrives.
    A valid complete choice or an explicit policy abstention is write-once.
    """
    cutoff = utc(now or datetime.now(timezone.utc))
    records = read_latest(root) if originals is None else originals
    groups, excluded = eligible_slate(records, cutoff)
    created, skipped, inspected = [], Counter(), []
    for day, items in sorted(groups.items()):
        file = Path(out)/"decisions"/(day+".json")
        if file.exists():
            skipped["already_frozen_day"] += 1
            continue
        if len(items) < 3:
            skipped["insufficient_eligible_fixtures_retry_later"] += 1
            inspected.append({"sydney_day":day,"eligible_fixtures":len(items),
                              "state":"not_frozen_retry_when_new_sources_available"})
            continue
        evaluated = plan(items)
        if evaluated["eligible_four_team_pair_count"] < MIN_VALID_PAIRS:
            skipped["insufficient_four_team_pairs_retry_later"] += 1
            inspected.append({"sydney_day":day,"eligible_fixtures":len(items),
                              "state":"not_frozen_retry_when_new_sources_available"})
            continue
        # Production uses wall clock at actual issuance, not the earlier
        # start of a long/expensive full-slate ranking. Explicit simulation
        # tests pass now= for a deterministic synthetic decision time.
        issued = (cutoff if now is not None
                  else utc(datetime.now(timezone.utc)))
        if not cutoff <= issued <= cutoff + MAX_EVALUATION_AGE:
            skipped["slow_selection_generation_exceeded_20_minutes"] += 1
            continue
        if evaluated["policy"] == "FROZEN":
            lookup = {_key(item):item for item in items}
            if (_late_arm(lookup, evaluated["quality_selection"], issued) or
                    _late_arm(lookup, evaluated["robustness_control"], issued)):
                evaluated = {
                    **{k:v for k,v in evaluated.items() if k not in (
                        "quality_selection", "robustness_control",
                        "arms_use_identical_v25_robust_score_choice_policy",
                        "different_pair_selection_policies_not_different_score_models")},
                    "policy": "ABSTAIN",
                    "reason": "selected_pair_too_close_to_kickoff_at_actual_issue",
                }
        payload = {
            "schema_version": 1, "version": VERSION,
            "sydney_day": day,
            "policy_cutoff_utc": cutoff.isoformat(),
            "actually_issued_at_utc": issued.isoformat(),
            "source_exclusion_counts_at_decision": excluded,
            "ranking_criteria": {
                "quality": "0.60*v8_model_top5_mass + 0.40*robust_top5_under_v8_mass, times bounded source freshness",
                "control": "highest v25_robust_lower_envelope_mass",
                "score_picks_both_arms": "identical v25 robust five joint exact-score choices",
                "selection_uses_no_verified_match_results": True,
                "model_mass_is_not_win_probability": True,
            },
            **evaluated,
            "production_predictions_modified": False,
            "research_only_not_promoted": True,
            "at_least_70_percent_accuracy_proven": False,
        }
        if _write_once(file, payload):
            created.append({"day":day,"policy":payload["policy"],
                            "eligible_pairs":evaluated["eligible_four_team_pair_count"]})
    return {
        "created": created, "skipped": dict(skipped),
        "not_frozen_inspected": inspected,
        "source_ineligibility": excluded, "version": VERSION,
        "no_live_model_change": True,
    }


def verify_decision(file, *, root, clock):
    """Audit source identities and re-run every original candidate ranking."""
    payload = json.loads(file.read_text(encoding="utf-8"))
    day = payload.get("sydney_day")
    if (payload.get("schema_version") != 1 or
            payload.get("version") != VERSION or day != file.stem or
            payload.get("policy") not in ("FROZEN","ABSTAIN")):
        raise ValueError("invalid_frozen_decision_or_date")
    cutoff, issued = (utc(payload[k]) for k in (
        "policy_cutoff_utc","actually_issued_at_utc"))
    if not cutoff <= issued <= cutoff + MAX_EVALUATION_AGE or issued > clock:
        raise ValueError("invalid_or_future_issue_time")
    receipts = payload.get("full_slate_source_receipts")
    if (not isinstance(receipts, list) or
            len(receipts) != payload.get("eligible_fixture_count") or
            digest(receipts) != payload.get("source_receipts_sha256")):
        raise ValueError("changed_original_source_receipts")
    items = []
    identity = set()
    for rec in receipts:
        original, _ = _source(root, rec, cutoff, day)
        item, reason = _eligible(original, cutoff)
        if item is None or receipt(item) != rec or _key(item) in identity:
            raise ValueError("unreproducible_original_forecast_or_duplicate")
        identity.add(_key(item))
        items.append(item)
    expected = plan(items)
    # Independently reproduce original full slate, all ranks and selections.
    # If the frozen arm would have missed lead time at real issue, an
    # abstention is required, never a backdated valid prediction.
    if expected["policy"] == "FROZEN":
        lookup = {_key(x):x for x in items}
        if (_late_arm(lookup, expected["quality_selection"], issued) or
                _late_arm(lookup, expected["robustness_control"], issued)):
            expected = {
                **{k:v for k,v in expected.items() if k not in (
                    "quality_selection", "robustness_control",
                    "arms_use_identical_v25_robust_score_choice_policy",
                    "different_pair_selection_policies_not_different_score_models")},
                "policy":"ABSTAIN",
                "reason":"selected_pair_too_close_to_kickoff_at_actual_issue",
            }
    for k in (
        "eligible_fixture_count","eligible_four_team_pair_count",
        "all_eligible_pair_candidates_exhaustively_checked",
        "computed_pairs", "full_slate_source_receipts",
        "source_receipts_sha256", "candidate_catalog",
        "candidate_catalog_sha256", "policy", "reason",
        "quality_selection", "robustness_control",
        "arms_use_identical_v25_robust_score_choice_policy",
        "different_pair_selection_policies_not_different_score_models",
    ):
        if expected.get(k) != payload.get(k):
            raise ValueError("frozen_slate_changed_"+k)
    return payload, items


def _real_scores(arm, source_lookup, final_index, clock):
    selected = [source_lookup[tuple(k)]["record"] for k in arm["fixture_keys"]]
    return [_final(r, final_index.get((str(r["provider"]),
                                      str(r["fixture_id"]))), clock) for r in selected]


def _hit(arm, actual):
    expected = tuple(f"{h}-{a}" for h,a in actual)
    return any((pick["score1"],pick["score2"]) == expected
               for pick in arm["five_original_v25_robust_scores"])


def _discordance_p(quality_only, control_only):
    """Two-sided exact sign/McNemar test for paired verified policy days."""
    n = quality_only + control_only
    if n == 0:
        return None
    tail = math.fsum(math.comb(n,k) for k in range(min(
        quality_only,control_only)+1)) / (2.**n)
    return min(1.,2*tail)


def evaluate(*, root=ORIGINAL_ROOT, prematch=PREMATCH, out=OUT, now=None):
    clock = utc(now or datetime.now(timezone.utc))
    final_index, conflicts, invalid_results = index_verified_results(
        root, prematch, clock)
    rows, errors = [], Counter()
    for file in sorted((Path(out)/"decisions").glob("*.json")):
        try:
            packet, items = verify_decision(file, root=root, clock=clock)
            record = {
                "sydney_day":packet["sydney_day"],
                "policy":packet["policy"],
                "eligible_fixtures":packet["eligible_fixture_count"],
                "all_qualified_four_team_pairs":
                    packet["eligible_four_team_pair_count"],
                "same_pre_match_cutoff_for_both_arms":True,
            }
            if packet["policy"] == "ABSTAIN":
                rows.append({**record,"status":"abstained",
                             "reason":packet["reason"]})
                continue
            lookup = {_key(x):x for x in items}
            left = packet["quality_selection"]
            right = packet["robustness_control"]
            a = _real_scores(left,lookup,final_index,clock)
            b = _real_scores(right,lookup,final_index,clock)
            same = left["fixture_keys"] == right["fixture_keys"]
            record.update({
                "status":"verified" if all(v is not None for v in a+b)
                         else "pending",
                "same_fixture_pair": same,
                "quality_pair":left["fixture_keys"],
                "control_pair":right["fixture_keys"],
                "verified_quality":all(v is not None for v in a),
                "verified_control":all(v is not None for v in b),
            })
            if record["status"] == "verified":
                record.update({
                    "quality_actual_scores":[f"{h}-{g}" for h,g in a],
                    "control_actual_scores":[f"{h}-{g}" for h,g in b],
                    "quality_five_double_hit":_hit(left,a),
                    "control_five_double_hit":_hit(right,b),
                })
            rows.append(record)
        except (OSError,ValueError,KeyError,TypeError,OverflowError,IndexError) as exc:
            errors[str(exc) if isinstance(exc,ValueError)
                   else type(exc).__name__] += 1
    finished = [r for r in rows if r["status"] == "verified"]
    q = sum(r["quality_five_double_hit"] for r in finished)
    c = sum(r["control_five_double_hit"] for r in finished)
    q_only = sum(r["quality_five_double_hit"] and not
                 r["control_five_double_hit"] for r in finished)
    c_only = sum(r["control_five_double_hit"] and not
                 r["quality_five_double_hit"] for r in finished)
    review = len(finished) >= MIN_PROSPECTIVE_REVIEW_DAYS
    return {
        "schema_version":1, "version":VERSION,
        "evaluated_at_utc":clock.isoformat(),
        "study":"standalone_genuinely_prospective_two_pair_selection_policies_same_slate",
        "original_frozen_decisions":len(rows),
        "abstained_days":sum(r["status"]=="abstained" for r in rows),
        "pending_days":sum(r["status"]=="pending" for r in rows),
        "verified_paired_days":len(finished),
        "both_policies_same_pair_days":sum(r["same_fixture_pair"] for r in finished),
        "different_pair_days":sum(not r["same_fixture_pair"] for r in finished),
        "quality_wins":q, "robustness_control_wins":c,
        "quality_hit_rate":q/len(finished) if finished else None,
        "control_hit_rate":c/len(finished) if finished else None,
        "quality_hit_wilson_95":wilson(q,len(finished)),
        "control_hit_wilson_95":wilson(c,len(finished)),
        "quality_only_wins":q_only,
        "control_only_wins":c_only,
        "exact_paired_test_two_sided_p":_discordance_p(q_only,c_only),
        "initial_verified_day_review_target":MIN_PROSPECTIVE_REVIEW_DAYS,
        "initial_review_ready":review,
        "excluded_invalid_or_tampered_files":dict(errors),
        "verified_final_conflicts_excluded":len(conflicts),
        "invalid_verified_final_evidence":dict(invalid_results),
        "rows":rows,
        "caution":"Same-day DIFFERENT-fixture pair-policy selection test, not same-fixture model calibration",
        "no_promotion_to_live_model":True,
        "seventy_percent_accuracy_demonstrated":False,
        "do_not_treat_pending_as_losses":True,
    }


def run(*, root=ORIGINAL_ROOT, prematch=PREMATCH, out=OUT,
        now=None, verify_only=False, originals=None):
    clock = utc(now or datetime.now(timezone.utc))
    frozen = ({"created":[],"skipped":{},"not_frozen_inspected":[],
               "source_ineligibility":{},"no_live_model_change":True}
              if verify_only else freeze(root=root,out=out,now=now,
                                         originals=originals))
    study = evaluate(root=root,prematch=prematch,out=out,now=clock)
    output = {
        "version":VERSION, "generated_at_utc":clock.isoformat(),
        "freeze":frozen,"verified_comparison":study,
        "forecast_production_model_modified":False,
        "original_v8_v25_v26_v27_forecasts_changed":False,
        "no_additional_provider_or_llm_calls":True,
        "prospective_accuracy_improvement_not_established":True,
    }
    file = Path(out)/"status.json"
    file.parent.mkdir(parents=True,exist_ok=True)
    text = json.dumps(output,indent=2,sort_keys=True,allow_nan=False)+"\n"
    if not file.exists() or file.read_text(encoding="utf-8") != text:
        file.write_text(text,encoding="utf-8")
    return output


if __name__ == "__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--verify-only",action="store_true")
    args=cli.parse_args()
    o=run(verify_only=args.verify_only)
    print(json.dumps({
        "created":o["freeze"]["created"],
        "verified_days":o["verified_comparison"]["verified_paired_days"],
        "pending_days":o["verified_comparison"]["pending_days"],
        "abstained_days":o["verified_comparison"]["abstained_days"],
        "quality_wins":o["verified_comparison"]["quality_wins"],
        "control_wins":o["verified_comparison"]["robustness_control_wins"],
        "model_promoted":False,
    },sort_keys=True))
