"""V29 forward-only distinct-pair sensitivity trial and V28 outcome-readiness audit.

V28 quality and its V25-style control can select exactly the same two fixtures.
That provides zero pair-policy discrimination, even after both scores settle.
V29 preregisters the highest-quality *different* eligible pair, on the same
original V28 source slate. It is an exploratory contrast, not a claimed
higher-probability model, and never rewrites or backfills V28 selections.

The independent monitor always grades saved V28 decisions and reports why
pending fixtures remain unverified: future kickoff versus post-kickoff result
awaiting authenticated verification. Pending never means a loss.
"""
import argparse
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
import json

from five_double_accuracy import _final, utc, wilson
from verified_result_index import index_verified_results
from v28_independent_pair_trial import (
    ORIGINAL_ROOT, PREMATCH, OUT as V28_OUT,
    _discordance_p, _hit, _key, _real_scores, verify_decision,
)
from v26_temporal_score import _write_once
from v27_pair_selection import digest

VERSION = "v29-distinct-pair-sensitivity-v1-research-only"
OUT = Path("data/v29_pair_sensitivity")
SOURCE_WINDOW = timedelta(minutes=20)
MIN_LEAD = timedelta(minutes=45)
INITIAL_EVIDENCE_TARGET = 100


def _distinct_alternative(v28, items, issued):
    """Deterministic ranked runner-up, without consulting any final results.

    The V28 pair choice must be excluded from the V29 contrast: if both
    original arms agree, testing them again would not answer this question.
    A potential alternative can share one fixture, so it is NOT a statistically
    independent fixture sample; exact shared-fixture counts are disclosed.
    """
    reference_keys = {
        tuple(tuple(k) for k in v28[x]["fixture_keys"])
        for x in ("quality_selection", "robustness_control")
    }
    sources = {_key(x): x for x in items}
    ranked = sorted(v28["candidate_catalog"],
                    key=lambda p: (-p["quality_rank_score"], p["fixture_keys"]))
    feasible = []
    for candidate in ranked:
        keys = tuple(tuple(k) for k in candidate["fixture_keys"])
        if keys in reference_keys:
            continue
        if any(tuple(key) not in sources or
               issued > sources[tuple(key)]["kickoff"]-MIN_LEAD
               for key in candidate["fixture_keys"]):
            continue
        feasible.append(candidate)
    return feasible[0] if feasible else None


def _choice_report(v28, alternate):
    main = {tuple(k) for k in v28["quality_selection"]["fixture_keys"]}
    control = {tuple(k) for k in v28["robustness_control"]["fixture_keys"]}
    explorer = {tuple(k) for k in alternate["fixture_keys"]}
    return {
        "shared_fixtures_with_v28_quality": len(main & explorer),
        "shared_fixtures_with_v28_robustness": len(control & explorer),
        "quality_and_control_same_pair": main == control,
        "different_pair_from_both_v28_arms": explorer != main and explorer != control,
        "selected_rank_score_is_not_winning_probability": True,
    }


def freeze(*, root=ORIGINAL_ROOT, v28_out=V28_OUT, out=OUT, now=None):
    """Only frozen V28 packets from the previous 20 minutes may create V29.

    The source decision must itself be independently valid, including every
    original forecast, every pair ranking and the issue-time cutoff.
    """
    clock = utc(now or datetime.now(timezone.utc))
    created, excluded = [], Counter()
    for old_file in sorted((Path(v28_out)/"decisions").glob("*.json")):
        day = old_file.stem
        destination = Path(out)/"decisions"/(day+".json")
        if destination.exists():
            excluded["write_once_v29_already_exists"] += 1
            continue
        try:
            original, items = verify_decision(old_file, root=root, clock=clock)
            source_issued = utc(original["actually_issued_at_utc"])
            if not source_issued <= clock <= source_issued+SOURCE_WINDOW:
                excluded["original_v28_older_than_20_minute_window"] += 1
                continue
            if original["policy"] != "FROZEN":
                excluded["v28_abstained"] += 1
                continue
            issued = clock if now is not None else utc(datetime.now(timezone.utc))
            if not source_issued <= issued <= source_issued+SOURCE_WINDOW:
                excluded["generated_too_late"] += 1
                continue
            alternative = _distinct_alternative(original, items, issued)
            packet = {
                "schema_version": 1,
                "version": VERSION,
                "sydney_day": day,
                "source_v28_packet_sha256": digest(original),
                "original_v28_cutoff_utc": original["policy_cutoff_utc"],
                "original_v28_issued_at_utc": source_issued.isoformat(),
                "v29_issued_at_utc": issued.isoformat(),
                "source_candidate_catalog_sha256":
                    original["candidate_catalog_sha256"],
                "source_original_forecast_receipts_sha256":
                    original["source_receipts_sha256"],
                "decision": "DISTINCT_PAIR_FROZEN" if alternative else "ABSTAIN",
                "alternative": alternative,
                "absence_reason": (None if alternative else
                                   "no_distinct_pair_still_45_minutes_before_kickoff"),
                "comparison": (_choice_report(original, alternative)
                               if alternative else None),
                "sensitivity_arm_not_expected_to_beat_best_rank": True,
                "same_original_v28_source_slate": True,
                "no_original_v28_or_v25_edits": True,
                "not_promoted_to_live_model": True,
                "70_percent_accuracy_not_demonstrated": True,
            }
            if _write_once(destination, packet):
                created.append({"sydney_day": day, "decision": packet["decision"]})
        except (OSError, KeyError, ValueError, TypeError, OverflowError):
            excluded["invalid_or_unverifiable_original_v28"] += 1
    return {"created": created, "excluded": dict(excluded),
            "live_forecast_changed": False}


def _verify_v29(file, v28_file, root, clock):
    packet = json.loads(file.read_text(encoding="utf-8"))
    day = file.stem
    if (packet.get("schema_version") != 1 or packet.get("version") != VERSION
            or packet.get("sydney_day") != day
            or packet.get("decision") not in ("DISTINCT_PAIR_FROZEN","ABSTAIN")):
        raise ValueError("wrong_v29_version_or_day")
    v28, items = verify_decision(v28_file, root=root, clock=clock)
    source_issued = utc(v28["actually_issued_at_utc"])
    own_issued = utc(packet["v29_issued_at_utc"])
    if not source_issued <= own_issued <= source_issued+SOURCE_WINDOW or own_issued > clock:
        raise ValueError("v29_frozen_outside_pre_match_registration_window")
    for key, check in (
        ("source_v28_packet_sha256",digest(v28)),
        ("original_v28_cutoff_utc",v28["policy_cutoff_utc"]),
        ("original_v28_issued_at_utc",source_issued.isoformat()),
        ("source_candidate_catalog_sha256",v28["candidate_catalog_sha256"]),
        ("source_original_forecast_receipts_sha256",v28["source_receipts_sha256"]),
    ):
        if packet.get(key) != check:
            raise ValueError("v28_provenance_changed_"+key)
    expected = (_distinct_alternative(v28,items,own_issued)
                if v28["policy"]=="FROZEN" else None)
    desired = "DISTINCT_PAIR_FROZEN" if expected else "ABSTAIN"
    expected_reason = (None if expected else
                       "no_distinct_pair_still_45_minutes_before_kickoff")
    expected_comparison = (_choice_report(v28,expected) if expected else None)
    if (packet["decision"] != desired or packet.get("alternative") != expected
            or packet.get("comparison") != expected_comparison
            or packet.get("absence_reason") != expected_reason):
        raise ValueError("v29_alternative_not_reproducible")
    return packet,v28,items


def _progress_for_v28(v28_out, *, root, finals, clock):
    """Separate future kickoffs from results awaiting provider verification."""
    rows, invalid = [], Counter()
    for file in sorted((Path(v28_out)/"decisions").glob("*.json")):
        try:
            packet, items = verify_decision(file, root=root, clock=clock)
            row = {"sydney_day": file.stem,
                   "eligible_pairs": packet["eligible_four_team_pair_count"],
                   "policy": packet["policy"]}
            if packet["policy"] != "FROZEN":
                row["status"] = "abstained"
                rows.append(row)
                continue
            quality = packet["quality_selection"]
            robust = packet["robustness_control"]
            q_keys = {tuple(x) for x in quality["fixture_keys"]}
            c_keys = {tuple(x) for x in robust["fixture_keys"]}
            sources = {_key(x): x for x in items}
            fixtures = []
            for key in sorted(q_keys | c_keys):
                source = sources[key]
                actual = _real_scores(
                    {"fixture_keys":[list(key)]},sources,finals,clock)[0]
                status = ("verified_final" if actual is not None else
                          "waiting_for_kickoff" if clock < source["kickoff"] else
                          "awaiting_independently_verified_result")
                fixtures.append({
                    "provider":key[0], "fixture_id":key[1],
                    "kickoff_utc":source["kickoff"].isoformat(),
                    "final_status":status,
                    "verified_goals":(list(actual) if actual is not None else None),
                })
            complete = all(x["final_status"]=="verified_final" for x in fixtures)
            row.update({
                "status":"verified" if complete else "pending",
                "quality_and_robustness_identical_pair": q_keys == c_keys,
                "unique_fixtures_awaiting_kickoff":sum(
                    x["final_status"]=="waiting_for_kickoff" for x in fixtures),
                "unique_fixtures_awaiting_verified_final":sum(
                    x["final_status"]=="awaiting_independently_verified_result"
                    for x in fixtures),
                "verified_unique_fixtures":sum(
                    x["final_status"]=="verified_final" for x in fixtures),
                "unique_fixture_count":len(fixtures),
                "fixture_statuses":fixtures,
            })
            rows.append(row)
        except (OSError, ValueError, KeyError, TypeError, OverflowError,
                IndexError) as error:
            invalid[str(error) if isinstance(error,ValueError)
                    else type(error).__name__] += 1
    active = [r for r in rows if r["policy"]=="FROZEN"]
    total = sum(r["unique_fixture_count"] for r in active)
    verified = sum(r["verified_unique_fixtures"] for r in active)
    next_kickoffs = sorted(
        f["kickoff_utc"] for row in active for f in row["fixture_statuses"]
        if f["final_status"]=="waiting_for_kickoff")
    return {
        "v28_frozen_active_days":len(active),
        "v28_identical_policy_pair_days":sum(
            r["quality_and_robustness_identical_pair"] for r in active),
        "v28_distinct_policy_pair_days":sum(
            not r["quality_and_robustness_identical_pair"] for r in active),
        "v28_verified_distinct_fixture_count":verified,
        "v28_total_distinct_fixture_count":total,
        "v28_fixtures_before_kickoff":sum(
            r["unique_fixtures_awaiting_kickoff"] for r in active),
        "v28_fixtures_past_kickoff_awaiting_verified_result":sum(
            r["unique_fixtures_awaiting_verified_final"] for r in active),
        "v28_verified_day_count":sum(r["status"]=="verified" for r in active),
        "next_saved_fixture_kickoff_utc":next_kickoffs[0] if next_kickoffs else None,
        "invalid_original_v28_days_excluded":dict(invalid),
        "day_details":rows,
        "verified_fixture_count_is_not_five_double_win_rate":True,
    }


def evaluate(*, root=ORIGINAL_ROOT, prematch=PREMATCH,
             v28_out=V28_OUT, out=OUT, now=None):
    clock = utc(now or datetime.now(timezone.utc))
    final_index, conflicts, result_errors = index_verified_results(
        root, prematch, clock)
    progress = _progress_for_v28(
        v28_out,root=root,finals=final_index,clock=clock)
    rows, invalid = [], Counter()
    for file in sorted((Path(out)/"decisions").glob("*.json")):
        try:
            original = Path(v28_out)/"decisions"/file.name
            packet, v28, items = _verify_v29(file,original,root,clock)
            row = {
                "sydney_day":file.stem,
                "decision":packet["decision"],
                "alt_pair_shares_fixtures_with_v28_quality":(
                    packet["comparison"]["shared_fixtures_with_v28_quality"]
                    if packet["comparison"] else None),
            }
            if packet["decision"]=="ABSTAIN":
                rows.append({**row,"status":"abstained"})
                continue
            sources = {_key(x):x for x in items}
            choice = packet["alternative"]
            q = v28["quality_selection"]
            c = v28["robustness_control"]
            a,b,d = (_real_scores(arm,sources,final_index,clock)
                     for arm in (choice,q,c))
            row.update({
                "alternative_fixture_keys":choice["fixture_keys"],
                "v28_quality_fixture_keys":q["fixture_keys"],
                "v28_robustness_fixture_keys":c["fixture_keys"],
                "alternative_verified_legs":sum(x is not None for x in a),
                "control_verified_legs":sum(x is not None for x in b),
                "status":("verified" if all(x is not None for x in a+b+d)
                          else "pending"),
            })
            if row["status"]=="verified":
                row.update({
                    "alternative_scores":[f"{h}-{g}" for h,g in a],
                    "control_scores":[f"{h}-{g}" for h,g in b],
                    "alternative_five_hit":_hit(choice,a),
                    "quality_control_five_hit":_hit(q,b),
                    "original_robustness_five_hit":_hit(c,d),
                })
            rows.append(row)
        except (OSError,ValueError,KeyError,TypeError,OverflowError,IndexError) as exc:
            invalid[str(exc) if isinstance(exc,ValueError)
                    else type(exc).__name__] += 1
    complete = [r for r in rows if r["status"]=="verified"]
    aw = sum(r["alternative_five_hit"] for r in complete)
    qw = sum(r["quality_control_five_hit"] for r in complete)
    a_only = sum(r["alternative_five_hit"] and not r["quality_control_five_hit"]
                 for r in complete)
    q_only = sum(r["quality_control_five_hit"] and not r["alternative_five_hit"]
                 for r in complete)
    return {
        "version":VERSION,
        "evaluated_at_utc":clock.isoformat(),
        "v28_capture_and_result_readiness":progress,
        "v29_distinct_pair_experiment":{
            "decisions":len(rows),
            "frozen_alternatives":sum(
                r["decision"]=="DISTINCT_PAIR_FROZEN" for r in rows),
            "abstentions":sum(r["status"]=="abstained" for r in rows),
            "pending_alternatives":sum(r["status"]=="pending" for r in rows),
            "both_arms_verified_days":len(complete),
            "alternative_five_wins":aw,
            "v28_quality_control_five_wins":qw,
            "alternative_wilson95":wilson(aw,len(complete)),
            "v28_quality_wilson95":wilson(qw,len(complete)),
            "alternative_only_wins":a_only,
            "v28_quality_only_wins":q_only,
            "paired_exact_p":_discordance_p(a_only,q_only),
            "minimum_initial_verified_day_review":INITIAL_EVIDENCE_TARGET,
            "initial_review_ready":len(complete)>=INITIAL_EVIDENCE_TARGET,
            "invalid_or_modified_set_exclusions":dict(invalid),
            "rows":rows,
        },
        "verified_final_evidence_conflicts_excluded":len(conflicts),
        "invalid_final_result_sources_excluded":dict(result_errors),
        "no_source_or_forecast_changes":True,
        "prospective_model_improvement_not_proven":True,
        "live_predictions_unchanged":True,
        "seventy_percent_not_proven":True,
    }


def run(*, root=ORIGINAL_ROOT, prematch=PREMATCH, v28_out=V28_OUT,
        out=OUT, now=None, verify_only=False):
    clock = utc(now or datetime.now(timezone.utc))
    frozen = ({"created":[],"excluded":{},"live_forecast_changed":False}
              if verify_only else freeze(
                  root=root,v28_out=v28_out,out=out,now=now))
    # The run may have just issued a new write-once packet. Its validity is
    # assessed after issuance rather than with the earlier run-start instant.
    evaluation_clock = clock if now is not None else utc(datetime.now(timezone.utc))
    result = evaluate(root=root,prematch=prematch,v28_out=v28_out,
                      out=out,now=evaluation_clock)
    payload = {"version":VERSION,"freeze":frozen,"research":result,
               "generated_at_utc":evaluation_clock.isoformat(),
               "no_new_provider_or_llm_requests":True}
    dest=Path(out)/"status.json"
    dest.parent.mkdir(parents=True,exist_ok=True)
    report=json.dumps(payload,indent=2,sort_keys=True,allow_nan=False)+"\n"
    if not dest.exists() or dest.read_text(encoding="utf-8")!=report:
        dest.write_text(report,encoding="utf-8")
    return payload


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only",action="store_true",
                        help="Grade saved decisions only; do not create alternatives")
    args=parser.parse_args()
    saved=run(verify_only=args.verify_only)
    report=saved["research"]
    print(json.dumps({
        "new_alternate_decisions":saved["freeze"]["created"],
        "verified_v28_fixtures":report["v28_capture_and_result_readiness"][
            "v28_verified_distinct_fixture_count"],
        "v28_same_pair_days":report["v28_capture_and_result_readiness"][
            "v28_identical_policy_pair_days"],
        "v29_verified_alternative_days":report["v29_distinct_pair_experiment"][
            "both_arms_verified_days"],
        "promoted":False,
    },sort_keys=True))
