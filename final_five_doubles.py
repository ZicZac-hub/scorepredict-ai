"""Exhaustive, prospective selection of one five-ticket exact-score set per slate."""
import argparse
from collections import Counter
from datetime import datetime, timezone
from itertools import combinations
import json
import math
from pathlib import Path
from tempfile import NamedTemporaryFile
import unicodedata

from five_double_accuracy import _final, utc, wilson
from professional_double_optimizer import optimal_five
from source_capture_clock import capture_clock
from v26_temporal_score import _write_once
from v27_pair_selection import MAX_EVALUATED_PAIRS, digest
from v28_independent_pair_trial import (
    MAX_EVALUATION_AGE, MIN_LEAD, ORIGINAL_ROOT, OUT as V28_OUT, PREMATCH,
    SYDNEY, verify_decision,
)
from v41_unified_ensemble import core, final_pair
from v41_unified_trial import (
    PROVIDER_ROOT, _components, _same_calculation,
)
from verified_result_index import index_verified_results

ROOT = Path("data/final_five_doubles")
VERSION = "full-slate-v41-final-five-v1"
TARGET = .70
MINIMUM_VERIFIED = 100
MINIMUM_ISSUANCE = .80
METHODS = ("baseline_v8", "poisson_marginal", "unified_core", "unified_final")


def _key(item):
    return tuple(map(str, item["key"]))


def _team_name(value):
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def _team_names(item):
    record = item["record"]
    return {_team_name(record["home_team"]), _team_name(record["away_team"])}


def _choices(first, second):
    choices = optimal_five(first, second)
    if len(choices) != 5:
        raise ValueError("five_supported_score_combinations_required")
    return choices, math.fsum(row["joint_probability"] for row in choices)


def compare_slate(items, *, provider_root, cutoff, max_pairs=MAX_EVALUATED_PAIRS):
    """Compare every four-team pair and all supported scores at one cutoff.

    For each normalized independent-match model, the five largest Cartesian
    products exactly maximize set mass. A cell outside either leg's top five
    cannot improve that optimum: at least five products using the other leg's
    same score are already as large. Equal-mass ties are resolved by the shared
    optimizer, then slate ties by canonical fixture keys.
    """
    items = sorted(items, key=_key)
    if len({_key(item) for item in items}) != len(items):
        raise ValueError("duplicate_fixture_on_slate")
    if type(max_pairs) is not int or max_pairs < 1:
        raise ValueError("invalid_pair_comparison_budget")
    names = {_key(item): _team_names(item) for item in items}
    pairs = [(a, b) for a, b in combinations(items, 2)
             if len(a["teams"] | b["teams"]) == 4
             and len(names[_key(a)] | names[_key(b)]) == 4]
    result = {
        "eligible_fixtures": len(items), "eligible_pairs": len(pairs),
        "compared_pairs": 0, "methods": list(METHODS),
        "evidence_counts": {"xg": 0, "market": 0}, "selection": None,
        "baseline_selection": None, "candidate_catalog": [],
    }
    if not pairs or len(pairs) > max_pairs:
        return {**result, "status": "abstained", "reason": (
            "no_eligible_distinct_team_pair" if not pairs else "full_slate_exceeds_comparison_budget")}

    participating = {_key(item): item for pair in pairs for item in pair}
    models = {}
    for key, item in participating.items():
        bundle = _components(item["record"], provider_root=provider_root, cutoff=cutoff)
        models[key] = {
            "bundle": bundle, "baseline_v8": item["grid"],
            "unified_core": core(item["grid"]),
            "poisson_marginal": item["alternatives"]["poisson_marginal"],
        }
    result["component_receipts"] = [
        {"fixture_key": list(key), "sources": {
            kind: component["reference"] if component else None
            for kind, component in model["bundle"].items()}}
        for key, model in models.items()
    ]

    best = {"unified_final": None, "baseline_v8": None}
    for a, b in pairs:
        keys = [_key(a), _key(b)]
        left, right = (models[key] for key in keys)
        choices = {name: _choices(left[name], right[name]) for name in METHODS[:-1]}
        bundles = [left["bundle"], right["bundle"]]
        evidence = {"xg": False, "market": False}
        if any(all(bundle[kind] is not None for bundle in bundles) for kind in evidence):
            grids, evidence = final_pair([a["grid"], b["grid"]], bundles)
            choices["unified_final"] = _choices(*grids)
        else:
            choices["unified_final"] = choices["unified_core"]
        for kind, used in evidence.items():
            result["evidence_counts"][kind] += int(used)

        fixture_keys = [list(key) for key in keys]
        masses = {name: choices[name][1] for name in METHODS}
        result["candidate_catalog"].append({
            "fixture_keys": fixture_keys, "model_mass": masses, "paired_evidence": evidence,
        })
        for name, leader in best.items():
            if leader is None or round(masses[name], 12) > round(leader["estimate"], 12):
                best[name] = {
                    "fixture_keys": fixture_keys, "estimate": masses[name],
                    "five_doubles": choices[name][0],
                    "names": [f"{item['record']['home_team']} vs {item['record']['away_team']}"
                              for item in (a, b)],
                }

    return {
        **result, "status": "compared", "compared_pairs": len(pairs),
        "selection": best["unified_final"], "baseline_selection": best["baseline_v8"],
    }


def _read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _signed(record):
    return {**record, "sha256": digest(record)}


def _unchanged(record):
    return record.get("sha256") == digest({k: v for k, v in record.items() if k != "sha256"})


def _comparison(items, *, cutoff, provider_root):
    try:
        return compare_slate(items, cutoff=cutoff, provider_root=provider_root)
    except (ValueError, KeyError, TypeError, AttributeError, ArithmeticError, OSError):
        # A failed candidate makes the full comparison incomplete.
        return {
            "status": "abstained", "reason": "candidate_model_or_evidence_unusable",
            "eligible_fixtures": len(items), "eligible_pairs": None, "compared_pairs": 0,
            "methods": list(METHODS), "evidence_counts": {},
            "selection": None, "baseline_selection": None, "candidate_catalog": [],
        }


def _pre_match(items, predicted):
    return all(predicted <= item["kickoff"] - MIN_LEAD for item in items)


def freeze(*, root=ROOT, v28_out=V28_OUT, source_root=ORIGINAL_ROOT,
           provider_root=PROVIDER_ROOT, now=None, clock=None):
    """Record new shadow decisions only, before any compared fixture is in play."""
    started, observe = capture_clock(now, clock)
    activation_path = Path(root) / "activation.json"
    _write_once(activation_path, {
        "version": VERSION, "first_window_utc": (started - MAX_EVALUATION_AGE).isoformat(),
    })
    activation = _read(activation_path)
    if activation["version"] != VERSION:
        raise ValueError("final_policy_activation_version_changed")
    first_window = utc(activation["first_window_utc"])
    created = []
    for path in sorted((Path(v28_out) / "decisions").glob("*.json")):
        destination = Path(root) / "decisions" / path.name
        if destination.exists():
            continue
        try:
            source = _read(path)
            issued = utc(source["actually_issued_at_utc"])
            if issued < first_window or not issued <= started <= issued + MAX_EVALUATION_AGE:
                continue
            source, items = verify_decision(path, root=source_root, clock=started)
            cutoff = utc(source["policy_cutoff_utc"])
            comparison = _comparison(items, cutoff=cutoff, provider_root=provider_root)
            predicted = observe()
            if predicted > issued + MAX_EVALUATION_AGE or not _pre_match(items, predicted):
                comparison = {
                    **comparison, "status": "abstained", "reason": "full_slate_window_closed",
                    "selection": None, "baseline_selection": None, "candidate_catalog": [],
                }
            record = _signed({
                "version": VERSION, "sydney_day": path.stem,
                "source_slate_sha256": digest(source), "cutoff_utc": cutoff.isoformat(),
                "predicted_at_utc": predicted.isoformat(), "comparison": comparison,
            })
            if _write_once(destination, record):
                created.append(path.stem)
        except (OSError, ValueError, TypeError, KeyError) as exc:
            # Preserve an explicit failure in the job log rather than issuing a partial set.
            print(f"Final Five Doubles skipped {path.stem}: {exc}")
    return created


def _verified_decision(path, *, v28_out, source_root, provider_root, clock):
    record = _read(path)
    if record["version"] != VERSION or record["sydney_day"] != path.stem or not _unchanged(record):
        raise ValueError("final_decision_version_date_or_digest_invalid")
    predicted = utc(record["predicted_at_utc"])
    if predicted > clock:
        raise ValueError("final_decision_in_future")
    source, items = verify_decision(Path(v28_out) / "decisions" / path.name,
                                    root=source_root, clock=predicted)
    cutoff = utc(source["policy_cutoff_utc"])
    issued = utc(source["actually_issued_at_utc"])
    if digest(source) != record["source_slate_sha256"] or utc(record["cutoff_utc"]) != cutoff:
        raise ValueError("final_source_slate_changed")
    comparison = record["comparison"]
    if predicted > issued + MAX_EVALUATION_AGE or not _pre_match(items, predicted):
        if (comparison["status"] != "abstained" or comparison.get("reason") != "full_slate_window_closed"
                or comparison["selection"] is not None or comparison["baseline_selection"] is not None):
            raise ValueError("final_decision_missed_pre_match_window")
    elif not _same_calculation(comparison, _comparison(items, cutoff=cutoff, provider_root=provider_root)):
        raise ValueError("final_slate_comparison_changed")
    return record, {_key(item): item for item in items}


def _outcome(selection, items, finals, clock):
    models = [items[tuple(key)]["record"] for key in selection["fixture_keys"]]
    scores = [_final(model, finals.get(tuple(key)), clock)
              for model, key in zip(models, selection["fixture_keys"])]
    if any(score is None for score in scores):
        return None
    score1, score2 = (f"{home}-{away}" for home, away in scores)
    return any((row["score1"], row["score2"]) == (score1, score2)
               for row in selection["five_doubles"])


def validation_summary(rows):
    verified = [row for row in rows if row["status"] == "verified"]
    counts = Counter(row["status"] for row in rows)
    wins = sum(row["hit"] is True for row in verified)
    issued = len(verified) + counts["pending"]
    coverage = issued / len(rows) if rows else 0.
    conservative = wilson(wins, issued)
    # The fixed first cohort prevents repeated statistical looks at an
    # expanding cohort from being the sole trigger for publication.
    first_cohort = [r for r in rows if r["status"] in {"verified", "pending"}][:MINIMUM_VERIFIED]
    cohort_verified = sum(r["status"] == "verified" for r in first_cohort)
    cohort_wins = sum(r["status"] == "verified" and r["hit"] is True for r in first_cohort)
    cohort_interval = wilson(cohort_wins, MINIMUM_VERIFIED) if cohort_verified == MINIMUM_VERIFIED else None
    paired = [r for r in verified if r.get("baseline_hit") is not None]
    return {
        "wins": wins, "verified": len(verified), "pending": counts["pending"],
        "abstained": counts["abstained"], "invalid": counts["invalid"],
        "issued_fraction": coverage, "minimum_issued_fraction": MINIMUM_ISSUANCE,
        "observed_hit_rate": wins / len(verified) if verified else None,
        "wilson_95": wilson(wins, len(verified)),
        "publication_wilson_95": conservative, "target": TARGET,
        "minimum_verified": MINIMUM_VERIFIED,
        "first_cohort_verified": cohort_verified, "first_cohort_wins": cohort_wins,
        "first_cohort_wilson_95": cohort_interval,
        "matched_v8_control": {"verified_days": len(paired),
                               "v8_hits": sum(r["baseline_hit"] for r in paired),
                               "final_hits": sum(r["hit"] for r in paired)},
        "ready": bool(cohort_interval and cohort_interval[0] >= TARGET
                      and coverage >= MINIMUM_ISSUANCE and not counts["invalid"]
                      and conservative and conservative[0] >= TARGET),
    }


def evaluate(*, root=ROOT, v28_out=V28_OUT, source_root=ORIGINAL_ROOT,
             provider_root=PROVIDER_ROOT, prematch=PREMATCH, now=None):
    clock = utc(now or datetime.now(timezone.utc))
    paths = sorted((Path(root) / "decisions").glob("*.json"))
    finals, _, _ = index_verified_results(source_root, prematch, clock) if paths else ({}, None, None)
    rows, active, seen = [], {}, set()
    for path in paths:
        row = {"sydney_day": path.stem}
        try:
            record, items = _verified_decision(
                path, v28_out=v28_out, source_root=source_root,
                provider_root=provider_root, clock=clock,
            )
            comparison = record["comparison"]
            selection = comparison["selection"]
            row.update({k: comparison[k] for k in (
                "eligible_fixtures", "eligible_pairs", "compared_pairs", "methods", "evidence_counts")})
            row["decision_sha256"] = record["sha256"]
            if selection is None:
                rows.append({**row, "status": "abstained", "reason": comparison["reason"]})
                continue
            selected = [items[tuple(key)] for key in selection["fixture_keys"]]
            fixture_keys = {_key(item) for item in selected} | {
                (item["kickoff"], *sorted(_team_names(item))) for item in selected
            }
            if seen & fixture_keys:
                raise ValueError("repeated_fixture_across_evaluation_days")
            seen.update(fixture_keys)
            hit = _outcome(selection, items, finals, clock)
            baseline = _outcome(comparison["baseline_selection"], items, finals, clock)
            first_kickoff = min(item["kickoff"] for item in selected)
            rows.append({**row, "status": "pending" if hit is None else "verified",
                         "hit": hit, "baseline_hit": baseline, "estimate": selection["estimate"],
                         "first_kickoff_utc": first_kickoff.isoformat()})
            if first_kickoff > clock and hit is None:
                active[path.stem] = record["sha256"]
        except (OSError, ValueError, TypeError, KeyError, OverflowError) as exc:
            rows.append({**row, "status": "invalid", "reason": str(exc)})
    activation_path = Path(root) / "activation.json"
    if activation_path.exists():
        activation = _read(activation_path)
        if activation["version"] != VERSION:
            raise ValueError("final_policy_activation_version_changed")
        first_window = utc(activation["first_window_utc"])
        recorded = {path.stem for path in paths}
        for path in sorted((Path(v28_out) / "decisions").glob("*.json")):
            if path.stem in recorded:
                continue
            try:
                issued = utc(_read(path)["actually_issued_at_utc"])
                if not first_window <= issued < clock - MAX_EVALUATION_AGE:
                    continue
                source, _ = verify_decision(path, root=source_root, clock=clock)
                rows.append({"sydney_day": path.stem, "status": "abstained",
                             "reason": "scheduled_freeze_window_missed",
                             "eligible_fixtures": source["eligible_fixture_count"],
                             "eligible_pairs": source["eligible_four_team_pair_count"],
                             "compared_pairs": 0})
            except (OSError, ValueError, TypeError, KeyError, OverflowError) as exc:
                rows.append({"sydney_day": path.stem, "status": "invalid", "reason": str(exc)})
    rows.sort(key=lambda row: row["sydney_day"])
    # The open day's unsettled result is not evidence for its own verification.
    previous = [row for row in rows if row["sydney_day"] < clock.astimezone(SYDNEY).date().isoformat()]
    return {
        "version": VERSION, "evaluated_at_utc": clock.isoformat(),
        "validation": validation_summary(previous), "rows": rows, "active_decisions": active,
    }


def presentation(*, root=ROOT, now=None):
    """Expose selection details only for a current, signed, independently reviewed decision."""
    root = Path(root)
    clock = utc(now or datetime.now(timezone.utc))
    empty = {"state": "no_decision", "decision": None, "validation": validation_summary([])}
    try:
        report = _read(root / "status.json")
        if report["version"] != VERSION or utc(report["evaluated_at_utc"]) > clock:
            return {**empty, "state": "evidence_unavailable"}
        day = clock.astimezone(SYDNEY).date().isoformat()
        previous = [row for row in report["rows"] if row["sydney_day"] < day]
        validation = validation_summary(previous)
        view = {**empty, "validation": validation}
        upcoming = [r for r in report["rows"] if r["sydney_day"] >= day]
        row = next((r for r in upcoming if r["status"] == "pending"
                    and utc(r["first_kickoff_utc"]) > clock), None)
        if row is None and upcoming:
            row = upcoming[0]
        if row is None:
            return view
        decision_day = row["sydney_day"]
        change = {key: row.get(key) for key in (
            "sydney_day", "eligible_fixtures", "eligible_pairs", "compared_pairs",
            "estimate", "methods", "evidence_counts")}
        view["decision"] = change
        if row["status"] == "invalid":
            return {**view, "state": "evidence_unavailable"}
        if row["status"] == "abstained":
            return {**view, "state": "incomplete_slate"}
        if utc(row["first_kickoff_utc"]) <= clock:
            return {**view, "state": "expired"}
        if not validation["ready"]:
            return {**view, "state": "awaiting_validation"}
        if report["active_decisions"].get(decision_day) != row["decision_sha256"]:
            return {**view, "state": "evidence_unavailable"}
        # Require a canonical calendar key before deriving the local filename.
        day_name = datetime.strptime(decision_day, "%Y-%m-%d").date().isoformat()
        if day_name != decision_day:
            return {**view, "state": "evidence_unavailable"}
        decision = _read(root / "decisions" / (day_name + ".json"))
        if (not _unchanged(decision) or decision["sha256"] != row["decision_sha256"]
                or decision["version"] != VERSION or decision["sydney_day"] != day_name
                or decision["comparison"]["status"] != "compared"
                or decision["comparison"]["eligible_pairs"] != decision["comparison"]["compared_pairs"]):
            return {**view, "state": "evidence_unavailable"}
        selection = decision["comparison"]["selection"]
        change["names"] = selection["names"]
        change["five_doubles"] = selection["five_doubles"]
        return {**view, "state": "available"}
    except FileNotFoundError:
        return empty
    except (OSError, ValueError, TypeError, KeyError, ArithmeticError):
        return {**empty, "state": "evidence_unavailable"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    created = [] if args.verify_only else freeze()
    report = evaluate()
    ROOT.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(mode="w", dir=ROOT, encoding="utf-8", delete=False) as output:
        json.dump(report, output, indent=2, sort_keys=True, allow_nan=False)
        output.write("\n")
    Path(output.name).replace(ROOT / "status.json")
    print(f"Final Five Doubles: {len(created)} new decisions, "
          f"{report['validation']['verified']} verified historical sets")


if __name__ == "__main__":
    main()
