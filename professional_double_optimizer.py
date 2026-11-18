"""Probability-optimal doubles and conservative, time-safe premium readiness.

Manual forecasts use the supplied complete score grids. Premium selection
requires prior, non-repeated prospective trials of this exact policy and an
improvement in prequential Brier score; a high model estimate is insufficient.
No sources, odds, results or dependence coefficients are invented here.
"""
from collections import Counter
from datetime import timedelta
from itertools import combinations
import math

import numpy as np

from v10_prospective import utc
from v25_upcoming import qualified, _pair_issue
from source_capture_clock import capture_clock

POLICY = "professional-probability-five-v1"
AUTO_SELECTION = "auto_max_raw_coverage_same_day_v1"
MIN_CALIBRATION_TRAIN = 50
MIN_CALIBRATION_TEST = 50
MIN_BUCKET = 20


def optimal_five(first, second):
    """The largest five Cartesian cells maximize exclusive outcome coverage."""
    def leading(raw):
        grid = np.asarray(raw, dtype=float)
        if (grid.ndim != 2 or min(grid.shape) < 1 or max(grid.shape) > 256
                or not np.isfinite(grid).all() or np.any(grid < 0)
                or abs(float(grid.sum())-1) > 1e-6):
            raise ValueError("normalized_full_grid_required")
        return sorted(((float(p), h, a) for h, row in enumerate(grid)
                       for a, p in enumerate(row) if p > 0),
                      key=lambda item: (-item[0], item[1], item[2]))[:5]
    left, right = leading(first), leading(second)
    choices = [{"score1": f"{h}-{a}", "score2": f"{j}-{k}",
                "joint_probability": p*q, "first_probability": p,
                "second_probability": q}
               for p, h, a in left for q, j, k in right]
    return sorted(choices, key=lambda r: (-r["joint_probability"],
                  r["score1"], r["score2"]))[:5]


def manual(first, second, *, now, clock=None):
    started,observed=capture_clock(now,clock)
    clock = started
    pair = []
    for row in (first, second):
        item, reason = qualified(row, clock)
        if item is None:
            return {"status": "abstained", "reason": reason}
        pair.append(item)
    issue = _pair_issue(*pair, clock)
    if issue:
        return {"status": "abstained", "reason": issue}
    if any(row.get("dependence_warning") for row in (first, second)):
        return {"status": "abstained", "reason": "validated_joint_dependence_model_required"}
    choices = optimal_five(first["result"]["matrix"], second["result"]["matrix"])
    if len(choices) != 5:
        return {"status": "abstained", "reason": "five_supported_distinct_outcomes_required"}
    clock=observed()
    if _pair_issue(*pair,clock):
        return {"status":"abstained","reason":"analysis_completed_outside_pre_match_window"}
    return {"status": "ready", "policy_id": POLICY,
            "selection_scope":"manual_same_day_v1",
            "generated_at_utc": clock.isoformat(), "sydney_date": pair[0]["day"],
            "first_fixture": first["fixture"], "second_fixture": second["fixture"],
            "model_versions": [str(row["result"].get("model_version") or "unknown")
                               for row in (first, second)],
            "source_times_utc": [item["captured_at_utc"] for item in pair],
            "source_forecast_refs":dict(zip(("first","second"),
                                            [row.get("source_forecast") or {} for row in (first,second)])),
            "early_frozen_source": [item["early_source"] for item in pair],
            "choices": choices,
            "estimated_coverage": math.fsum(r["joint_probability"] for r in choices),
            "coverage_is_calibrated": False, "matches_assumed_independent": True,
            "dependence_model_available": False, "target_70_percent_proven": False,
            "expected_value": {"available": False, "reason": "executable_double_prices_not_supplied"},
            "realized_roi": None, "original_forecasts_changed": False}


def _bucket(probability):
    return min(9, int(probability*10))


def _adjust(probability, prior):
    rows = [r for r in prior if _bucket(r["estimated_coverage"]) == _bucket(probability)]
    if len(rows) < MIN_BUCKET:
        return None
    return (sum(r["hit"] for r in rows)+30*probability)/(len(rows)+30)


def calibration_trials(rows, *, now, model_versions):
    """Exclude future, repeated, wrong-policy and unverifiable trial metadata.

    Caller must supply trials extracted from its immutable verified ledger;
    these checks do not authenticate arbitrary externally supplied outcomes.
    """
    clock = utc(now)
    accepted, rejected, seen, seen_events = [], Counter(), set(), set()
    for record in sorted((r for r in rows or [] if isinstance(r,dict)),
                         key=lambda r: str(r.get("predicted_at_utc", ""))):
        try:
            if (record.get("policy_id") != POLICY or record.get("prospective") is not True
                    or record.get("selection_scope") != AUTO_SELECTION
                    or record.get("verification_status") != "verified"
                    or list(record.get("model_versions") or []) != list(model_versions)):
                raise ValueError("wrong_policy_model_or_unverified_trial")
            ids = [tuple(str(v) for v in identity) for identity in record["fixture_identities"]]
            if len(ids) != 2 or any(len(k) != 2 or not all(k) for k in ids) or len(set(ids)) != 2:
                raise ValueError("two_distinct_fixture_identities_required")
            if any(k in seen for k in ids):
                raise ValueError("repeated_fixture_trial")
            predicted = utc(record["predicted_at_utc"])
            verified = utc(record["verified_at_utc"])
            kickoffs = [utc(k) for k in record["kickoffs_utc"]]
            if len(kickoffs) != 2 or not predicted < min(kickoffs) <= max(kickoffs) < verified < clock:
                raise ValueError("not_available_before_decision")
            events=[]
            for at,names in zip(kickoffs,record["fixture_team_names"]):
                normalized=tuple(" ".join(str(name).casefold().split()) for name in names)
                if len(normalized)!=2 or not all(normalized) or normalized[0]==normalized[1]:
                    raise ValueError("verified_event_team_names_required")
                events.append((at.isoformat(),*normalized))
            if len(events)!=2 or len(set(events))!=2 or set(events)&seen_events:
                raise ValueError("repeated_cross_provider_event")
            p = record["estimated_coverage"]
            if type(p) not in (int, float) or not math.isfinite(p) or not 0 < p <= 1:
                raise ValueError("invalid_model_probability")
            if type(record["hit"]) is not bool:
                raise ValueError("verified_set_boolean_required")
            seen.update(ids)
            seen_events.update(events)
            accepted.append({**record, "predicted": predicted, "verified": verified})
        except (KeyError, TypeError, ValueError):
            rejected["invalid_or_repeated_calibration_trial"] += 1
    return accepted, dict(rejected)


def calibrate(probability, rows, *, now, model_versions):
    if type(probability) not in (int,float) or not math.isfinite(probability) or not 0<probability<=1:
        raise ValueError("valid_model_coverage_required")
    accepted, rejected = calibration_trials(rows, now=now, model_versions=model_versions)
    differences = []
    for row in accepted:
        prior = [r for r in accepted if r["verified"] < row["predicted"]]
        if len(prior) < MIN_CALIBRATION_TRAIN:
            continue
        adjusted = _adjust(row["estimated_coverage"], prior)
        if adjusted is not None:
            actual = float(row["hit"])
            differences.append((row["estimated_coverage"]-actual)**2-(adjusted-actual)**2)
    n = len(differences)
    lower = None
    if n > 1:
        mean = math.fsum(differences)/n
        variance = math.fsum((x-mean)**2 for x in differences)/(n-1)
        lower = mean-1.96*math.sqrt(variance/n)
    adjusted = _adjust(probability, accepted)
    ready = bool(n >= MIN_CALIBRATION_TEST and lower is not None and lower > 0
                 and adjusted is not None)
    return {"ready": ready, "prior_unique_pairs": len(accepted),
            "prequential_test_pairs": n, "brier_advantage_95_lower": lower,
            "calibrated_coverage_estimate": adjusted if ready else None,
            "rejections": rejected, "target_70_percent_proven": False,
            "reason": None if ready else "insufficient_independent_calibration_evidence",
            "cross_pair_independence_not_established": True}


def trials_from_verified_ledger(records, finals, *, source_root="data/v10_prospective"):
    """Compute hits from original choices and identity-matched final scores."""
    from ziczac_unified_audit import _final_matches
    output = []
    for record in records or []:
        try:
            if (not isinstance(record,dict) or record.get("source") != "ai_five_double"
                    or record.get("method") != POLICY):
                continue
            legs, choices = record["legs"], record["choices"]
            if len(legs)!=2 or len(choices)!=5 or len({(p["score1"],p["score2"]) for p in choices})!=5:
                continue
            results=[finals.get((str(leg["provider"]),str(leg["fixture_id"]))) for leg in legs]
            if not all(_final_matches(leg,result) for leg,result in zip(legs,results)):
                continue
            actual=[f"{r['home_goals']}-{r['away_goals']}" for r in results]
            metadata=record["policy_metadata"]
            if not isinstance(metadata,dict) or metadata.get("selection_scope") != AUTO_SELECTION:
                continue
            # Owner metadata alone cannot validate a probability estimate.
            # Reconstruct it from its immutable, pre-selection source grids.
            from five_double_accuracy import _sources
            predicted=utc(record["captured_at_utc"])
            references=metadata["source_forecast_refs"]
            models=_sources(source_root,references,predicted)
            if (metadata["model_versions"]!=[m["model_version"] for m in models]
                    or [utc(t) for t in metadata["source_times_utc"]] !=
                       [utc(m["captured_at_utc"]) for m in models]
                    or any(not timedelta(0)<=predicted-utc(m["captured_at_utc"])<=timedelta(hours=24)
                           or (utc(m["kickoff_utc"])-predicted).total_seconds()<=300 for m in models)
                    or any(not references[side].get("input_digest") or
                           references[side]["input_digest"]!=model["input_digest"]
                           for side,model in zip(("first","second"),models))):
                continue
            source_names=[]
            for leg,model in zip(legs,models):
                if (any(str(leg[k])!=str(model[k]) for k in
                        ("provider","fixture_id","home_team_id","away_team_id"))
                        or utc(leg["kickoff_utc"])!=utc(model["kickoff_utc"])):
                    raise ValueError("private_fixture_does_not_match_original_source")
                names=tuple(" ".join(str(model[k]).casefold().split())
                            for k in ("home_team","away_team"))
                if (not all(names) or names!=tuple(" ".join(str(leg[k]).casefold().split())
                                                  for k in ("home_team","away_team"))):
                    raise ValueError("private_team_names_do_not_match_original_source")
                source_names.extend(names)
            if (len(set(source_names))!=4 or
                    len({(m["provider"],str(m[k])) for m in models
                         for k in ("home_team_id","away_team_id")})!=4):
                continue
            from zoneinfo import ZoneInfo
            if len({utc(m["kickoff_utc"]).astimezone(ZoneInfo("Australia/Sydney")).date()
                    for m in models})!=1:
                continue
            expected=optimal_five(*(model["score_matrix"] for model in models))
            if ([(p["score1"],p["score2"]) for p in choices]!=
                    [(p["score1"],p["score2"]) for p in expected]
                    or type(metadata["estimated_coverage"]) not in (int,float)
                    or not math.isfinite(metadata["estimated_coverage"])
                    or not math.isclose(metadata["estimated_coverage"],
                          math.fsum(p["joint_probability"] for p in expected),rel_tol=0,abs_tol=1e-12)):
                continue
            output.append({"policy_id":POLICY,"prospective":True,"verification_status":"verified",
                           "selection_scope":metadata.get("selection_scope"),
                           "model_versions":metadata["model_versions"],
                           "estimated_coverage":metadata["estimated_coverage"],
                           "predicted_at_utc":record["captured_at_utc"],
                           "verified_at_utc":max(utc(r["verified_at_utc"]) for r in results).isoformat(),
                           "kickoffs_utc":[leg["kickoff_utc"] for leg in legs],
                           "fixture_identities":[[leg["provider"],str(leg["fixture_id"])] for leg in legs],
                           "fixture_team_names":[[leg["home_team"],leg["away_team"]] for leg in legs],
                           "hit":any(p["score1"]==actual[0] and p["score2"]==actual[1] for p in choices)})
        except (KeyError,ValueError,TypeError,OSError):
            continue
    return output


def _recent_rows(rows, clock):
    rejected=Counter()
    unique = {}
    for row in rows or []:
        item, reason = qualified(row, clock)
        if item is None:
            rejected[reason] += 1
            continue
        # A timestamp assigned to the current clock is not evidence of source age.
        stamped = (row.get("frozen_at_utc") or row.get("generated_at_utc") or row.get("captured_at_utc"))
        if stamped is None or clock-utc(stamped) > timedelta(hours=24):
            rejected["source_not_timestamped_within_24_hours"] += 1
            continue
        key = item["key"]
        if key not in unique or item["captured_at_utc"] > unique[key][0]:
            unique[key] = (item["captured_at_utc"], row)
    return unique,rejected


def auto_shadow(rows, *, now, clock=None):
    """Bootstrap the exact automatic selection cohort without claiming premium skill."""
    clock,observed=capture_clock(now,clock)
    unique,rejected=_recent_rows(rows,clock)
    found=[]
    for first,second in combinations((v[1] for v in unique.values()),2):
        report=manual(first,second,now=observed(),clock=observed)
        if report["status"]!="ready":
            rejected[report["reason"]]+=1
            continue
        report["selection_scope"]=AUTO_SELECTION
        found.append(report)
    found.sort(key=lambda r:(-r["estimated_coverage"],str(r["first_fixture"]["id"]),str(r["second_fixture"]["id"])))
    completed=observed()
    found=[r for r in found if _ready_at_completion(r,completed)]
    return {"status":"ready" if found else "abstained","selection":found[0] if found else None,
            "eligible_pairs_compared":len(found),"qualified_fixtures":len(unique),
            "rejections":dict(rejected),"research_only":True,"target_70_percent_proven":False}


def _ready_at_completion(report,at):
    if (any((utc(f["utcDate"])-at).total_seconds()<=300 for f in
            (report["first_fixture"],report["second_fixture"]))
            or any(at-utc(t)>timedelta(hours=24) for t in report["source_times_utc"])):
        return False
    report["generated_at_utc"]=at.isoformat()
    return True


def premium(rows, *, now, verified_trials=(), clock=None):
    """Compare every qualified same-day pair; abstain without usable calibration."""
    clock,observed=capture_clock(now,clock)
    unique,rejected=_recent_rows(rows,clock)
    found,screened=[],0
    for first, second in combinations((v[1] for v in unique.values()), 2):
        report = manual(first, second, now=observed(),clock=observed)
        if report["status"] != "ready":
            rejected[report["reason"]] += 1
            continue
        screened += 1
        calibrated = calibrate(report["estimated_coverage"], verified_trials,
                               now=clock, model_versions=report["model_versions"])
        if not calibrated["ready"]:
            rejected[calibrated["reason"]] += 1
            continue
        report["calibration"] = calibrated
        report["coverage_is_calibrated"] = True
        report["selection_scope"]="premium_max_calibrated_coverage_v1"
        found.append(report)
    found.sort(key=lambda r: (-r["calibration"]["calibrated_coverage_estimate"],
                              -r["estimated_coverage"], str(r["first_fixture"]["id"]),
                              str(r["second_fixture"]["id"])))
    completed=observed()
    found=[r for r in found if _ready_at_completion(r,completed)]
    return {"status": "ready" if found else "abstained",
            "selection": found[0] if found else None, "eligible_pairs_compared": screened,
            "qualified_fixtures": len(unique), "rejections": dict(rejected),
            "target_70_percent_proven": False, "external_requests": 0,
            "reason": None if found else "no_pair_has_validated_calibrated_coverage"}


def compare_score_prices(report, quotes, *, now):
    """Separate model hit probability from illustrative per-leg price products.

    A product of two single-leg prices is not proof that a bookmaker will
    accept that double. No realized ROI or executable edge is inferred.
    """
    clock = utc(now)
    indexed = {}
    for quote in quotes or []:
        try:
            at = utc(quote["updated_at_utc"])
            observed = utc(quote["observed_at_utc"])
            odds = quote["decimal_odds"]
            goals = quote["score"]
            match=next((f for f in (report["first_fixture"],report["second_fixture"])
                        if str(f.get("provider") or "football-data.org")==str(quote["provider"])
                        and str(f["id"])==str(quote["fixture_id"])),None)
            if (not at <= observed <= clock or clock-at > timedelta(hours=12)
                    or type(odds) not in (int, float) or not math.isfinite(odds) or odds <= 1
                    or not isinstance(goals, (list, tuple)) or len(goals) != 2
                    or any(type(g) is not int or g < 0 for g in goals)
                    or not quote.get("bookmaker") or match is None
                    or utc(quote["kickoff_utc"])!=utc(match["utcDate"])
                    or quote.get("market")!="correct_score_90min"):
                continue
            key = (str(quote["provider"]), str(quote["fixture_id"]),
                   str(quote["bookmaker"]), f"{goals[0]}-{goals[1]}")
            if key not in indexed or observed > indexed[key][0]:
                indexed[key] = (observed, float(odds))
        except (KeyError, ValueError, TypeError):
            continue
    first, second = report["first_fixture"], report["second_fixture"]
    if any(clock >= utc(f["utcDate"]) for f in (first, second)):
        return {"available": False, "reason": "price_comparison_after_kickoff"}
    comparison = []
    books = {key[2] for key in indexed}
    for choice in report["choices"]:
        for book in sorted(books):
            keys = [(str(f.get("provider") or "football-data.org"), str(f["id"]), book,
                     choice["score1" if i == 0 else "score2"])
                    for i, f in enumerate((first, second))]
            if all(key in indexed for key in keys):
                product = math.prod(indexed[key][1] for key in keys)
                comparison.append({"score1": choice["score1"], "score2": choice["score2"],
                                   "bookmaker": book, "leg_price_product": product,
                                   "model_joint_probability": choice["joint_probability"],
                                   "illustrative_expected_return_per_unit":
                                       product*choice["joint_probability"]-1,
                                   "executable_double_quote_verified": False})
    return {"available": bool(comparison), "comparisons": comparison,
            "price_products_are_not_executable_quotes": True,
            "model_probabilities_are_uncalibrated": True,
            "aggregate_coverage_is_calibrated":report["coverage_is_calibrated"],
            "realized_roi": None,
            "reason": None if comparison else "matching_timed_correct_score_prices_missing"}
