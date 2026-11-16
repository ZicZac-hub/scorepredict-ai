"""On-demand V25 exact-score doubles for every eligible upcoming fixture.

Strictly a *session-only* preview, not the official frozen V25 prospective
experiment. Uses the existing complete pre-kickoff probability grids and the
same four mathematical sensitivity distributions as the archived V25 research.
No new provider requests, no unverified goals or bookmaker prices.

Pairing convention: exactly two different fixtures on the same Sydney date,
four distinct teams, and exactly five distinct ordered exact-score outcomes.
"""
from collections import defaultdict
from datetime import datetime, timezone
from itertools import combinations
import csv
import io

import numpy as np

from ai_five_double_engine import eligible, identity, pair_check, teams, SYDNEY
from on_demand_ai import utc
from professional_analyst_shadow_v25 import _pair, VARIANTS
from v21_goal_model_comparison import variants

MAX_AUTO_PAIR_CHECKS_PER_DAY = 1500


def prepare(row, *, now=None):
    """Return a source-matched original grid or a documented exclusion."""
    clock = utc(now or datetime.now(timezone.utc))
    if not isinstance(row, dict) or not eligible(row, now=clock):
        return None, "not_upcoming_or_missing_score_candidates"
    fixture=row.get("fixture") or {}
    source=row.get("result") or {}
    raw=source.get("matrix")
    if raw is None:
        return None, "full_probability_grid_unavailable"
    try:
        matrix=np.asarray(raw, dtype=float)
        if (matrix.ndim != 2 or min(matrix.shape) < 5
                or max(matrix.shape) > 25 or not np.isfinite(matrix).all()
                or np.any(matrix < 0)
                or abs(float(matrix.sum())-1)>1e-5):
            return None, "invalid_full_probability_grid"
        # This validation also guards extreme / implausible mean-goal inputs.
        alternatives=variants(matrix)
        kickoff=utc(row.get("kickoff") or fixture["utcDate"])
        provider,fid,_ = identity(row)
        h_id=str((fixture.get("homeTeam") or {}).get("id") or "")
        a_id=str((fixture.get("awayTeam") or {}).get("id") or "")
        if (not h_id or not a_id or h_id==a_id
                or h_id=="None" or a_id=="None"):
            return None, "unverified_team_identity"
        source_tag="modelled_in_current_session_not_frozen"
        captured_at=None
        if row.get("archive_frozen"):
            src=row.get("source_forecast") or {}
            frozen=row.get("frozen_at_utc")
            if (not frozen or not src.get("forecast_id")
                    or not src.get("input_digest")
                    or str(src.get("provider"))!=provider
                    or str(src.get("fixture_id"))!=fid
                    or str(src.get("home_team_id"))!=h_id
                    or str(src.get("away_team_id"))!=a_id
                    or utc(src["kickoff_utc"])!=kickoff):
                return None, "archive_source_identity_not_verified"
            captured_at=utc(frozen)
            if not captured_at or not captured_at<kickoff or captured_at>clock:
                return None, "archive_timestamp_invalid"
            source_tag=("saved_recent_forecast" if
                        (clock-captured_at).total_seconds() <=72*3600 else
                        "saved_early_forecast_not_refreshed")
        return {
            "row":row, "grid":matrix, "alternatives":alternatives,
            "key":(provider,fid), "teams":set(teams(row)),
            "day":kickoff.astimezone(SYDNEY).date().isoformat(),
            "kickoff":kickoff, "source_type":source_tag,
            "source_captured_at":captured_at.isoformat() if captured_at else None,
            "label":str((fixture.get("homeTeam") or {}).get("name"))+" vs "+
                    str((fixture.get("awayTeam") or {}).get("name")),
        }, None
    except (TypeError, KeyError, ValueError, OverflowError, IndexError):
        return None, "invalid_source_probability_or_identity"


def candidates(rows, *, now=None):
    """All genuinely eligible future grids, across all available Sydney dates."""
    clock=utc(now or datetime.now(timezone.utc))
    valid={}
    rejected=defaultdict(int)
    for row in rows or []:
        prepared,why=prepare(row,now=clock)
        if prepared is None:
            rejected[why]+=1
            continue
        key=prepared["key"]
        existing=valid.get(key)
        # Prefer real immutable source grids over session models where both
        # refer to the identical provider+fixture and kickoff identity.
        if existing is None or (prepared["source_type"].startswith("saved_")
                                and not existing["source_type"].startswith("saved_")):
            valid[key]=prepared
    groups=defaultdict(list)
    for item in valid.values():
        groups[item["day"]].append(item)
    return {day:sorted(items,key=lambda x:(x["kickoff"],x["key"]))
            for day,items in sorted(groups.items())},dict(rejected)


def predict(first,second, *, now=None, prepared=False):
    """Compute five V25 robustness-ranked exact-score pairs before kickoff."""
    clock=utc(now or datetime.now(timezone.utc))
    a=first if prepared else prepare(first,now=clock)[0]
    b=second if prepared else prepare(second,now=clock)[0]
    if a is None or b is None:
        return {"status":"unavailable","reason":"Both matches need complete, source-valid pre-kickoff goal distributions."}
    if pair_check(a["row"],b["row"],now=clock):
        return {"status":"unavailable","reason":pair_check(a["row"],b["row"],now=clock)}
    try:
        score=_pair(a,b)
    except (TypeError,ValueError,KeyError,FloatingPointError):
        return {"status":"unavailable","reason":"Four-model exact-score comparison could not be validated."}
    five=score["five_doubles"]
    if len(five)!=5 or len({(x["score1"],x["score2"]) for x in five})!=5:
        return {"status":"unavailable","reason":"Fewer than five distinct model-supported score doubles."}
    return {
        "status":"ready","date":a["day"],
        "first":a["label"],"second":b["label"],
        "first_fixture":a["row"]["fixture"],
        "second_fixture":b["row"]["fixture"],
        "first_source":a["source_type"],"second_source":b["source_type"],
        "first_captured_at":a["source_captured_at"],
        "second_captured_at":b["source_captured_at"],
        "five":five, "baseline_five":score["baseline_five"],
        "robust_model_mass":score["robust_lower_envelope_model_mass"],
        "baseline_v8_mass":score["v8_optimal_five_model_mass"],
        "robust_scores_under_v8_mass":score["robust_five_v8_model_mass"],
        "model_names":list(VARIANTS),
        "generated_at_utc":clock.isoformat(),
        "not_a_calibrated_win_probability":True,
        "session_only_until_explicitly_saved":True,
        "model_promotion":False,
    }


def rank_day(rows, *, now=None, max_checks=MAX_AUTO_PAIR_CHECKS_PER_DAY):
    """Rank same-day fixture pairs; disclose incomplete searches if bounded."""
    clock=utc(now or datetime.now(timezone.utc))
    if type(max_checks) is not int or max_checks < 1:
        raise ValueError("Positive pair-check limit required")
    valid=[item for item in rows if item["kickoff"]>clock]
    ranked=[]
    checked=0
    total_possible=0
    for a,b in combinations(valid,2):
        if pair_check(a["row"],b["row"],now=clock):
            continue
        total_possible+=1
        if checked>=max_checks:
            continue
        checked+=1
        got=predict(a,b,now=clock,prepared=True)
        if got["status"]=="ready":
            ranked.append((got["robust_model_mass"],a["key"],b["key"],got))
    ranked.sort(key=lambda t:(-t[0],t[1],t[2]))
    return {
        "pairs":[p[3] for p in ranked],
        "screened_pairs":checked,
        "total_pair_candidates":total_possible,
        "exhaustive":checked==total_possible,
    }


def cover_all(groups, *, now=None, max_checks=MAX_AUTO_PAIR_CHECKS_PER_DAY):
    """Choose research pairs covering all pairable upcoming fixtures.

    On a day with an odd number of matches, reuse one opponent fixture once
    rather than pretending a single match constitutes a two-match double.
    """
    clock=utc(now or datetime.now(timezone.utc))
    selected=[]
    skipped=[]
    incomplete_days=[]
    for day, items in sorted(groups.items()):
        report=rank_day(items,now=clock,max_checks=max_checks)
        if not report["exhaustive"]:
            incomplete_days.append(day)
        if not report["pairs"]:
            skipped.append({"day":day,"reason":"No valid two-fixture pair"})
            continue
        used=set()
        for pair in report["pairs"]:
            first_id=(str(pair["first_fixture"].get("provider")),
                      str(pair["first_fixture"]["id"]))
            second_id=(str(pair["second_fixture"].get("provider")),
                       str(pair["second_fixture"]["id"]))
            if first_id not in used and second_id not in used:
                selected.append(pair)
                used.update((first_id,second_id))
        # For an odd fixture count, a remaining match can be paired with one
        # already used fixture, but never with itself or a different date.
        all_keys={x["key"] for x in items}
        remaining=all_keys-used
        for missing in sorted(remaining):
            found=next((pair for pair in report["pairs"]
                        if (str(pair["first_fixture"].get("provider")),
                            str(pair["first_fixture"]["id"]))==missing or
                           (str(pair["second_fixture"].get("provider")),
                            str(pair["second_fixture"]["id"]))==missing),None)
            if found is None:
                skipped.append({"day":day,"reason":"Unpairable fixture "+missing[1]})
            elif not any(p["first_fixture"]["id"]==found["first_fixture"]["id"]
                         and p["second_fixture"]["id"]==found["second_fixture"]["id"]
                         for p in selected):
                selected.append(found)
    return {
        "status":"ready" if selected else "unavailable",
        "pairs":selected,"skipped":skipped,
        "incomplete_days":incomplete_days,
        "covered_pair_count":len(selected),
        "session_only":True,
        "all_upcoming_fixtures_guaranteed":False,
        "not_a_verified_accuracy_claim":True,
    }


def csv_bytes(pairs):
    """Export genuine source-backed session predictions (not bookmaker odds)."""
    out=io.StringIO()
    w=csv.writer(out)
    w.writerow(["Sydney date","First fixture","Second fixture","Double",
                "First exact score","Second exact score","V25 lower model mass",
                "Frozen source 1","Frozen source 2","Created UTC"])
    for pair in pairs:
        if pair.get("status")!="ready":
            continue
        for index,score in enumerate(pair["five"],1):
            w.writerow([pair["date"],pair["first"],pair["second"],index,
                        score["score1"],score["score2"],
                        format(pair["robust_model_mass"],".8f"),
                        pair["first_source"],pair["second_source"],
                        pair["generated_at_utc"]])
    return out.getvalue().encode("utf-8")
