"""Bounded ChatGPT API research coordinator for ZicZac's 24/7 five-double bot.

GitHub Actions runs the clock. The OpenAI API optionally writes one concise
research memo per UTC day; its free-text is never executed, committed as code,
used to select historical winning scores, or treated as statistical evidence.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path

import requests

ROOT = Path("data/autobot")
VERSION = "ziczac-five-double-research-bot-v1"
OPENAI_URL = "https://api.openai.com/v1/responses"
REPO = "ZicZac-hub/scorepredict-ai"
MAX_AI_CALLS_PER_UTC_DAY = 1
DEFAULT_AI_MODEL = "gpt-4o-mini"


def load(path):
    path = Path(path)
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except (OSError, ValueError):
        return None


def as_rate(hits, verified):
    if type(hits) is not int or type(verified) is not int or verified < 0 or hits < 0 or hits > verified:
        return None
    return hits / verified if verified else None


def source_capture_health(coverage, *, now):
    """Read-only health report; no source capture, model fitting, or API calls.

    A historical report without a trustworthy UTC timestamp is UNKNOWN,
    never presented as a fresh successful collector run. These counts are
    source-attempt diagnostics, not verified predictions or hit rates.
    """
    coverage = coverage if isinstance(coverage, dict) else {}
    def nonnegative(value):
        return value if type(value) is int and value >= 0 else 0
    def counts(value):
        return {str(k):nonnegative(v) for k,v in value.items()
                if isinstance(k,str) and type(v) is int and v >= 0
                } if isinstance(value,dict) else {}
    skipped=counts(coverage.get("v10_shadow_skipped"))
    leagues={}
    raw=coverage.get("v10_shadow_skipped_by_league")
    if isinstance(raw,dict):
        leagues={str(code):counts(reasons)
                 for code,reasons in raw.items() if isinstance(code,str)}
    def per_league_bands(raw):
        if not isinstance(raw,dict):
            return {}
        return {str(code):counts(bins) for code,bins in raw.items()
                if isinstance(code,str) and isinstance(bins,dict)}
    stamp=coverage.get("collector_started_at_utc")
    age=None
    freshness="unknown_missing_capture_timestamp"
    if isinstance(stamp,str):
        try:
            dt=datetime.fromisoformat(stamp.replace("Z","+00:00"))
            if dt.tzinfo is not None and dt.utcoffset() is not None:
                age=round((now.astimezone(timezone.utc)-
                           dt.astimezone(timezone.utc)).total_seconds()/60,1)
                if age < -1:
                    freshness="invalid_future_capture_timestamp"
                    age=None
                else:
                    freshness="stale" if age > 180 else "recent"
            else:
                freshness="invalid_capture_timestamp"
        except (ValueError,TypeError,OverflowError):
            freshness="invalid_capture_timestamp"
    raw_uefa = coverage.get("uefa_two_season_history_coverage")
    uefa_rows = raw_uefa if isinstance(raw_uefa, list) else []
    uefa_states = {}
    for row in uefa_rows:
        if not isinstance(row, dict) or row.get("competition") not in ("CL", "EL"):
            continue
        for season in row.get("seasons") or []:
            if not isinstance(season, dict):
                continue
            state = season.get("source_state")
            if state in ("not_observed", "request_unavailable", "stale_response", "empty_response", "rows_returned"):
                uefa_states[state] = uefa_states.get(state, 0) + 1
    one_short = sum(isinstance(row,dict) and type(row.get("minimum_missing_to_five")) is int and row["minimum_missing_to_five"]==1 for row in uefa_rows)
    return {
        "uefa_fixtures_one_qualified_match_short": one_short,
        "uefa_history_audit_available": isinstance(raw_uefa, list),
        "uefa_history_fixture_observations": len(uefa_rows),
        "uefa_history_season_source_states": uefa_states,
        "collector_started_at_utc":stamp if freshness in ("recent","stale") else None,
        "collector_freshness":freshness,
        "age_minutes":age,
        "processed_fixtures":nonnegative(coverage.get("processed_fixtures")),
        "v8_saved":nonnegative(coverage.get("v10_shadow_v8_saved")),
        "v8_abstentions":sum(skipped.values()),
        "v8_source_errors":len(coverage.get("v10_shadow_errors") or [])
                           if isinstance(coverage.get("v10_shadow_errors"),list) else 0,
        "v8_abstention_reasons":skipped,
        "v37_sparse_saved":nonnegative(coverage.get("v37_sparse_shadow_saved")),
        "v37_sparse_saved_by_league":counts(
            coverage.get("v37_sparse_shadow_saved_by_league")),
        "v37_sparse_status_by_league":{
            str(k):counts(v) for k,v in
            (coverage.get("v37_sparse_shadow_coverage") or {}).items()
            if isinstance(k,str) and isinstance(v,dict)
        },
        "v8_abstention_reasons_by_league":leagues,
        "v8_saved_by_league":counts(coverage.get("v10_shadow_v8_saved_by_league")),
        "v8_min_team_history_bins_by_league":per_league_bands(
            coverage.get("v8_min_team_history_bins_by_league")),
        "v8_home_history_bins_by_league":per_league_bands(
            coverage.get("v8_qualified_home_history_bins_by_league")),
        "v8_away_history_bins_by_league":per_league_bands(
            coverage.get("v8_qualified_away_history_bins_by_league")),
        "producer_api_budget_unchanged":True,
        "not_a_prediction_accuracy_measure":True,
    }


def unified_status(performance, capture):
    """Keep original results, V41 outcomes and availability distinct."""
    available = isinstance(performance, dict) and bool(performance)
    report = performance if available else {}
    capture = capture if isinstance(capture,dict) else {}
    arm = (report.get("methods") or {}).get("unified_final") or {}
    comparison = (report.get("paired_comparisons") or {}).get("unified_final") or {}
    n, hits = arm.get("verified"), arm.get("hits")
    return {
        "report_available": available,
        "evaluated_at_utc": report.get("evaluated_at_utc"),
        "capture_at_utc": capture.get("completed_at_utc"),
        "outside_decision_window": capture.get("outside_decision_window"),
        "new_trials": capture.get("new_trials") or [],
        "issued": arm.get("issued"), "pending": arm.get("pending"),
        "abstained": arm.get("abstained"), "verified": n, "hits": hits,
        "observed_hit_rate": as_rate(hits,n),
        "same_pair_comparison": comparison,
        "source_rejections": report.get("source_rejections") or {},
        "minimum_review_pairs": 100,
        "research_only": True,
        "target_70_percent_proven": False,
    }


def assemble(v8, v25, v30, v34, v35, *, now, v36=None, v28=None, coverage=None, v37=None, v38=None, v39=None, v41=None, v41_capture=None):
    """Only preexisting independently validated archived report totals."""
    baseline = (v8 or {}).get("observed") or {}
    matched_v8 = (v25 or {}).get("v8") or {}
    matched_v25 = (v25 or {}).get("v25") or {}
    v35 = v35 or {}
    stats = v35.get("methods") or {}
    v36 = v36 or {}
    v36stats = v36.get("methods") or {}
    v37 = v37 or {}
    v38_available = isinstance(v38, dict) and bool(v38)
    v38 = v38 if v38_available else {}
    v28 = v28 or {}
    v28freeze = v28.get("freeze") or {}
    v28grade = v28.get("verified_comparison") or {}
    v8n, v8hits = baseline.get("verified_sets", 0), baseline.get("hits", 0)
    matched_n = (v25 or {}).get("verified_matched_pairs", 0)
    original = {
        "verified_sets": v8n,
        "hits": v8hits,
        "hit_rate": as_rate(v8hits, v8n),
        "pending_sets": baseline.get("pending_sets", 0),
    }
    matched = {
        "verified_same_pair_sets": matched_n,
        "v25_wins": matched_v25.get("five_double_hits", 0),
        "v8_wins": matched_v8.get("five_double_hits", 0),
        "pending_pairs": (v25 or {}).get("pending_matched_pairs", 0),
    }
    if not all([v8, v25, v30, v34, v35]):
        integrity = "INCOMPLETE_REQUIRED_RESEARCH_FILES"
    elif (v30.get("invalid_or_tampered_frozen_days_excluded") or
          v35.get("invalid_or_altered_v35_sets_excluded") or
          v34.get("original_source_exclusions")):
        integrity = "AUDIT_EXCLUSIONS_REQUIRE_REVIEW"
    else:
        integrity = "SOURCE_AUDITS_RECORDED"
    both_absent = (v34 or {}).get("frozen_ticket_failure_attribution", {}).get(
        "v8_optimal_original", {}).get("both_leg_scores_absent", 0)
    source_health = source_capture_health(coverage, now=now)
    next_priority = (
        "Investigate why original score distributions miss one or both final "
        "leg scores; prioritize independent high-goal calibration trials."
        if both_absent else
        "Collect enough independent, frozen pre-kickoff same-pair results "
        "before comparing or promoting models."
    )
    if source_health["collector_freshness"] != "recent":
        next_priority = ("Verify the pre-match source collector's freshness, then "
                         "diagnose authenticated V8 historical-evidence gaps. "
                         "Do not issue or backdate V36 scores without fresh originals.")
    elif source_health["v8_abstentions"] > source_health["v8_saved"]:
        next_priority = ("Investigate league-specific authenticated V8 history "
                         "shortfalls before changing score distributions; preserve "
                         "the existing minimums and prospectively frozen tickets.")
    return {
        "version": VERSION,
        "last_run_utc": now.astimezone(timezone.utc).isoformat(),
        "target_five_double_hit_rate": 0.70,
        "goal_verified": False,
        "runtime_mode": "recurring_GitHub_Actions_not_continuous_host_process",
        "priority": "five_exact_score_doubles",
        "integrity": integrity,
        "original_v8_daily_five_doubles": original,
        "v8_v25_same_pair": matched,
        "source_capture_health": source_health,
        "v37_sparse_uefa_research": {
            "frozen_unique_physical_fixtures": v37.get("frozen_unique_physical_fixtures", 0),
            "verified_matched_physical_fixtures": v37.get("verified_matched_physical_fixtures", 0),
            "pending": v37.get("pending", 0),
            "invalid_or_tampered_excluded": v37.get("invalid_or_tampered_excluded", 0),
            "methods": v37.get("methods") or {},
            "research_only": True,
            "not_a_five_double_trial": True,
            "ready_for_model_promotion": False,
        },
        "v28_upstream_capture": {
            "new_original_decisions_last_run": len(v28freeze.get("created") or []),
            "original_frozen_decisions": v28grade.get("original_frozen_decisions", 0),
            "verified_paired_days": v28grade.get("verified_paired_days", 0),
            "pending_days": v28grade.get("pending_days", 0),
            "source_ineligibility": v28freeze.get("source_ineligibility") or {},
            "new_source_capture_required_for_future_v36": True,
        },
        "v30_frozen_candidate_pair_rows": {
            "correlated_verified": (v30 or {}).get("candidate_pair_rows_independently_verified", 0),
            "pending": (v30 or {}).get("candidate_pair_rows_awaiting_verified_finals", 0),
            "completed_independent_sydney_days": (v30 or {}).get("fully_verified_slate_days", 0),
        },
        "v34_diagnosis": {
            "unique_verified_fixtures": (v34 or {}).get("unique_verified_physical_fixtures", 0),
            "both_actual_leg_scores_missing": both_absent,
        },
        "v35_challengers": {
            "method_names": list(stats),
            "frozen_days": v35.get("frozen_prospective_days", 0),
            "verified_same_pair_days": v35.get("verified_same_pair_days", 0),
            "pending_days": v35.get("pending_same_pair_days", 0),
            "methods": {k: {
                "hits": row.get("hits"),
                "observed_hit_rate": row.get("observed_hit_rate"),
                "verified_days": row.get("verified_same_pair_days"),
                "review_gate_passed": row.get("review_gate_passed_on_hit_rate_only", False),
            } for k, row in stats.items()},
            "ready_for_separate_production_review": v35.get("ready_for_separate_production_review", []),
        },
        "v36_challengers": {
            "frozen_days": v36.get("frozen_prospective_days", 0),
            "verified_independent_sydney_days": v36.get("verified_independent_sydney_days", 0),
            "pending_days": v36.get("pending_days", 0),
            "unique_verified_physical_fixtures": v36.get("unique_verified_physical_fixtures", 0),
            "methods": {name: {
                "hits": item.get("hits"),
                "verified_days": item.get("verified_independent_sydney_days"),
                "observed_hit_rate": item.get("observed_five_double_hit_rate"),
                "mean_proper_scores": item.get("mean_proper_scores"),
            } for name, item in v36stats.items()},
            "ready_for_separate_production_review": v36.get("ready_for_separate_production_review", []),
            "research_only_not_live_promoted": True,
        },
        "v38_challengers": {
            "report_available": v38_available,
            "frozen_days": v38.get("frozen_prospective_days") if v38_available else None,
            "verified_days": v38.get("verified_independent_sydney_days") if v38_available else None,
            "pending_days": v38.get("pending_days") if v38_available else None,
            "capture_skipped": (v38.get("capture") or {}).get("skipped") or {},
            "methods": {name: {
                "hits": row.get("hits"),
                "verified_days": row.get("verified_independent_sydney_days"),
                "observed_hit_rate": as_rate(row.get("hits"), row.get("verified_independent_sydney_days")),
                "mean_proper_scores": row.get("mean_proper_scores"),
            } for name, row in (v38.get("methods") or {}).items() if isinstance(row, dict)},
            "research_only_not_live_promoted": True,
        },
        "v39_challengers": {
            "report_available": bool(v39),
            "frozen_days": (v39 or {}).get("frozen_prospective_days"),
            "verified_days": (v39 or {}).get("verified_independent_sydney_days"),
            "pending_days": (v39 or {}).get("pending_days"),
            "research_only": True,
        },
        "v41_unified_final": unified_status(v41, v41_capture),
        "next_priority": next_priority,
        "production_model_changed_by_bot": False,
        "ai_research_memo_is_non_executable": True,
        "no_claimed_70_percent_without_independent_evidence": True,
    }


def classify_openai_http_error(response):
    """Return only allowlisted diagnostic fields; never persist raw API bodies."""
    status = getattr(response, "status_code", None)
    result = {"http_status": status if type(status) is int else None,
              "error_code": "unavailable", "error_type": "unavailable"}
    if response is None:
        return result
    try:
        error = response.json().get("error", {})
        if not isinstance(error, dict):
            return result
        codes = {"insufficient_quota", "rate_limit_exceeded",
                 "billing_hard_limit_reached", "invalid_api_key",
                 "invalid_request_error"}
        types = {"insufficient_quota", "rate_limit_exceeded",
                 "invalid_request_error", "authentication_error"}
        if error.get("code") in codes:
            result["error_code"] = error["code"]
        if error.get("type") in types:
            result["error_type"] = error["type"]
    except (ValueError, AttributeError, TypeError):
        pass
    return result


def api_research_memo(report, *, key, model=DEFAULT_AI_MODEL):
    """One bounded optional API request: recommendations are TEXT ONLY."""
    snapshot = {key: report[key] for key in (
        "target_five_double_hit_rate", "integrity", "original_v8_daily_five_doubles",
        "v8_v25_same_pair", "v28_upstream_capture", "source_capture_health", "v30_frozen_candidate_pair_rows", "v34_diagnosis",
        "v35_challengers", "v36_challengers", "v38_challengers", "v39_challengers", "v41_unified_final", "next_priority",
    )}
    payload = {
        "model": model,
        "max_output_tokens": 450,
        "store": False,
        "input": [
            {"role":"system", "content": (
                "You are a skeptical football-probability research reviewer. "
                "Recommend ONE specific new method for future-only pre-kickoff research, "
                "and ONE falsifiable validation test. No guarantees or betting advice. "
                "Never treat overlapping candidate pairs as independent wins; never "
                "claim that a model has hit 70% unless verified prospective results prove it. "
                "No generated code, shell commands, GitHub actions, or unsafe suggestions. "
                "Respond in at most 170 words."
            )},
            {"role":"user", "content": json.dumps(snapshot, sort_keys=True)},
        ],
    }
    response = requests.post(
        OPENAI_URL,
        headers={"Authorization":"Bearer "+key, "Content-Type":"application/json"},
        json=payload, timeout=35,
    )
    response.raise_for_status()
    data = response.json()
    answer = " ".join(item.get("text", "") for block in data.get("output", [])
        for item in block.get("content", [])
        if item.get("type") == "output_text" and isinstance(item.get("text"), str))
    answer = answer.strip()
    if not answer:
        raise ValueError("openai_api_returned_no_research_text")
    return {"model": model, "memo": answer[:2000], "source": "OpenAI_API_not_the_ChatGPT_website",
            "authorizes_code_execution": False, "authorizes_model_promotion": False}


def run(*, root=ROOT, now=None, use_ai=True, manual_retry=False):
    clock = now or datetime.now(timezone.utc)
    data = assemble(
        load("data/accuracy_diagnosis/report.json"),
        load("data/accuracy_diagnosis/v8_v25_matched.json"),
        load("data/v30_full_slate_ranking/status.json"),
        load("data/v34_score_error_research/status.json"),
        load("data/v35_continuous_challengers/status.json"),
        now=clock, v36=load("data/v36_prospective_challengers/status.json"),
        v28=load("data/v28_independent_pair_trial/status.json"),
        coverage=load("data/prospective_ledger/capture_coverage.json"),
        v37=load("data/v37_sparse_prospective/status.json"),
        v38=load("data/v38_prospective_challengers/status.json"),
        v39=load("data/v39_prospective_challengers/status.json"),
        v41=load("data/v41_unified_ensemble/performance.json"),
        v41_capture=load("data/v41_unified_ensemble/status.json"),
    )
    destination = Path(root)
    destination.mkdir(parents=True, exist_ok=True)
    day = clock.astimezone(timezone.utc).date().isoformat()
    note_path = destination / ("openai_research_" + day + ".json")
    attempt_path = destination / ("openai_attempt_" + day + ".json")
    retry_path = destination / ("openai_retry_" + day + ".json")
    prior_attempt = load(attempt_path) if attempt_path.is_file() else None
    # Only one explicit owner-requested retry for a failed 401 after key rotation.
    # Scheduled runs, failed 429s and repeat manual dispatches cannot bypass it.
    retry_allowed = (manual_retry is True and
                     isinstance(prior_attempt, dict) and
                     prior_attempt.get("status") == "api_http_401" and
                     not retry_path.exists())
    key = os.getenv("OPENAI_API_KEY", "")
    allowed = os.getenv("AI_RESEARCH_BOT_ENABLED", "").strip().lower() not in ("false", "0", "off", "no")
    if note_path.is_file():
        existing = load(note_path)
        if existing is None:
            data["ai_connection"] = "saved_memo_invalid"
        else:
            data["ai_connection"] = "daily_research_memo_saved"
            data["ai_memo"] = existing.get("memo", "")
    elif not use_ai:
        data["ai_connection"] = "test_mode_no_ai_call"
    elif not allowed:
        data["ai_connection"] = "AI_research_disabled_by_GitHub_variable"
    elif not key:
        data["ai_connection"] = "requires_OPENAI_API_KEY_GitHub_secret"
    elif attempt_path.is_file() and not retry_allowed:
        # Preserve the original attempt, but surface the more recent owner
        # retry's status if one has been saved.
        previous = load(retry_path) or prior_attempt or {}
        # Even errors count toward the one-per-day attempt budget. Repeated
        # runs may inspect reports, but cannot repeatedly charge the API.
        data["ai_connection"] = previous.get("status", "daily_api_attempt_already_recorded")
        if isinstance(previous.get("http_error_diagnostic"), dict):
            data["ai_http_error_diagnostic"] = previous["http_error_diagnostic"]
    else:
        model = (os.getenv("AI_RESEARCH_MODEL") or DEFAULT_AI_MODEL).strip()
        try:
            if model not in ("gpt-4o-mini", "gpt-4.1-mini", "gpt-5-mini"):
                raise ValueError("model_not_in_explicit_cost_bounded_allowlist")
            note = api_research_memo(data, key=key, model=model)
            note["date_utc"] = day
            note_path.write_text(json.dumps(note, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            data["ai_connection"] = "daily_research_memo_saved"
            data["ai_memo"] = note["memo"]
        except requests.HTTPError as exc:
            response = getattr(exc, "response", None)
            http_code = getattr(response, "status_code", None)
            data["ai_connection"] = "api_http_" + str(http_code if type(http_code) is int else "unknown")
            data["ai_http_error_diagnostic"] = classify_openai_http_error(response)
        except (requests.RequestException, ValueError, KeyError) as exc:
            # Do not publish secret-bearing request details or raw API content.
            data["ai_connection"] = ("api_request_failed_" +
                (str(exc)[:100] if isinstance(exc, ValueError) else type(exc).__name__))
        finally:
            # Keep the original once-daily receipt immutable. A manually
            # authorized 401 key-rotation test gets ONE separate receipt.
            receipt = retry_path if retry_allowed else attempt_path
            receipt.write_text(json.dumps({
                "date_utc": day, "attempted_at_utc": clock.isoformat(),
                "status": data.get("ai_connection", "no_api_result"),
                "model": model, "max_attempts_per_day": MAX_AI_CALLS_PER_UTC_DAY,
                "manual_key_rotation_retry": bool(retry_allowed),
                **({"http_error_diagnostic": data["ai_http_error_diagnostic"]}
                   if "ai_http_error_diagnostic" in data else {}),
            }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    status = destination / "status.json"
    status.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n",
                      encoding="utf-8")
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-ai", action="store_true")
    parser.add_argument("--retry-after-key-rotation", action="store_true",
                        help="One manual retry only after a recorded 401; never for 429 or scheduled runs")
    args = parser.parse_args()
    result = run(use_ai=not args.no_ai,
                 manual_retry=args.retry_after_key_rotation)
    print(json.dumps({
        "target":result["target_five_double_hit_rate"],
        "verified_v8":result["original_v8_daily_five_doubles"]["verified_sets"],
        "challenger_verified_pairs":result["v35_challengers"]["verified_same_pair_days"],
        "ai_connection":result["ai_connection"],
        "integrity":result["integrity"],
    }, sort_keys=True))
