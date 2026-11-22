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


def assemble(v8, v25, v30, v34, v35, *, now):
    """Only preexisting independently validated archived report totals."""
    baseline = (v8 or {}).get("observed") or {}
    matched_v8 = (v25 or {}).get("v8") or {}
    matched_v25 = (v25 or {}).get("v25") or {}
    v35 = v35 or {}
    stats = v35.get("methods") or {}
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
    next_priority = (
        "Investigate why original score distributions miss one or both final "
        "leg scores; prioritize independent high-goal calibration trials."
        if both_absent else
        "Collect enough independent, frozen pre-kickoff same-pair results "
        "before comparing or promoting models."
    )
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
        "next_priority": next_priority,
        "production_model_changed_by_bot": False,
        "ai_research_memo_is_non_executable": True,
        "no_claimed_70_percent_without_independent_evidence": True,
    }


def api_research_memo(report, *, key, model=DEFAULT_AI_MODEL):
    """One bounded optional API request: recommendations are TEXT ONLY."""
    snapshot = {key: report[key] for key in (
        "target_five_double_hit_rate", "integrity", "original_v8_daily_five_doubles",
        "v8_v25_same_pair", "v30_frozen_candidate_pair_rows", "v34_diagnosis",
        "v35_challengers", "next_priority",
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
        now=clock,
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
