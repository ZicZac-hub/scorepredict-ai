"""Optional ten-provider pre-kickoff research panel; NEVER changes production scores.

Uses identical saved evidence for ten independent AI adapters. No provider
is connected without its own API credentials.
All evaluations require verified regulation-time final scores.
"""
import json
import math
import os
import re
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlparse

import requests

from persist_prematch import load_json, verified_result, write_json
from v10_prospective import ROOT as V10_ROOT, read_latest, utc

ROOT = Path("data/llm_research")
PANEL_DIR = ROOT / "forecasts"
PROVIDERS = ("openai", "claude", "gemini", "grok", "deepseek",
             "perplexity", "mistral", "qwen", "groqcloud", "cohere")
SECRETS = {"openai": "OPENAI", "claude": "ANTHROPIC",
           "gemini": "GEMINI", "grok": "XAI", "deepseek": "DEEPSEEK",
           "perplexity": "PERPLEXITY", "mistral": "MISTRAL",
           "qwen": "QWEN", "groqcloud": "GROQ", "cohere": "COHERE"}
MAX_MATCHES_PER_RUN = 1
MIN_LEAD = timedelta(minutes=45)
MAX_LEAD = timedelta(hours=36)


def settings(environ=None):
    """Credentials are read from secrets/env only. Never stored in a record."""
    env = os.environ if environ is None else environ
    result = {}
    for name in PROVIDERS:
        prefix = SECRETS[name]
        key, model = env.get(prefix + "_API_KEY", ""), env.get(prefix + "_MODEL", "")
        if key and model:
            result[name] = (key, model)
    return result


def eligible(row, now):
    """All inputs must have been genuinely frozen before the match."""
    try:
        kickoff, recorded = utc(row["kickoff_utc"]), utc(row["captured_at_utc"])
        if (row.get("provider") != "API-Football" or
                row.get("model_key") not in ("v8", "v8_near") or
                not row.get("input_digest") or not row.get("forecast_id")):
            return False
        if not recorded <= now < kickoff or recorded >= kickoff:
            return False
        if not MIN_LEAD < kickoff-now <= MAX_LEAD:
            return False
        return bool(row.get("expected_goals") and row.get("top10_scores"))
    except (KeyError, TypeError, ValueError):
        return False


def panel_path(row, root=ROOT):
    provider = re.sub(r"[^a-zA-Z0-9_-]", "_", row["provider"])
    identifier = re.sub(r"[^a-zA-Z0-9_-]", "_", str(row["fixture_id"]))
    digest = re.sub(r"[^a-fA-F0-9]", "", str(row["input_digest"]))
    if len(digest) < 16:
        raise ValueError("Input digest must identify the frozen evidence")
    return Path(root) / "forecasts" / (provider + "_" + identifier + "_" + digest + ".json")


def briefing(row):
    """Identical source packet: no invented injuries, bookmaker values or xG."""
    info = {
        "home_team": row["home_team"], "away_team": row["away_team"],
        "league": row.get("competition"), "kickoff_utc": row["kickoff_utc"],
        "historical_matches": row.get("history_matches"),
        "expected_goals_model_estimate_not_true_xG": row.get("expected_goals"),
        "probabilities_from_frozen_statistical_model": row.get("outcome_probabilities"),
        "top_exact_score_options": row.get("top10_scores", [])[:5],
        "available_schedule_and_form_context": row.get("research_context", {}),
        "unavailable_features": row.get("unavailable_features", []),
    }
    return (
        "Independently assess this football fixture using ONLY these frozen "
        "pre-match facts. Do not claim access to external browsing, real xG, "
        "injuries, confirmed lineups or odds not in the packet. Distinguish "
        "the prior statistical forecast from your own judgement. Return ONLY "
        "JSON with keys home_win, draw, away_win (probabilities summing to 1), "
        "exact_home, exact_away (nonnegative integer goals), factors (up to 3 "
        "short points), and uncertainty (short string). No guarantees.\n"
        + json.dumps(info, ensure_ascii=True, sort_keys=True, default=str)
    )


def _extract(response, name):
    """Accept documented provider response shapes; require JSON, no prose guesses."""
    payload = response.json()
    if not isinstance(payload, dict):
        raise ValueError("Provider response must be an object")
    if name == "perplexity":
        text = "".join(part.get("text", "") for item in payload.get("output", [])
                       if isinstance(item, dict) and item.get("type") == "message"
                       for part in item.get("content", [])
                       if isinstance(part, dict) and part.get("type") == "output_text")
    elif name == "cohere":
        text = "".join(part.get("text", "") for part in
                       (payload.get("message") or {}).get("content", [])
                       if isinstance(part, dict) and part.get("type") == "text")
    elif name == "claude":
        parts = payload.get("content") or []
        text = "".join(part.get("text", "") for part in parts
                       if isinstance(part, dict) and part.get("type") == "text")
    elif name == "gemini":
        parts = ((payload.get("candidates") or [{}])[0].get("content") or {}).get("parts") or []
        text = "".join(part.get("text", "") for part in parts if isinstance(part, dict))
    else:
        message = ((payload.get("choices") or [{}])[0].get("message") or {})
        answer = message.get("content") or ""
        # Some OpenAI-compatible providers represent message content as
        # separate text segments. Reasoning text is intentionally ignored.
        text = "".join(p.get("text", "") for p in answer if
                       isinstance(p, dict) and p.get("type") == "text") if isinstance(answer, list) else answer
    if not isinstance(text, str):
        raise ValueError("Provider answer is not plain text")
    text = text.strip()
    if text.startswith("```"):
        text = text[3:].lstrip()
        if text[:4].lower() == "json":
            text = text[4:].lstrip()
        if text.endswith("```"):
            text = text[:-3].rstrip()
    if not text or len(text) > 30000:
        raise ValueError("Empty or oversized AI answer")
    return json.loads(text)


def validate_vote(answer):
    """Discard malformed/out-of-range guesses rather than presenting them."""
    if not isinstance(answer, dict):
        raise ValueError("Response is not an object")
    probabilities = []
    for key in ("home_win", "draw", "away_win"):
        p = answer.get(key)
        if type(p) not in (int, float) or not math.isfinite(p) or not 0 <= p <= 1:
            raise ValueError("Invalid outcome probability")
        probabilities.append(float(p))
    if abs(sum(probabilities)-1) > .025:
        raise ValueError("Model probabilities do not sum to one")
    h, a = answer.get("exact_home"), answer.get("exact_away")
    if type(h) is not int or type(a) is not int or not 0 <= h <= 9 or not 0 <= a <= 9:
        raise ValueError("Invalid exact score")
    factors = answer.get("factors") or []
    if not isinstance(factors, list):
        factors = []
    return {
        "home": probabilities[0]/sum(probabilities),
        "draw": probabilities[1]/sum(probabilities),
        "away": probabilities[2]/sum(probabilities),
        "score": [h, a],
        "factors": [str(p)[:180] for p in factors[:3]],
        "uncertainty": str(answer.get("uncertainty") or "")[:220]
    }


def _qwen_url(configured):
    """Only Alibaba-controlled HTTPS endpoints; no secret-bearing URL redirects."""
    value = configured or "https://dashscope-intl.aliyuncs.com/compatible-mode/v1/chat/completions"
    parsed = urlparse(value)
    host = (parsed.hostname or "").lower()
    legacy = host in ("dashscope.aliyuncs.com", "dashscope-intl.aliyuncs.com",
                      "dashscope-us.aliyuncs.com", "cn-hongkong.dashscope.aliyuncs.com")
    dedicated = host.endswith(".maas.aliyuncs.com") and bool(re.fullmatch(r"[a-z0-9.-]+", host))
    if (parsed.scheme != "https" or not (legacy or dedicated) or parsed.port not in (None, 443)
            or parsed.username or parsed.password or parsed.query or parsed.fragment
            or parsed.path != "/compatible-mode/v1/chat/completions"):
        raise ValueError("Qwen endpoint must be the official Alibaba compatible-mode chat URL")
    return value


def ask(name, key, model, prompt, request=None, endpoint=None):
    post = request or requests.post
    if not re.fullmatch(r"[A-Za-z0-9._/-]{1,120}", model) or ".." in model or "//" in model:
        raise ValueError("Invalid configured model identifier")
    if name == "perplexity":
        url = "https://api.perplexity.ai/v1/agent"
        headers = {"Authorization": "Bearer " + key}
        body = {"model": model, "input": prompt}
    elif name == "cohere":
        url = "https://api.cohere.com/v2/chat"
        headers = {"Authorization": "Bearer " + key}
        body = {"model": model, "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 800,
                "response_format": {"type": "json_object"}}
    elif name == "claude":
        url = "https://api.anthropic.com/v1/messages"
        headers = {"x-api-key": key, "anthropic-version": "2023-06-01"}
        body = {"model": model, "max_tokens": 650,
                "messages": [{"role": "user", "content": prompt}]}
    elif name == "gemini":
        url = "https://generativelanguage.googleapis.com/v1beta/models/" + model + ":generateContent"
        headers = {"x-goog-api-key": key}
        body = {"contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"maxOutputTokens": 650}}
    else:
        url = {"openai": "https://api.openai.com/v1/chat/completions",
               "grok": "https://api.x.ai/v1/chat/completions",
               "deepseek": "https://api.deepseek.com/chat/completions",
               "mistral": "https://api.mistral.ai/v1/chat/completions",
               "groqcloud": "https://api.groq.com/openai/v1/chat/completions",
               "qwen": None}[name]
        if name == "qwen":
            url = _qwen_url(endpoint)
        headers = {"Authorization": "Bearer " + key}
        body = {"model": model, "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 800}
        # JSON mode is supported by these APIs; schema validity is still
        # checked locally before freezing any pre-match vote.
        if name in ("openai", "grok", "groqcloud", "mistral"):
            body["response_format"] = {"type": "json_object"}
        if name == "groqcloud" and model.startswith("openai/gpt-oss-"):
            # GPT-OSS may otherwise spend the output budget on reasoning
            # and return no JSON answer. Only set flags for supported models.
            body["reasoning_effort"] = "low"
            body["reasoning_format"] = "hidden"
            body["max_tokens"] = 1024
    response = post(url, headers=headers, json=body, timeout=25, allow_redirects=False)
    response.raise_for_status()
    return validate_vote(_extract(response, name))


def consensus(votes):
    """Averages outcome opinions; never masquerades as exact-score probability."""
    if not votes:
        return None
    probs = {key: sum(v[key] for v in votes.values())/len(votes)
             for key in ("home", "draw", "away")}
    winning = max(probs, key=probs.get)
    def aligned(v):
        h, a = v["score"]
        return ("home" if h > a else "away" if h < a else "draw") == winning
    candidates = [tuple(v["score"]) for v in votes.values() if aligned(v)]
    score = list(sorted(Counter(candidates).items(),
                        key=lambda item: (-item[1], item[0]))[0][0]) if candidates else None
    return {"outcome_probabilities": probs, "outcome": winning,
            "suggested_score": score, "provider_count": len(votes),
            "complete_five_provider_panel": all(name in votes for name in PROVIDERS[:5]),
            "complete_ten_provider_panel": all(name in votes for name in PROVIDERS),
            "note": "Independent AI interpretation of the same supplied evidence; "
                    "not measured exact-score accuracy or a substitute for the baseline."}


def classify_provider_error(exc):
    """Safe diagnostics: no credential, URL query, request body or provider text."""
    response = getattr(exc, "response", None)
    code = getattr(response, "status_code", None) if response is not None else None
    reason = {400:"invalid_request_or_model", 401:"invalid_or_expired_key",
              402:"billing_required", 403:"insufficient_permissions_or_plan",
              404:"model_or_endpoint_unavailable",
              408:"provider_timeout", 409:"provider_conflict",
              422:"invalid_request_or_model", 429:"quota_or_rate_limit"}.get(code)
    if code is not None:
        return {"type": type(exc).__name__, "http_status": code,
                "reason":reason or ("provider_server_error" if code >= 500 else "provider_http_error")}
    if isinstance(exc, requests.Timeout):
        return {"type":type(exc).__name__, "reason":"provider_timeout"}
    if isinstance(exc, json.JSONDecodeError):
        return {"type":type(exc).__name__, "reason":"invalid_json_response"}
    if isinstance(exc, ValueError):
        return {"type":type(exc).__name__, "reason":"response_format_or_model_configuration"}
    return {"type":type(exc).__name__, "reason":"provider_request_failed"}


def capture(now=None, *, environ=None, root=ROOT, source=V10_ROOT, request=None,
            max_matches=MAX_MATCHES_PER_RUN):
    clock = utc(now or datetime.now(timezone.utc))
    env = os.environ if environ is None else environ
    configured = settings(env)
    status = {"checked_at_utc": clock.isoformat(), "enabled": env.get("LLM_RESEARCH_ENABLED") == "true",
              "configured_providers": list(configured), "captured": 0, "attempted": 0,
              "provider_error_types": {}, "provider_error_details": {}, "reason": ""}
    if not status["enabled"] or not configured:
        status["reason"] = "disabled" if not status["enabled"] else "provider_keys_or_models_missing"
        write_json(Path(root)/"status.json", status)
        return status
    # An earlier independently frozen research panel must remain the first
    # published version. Never repeatedly bill new providers for that same
    # fixture just because its statistical evidence digest has changed.
    previously_frozen = set()
    for saved in (Path(root)/"forecasts").glob("*.json"):
        try:
            prior = load_json(saved, {})
            if (prior.get("research_only") is True
                    and utc(prior["captured_at_utc"]) < utc(prior["kickoff_utc"])):
                previously_frozen.add((str(prior["provider"]), str(prior["fixture_id"])))
        except (OSError,ValueError,TypeError,KeyError):
            continue
    rows = list(read_latest(source).values())
    # Choose one valid frozen model per fixture, preferring newer near-kickoff evidence.
    eligible_rows = {}
    for row in rows:
        if not eligible(row, clock):
            continue
        identity = (row["provider"], str(row["fixture_id"]))
        if identity in previously_frozen:
            continue
        old = eligible_rows.get(identity)
        if old is None or (row["model_key"] == "v8_near" and old["model_key"] != "v8_near"):
            eligible_rows[identity] = row
    rows = sorted(eligible_rows.values(), key=lambda r: (utc(r["kickoff_utc"]),str(r["fixture_id"])))
    for row in rows:
        path = panel_path(row, root)
        if path.exists():
            continue
        if status["attempted"] >= max(0, min(5, int(max_matches))):
            break
        status["attempted"] += 1
        prompt = briefing(row)
        votes, models = {}, {}
        for name, (key, model) in configured.items():
            try:
                votes[name] = ask(name, key, model, prompt, request=request,
                                  endpoint=env.get("QWEN_API_URL") if name == "qwen" else None)
                models[name] = model
            except (requests.RequestException, ValueError, KeyError, TypeError, RuntimeError) as exc:
                status["provider_error_types"][name] = type(exc).__name__
                status["provider_error_details"][name] = classify_provider_error(exc)
        frozen = datetime.now(timezone.utc)
        if not votes or frozen >= utc(row["kickoff_utc"]):
            continue
        record = {"schema_version": 1, "research_only": True,
                  "provider": row["provider"], "fixture_id": str(row["fixture_id"]),
                  "home_team": row["home_team"], "away_team": row["away_team"],
                  "home_team_id": str(row["home_team_id"]),
                  "away_team_id": str(row["away_team_id"]),
                  "kickoff_utc": row["kickoff_utc"], "score_basis": "regulation",
                  "source_forecast_id": row["forecast_id"],
                  "source_model": row["model_key"],
                  "source_captured_at_utc": row["captured_at_utc"],
                  "input_digest": row["input_digest"],
                  "captured_at_utc": frozen.isoformat(), "models": models,
                  "votes": votes, "consensus": consensus(votes),
                  "baseline_predicted_score": row["predicted_score"],
                  "baseline_probabilities": row["outcome_probabilities"],
                  "provider_count": len(votes),
                  "configured_provider_count": len(configured),
                  "disclaimer": "No real-time web access by LLMs; source packet is frozen "
                                "statistical evidence. Shadow-only and not verified as superior."}
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with path.open("x", encoding="utf-8") as file:
                file.write(json.dumps(record,sort_keys=True,indent=2)+"\n")
            status["captured"] += 1
        except FileExistsError:
            pass
    status["reason"] = "frozen_or_no_eligible_upcoming_fixtures"
    write_json(Path(root)/"status.json",status)
    return status


def evaluate(root=ROOT, results_path=V10_ROOT/"verified_results.json"):
    """Compare five- and ten-provider panels on genuinely matched completed fixtures."""
    results = load_json(Path(results_path), {})
    stats = {size: {"complete": 0, "paired": 0, "panel_exact": 0,
                    "baseline_exact": 0, "panel_outcome": 0,
                    "baseline_outcome": 0}
             for size in ("five", "ten")}
    seen = set()
    # First historically frozen panel per fixture; never select a later
    # favourable vote after seeing the final result.
    entries = [load_json(path, {}) for path in (Path(root)/"forecasts").glob("*.json")]
    entries = [row for row in entries if row.get("captured_at_utc")]
    entries.sort(key=lambda row: utc(row["captured_at_utc"]))
    for row in entries:
        fixture_id = str(row.get("fixture_id"))
        identity = (row.get("provider"), fixture_id)
        if identity in seen:
            continue
        actual = results.get(fixture_id)
        if not verified_result(row, actual, fixture_id):
            continue
        if utc(row["captured_at_utc"]) >= utc(row["kickoff_utc"]):
            continue
        seen.add(identity)
        votes = row.get("votes") or {}
        home, away = actual["home_goals"], actual["away_goals"]
        actual_score = [home,away]
        outcome = "home" if home > away else "away" if home < away else "draw"
        baseline = row.get("baseline_probabilities") or {}
        for size,names in (("five",PROVIDERS[:5]), ("ten",PROVIDERS)):
            if not all(name in votes for name in names):
                continue
            item = stats[size]
            item["complete"] += 1
            panel = consensus({name:votes[name] for name in names})
            if panel is None or panel["suggested_score"] is None:
                continue
            if not all(k in baseline for k in ("home","draw","away")):
                continue
            item["paired"] += 1
            item["panel_exact"] += int(panel["suggested_score"] == actual_score)
            item["baseline_exact"] += int(row.get("baseline_predicted_score") == actual_score)
            item["panel_outcome"] += int(panel["outcome"] == outcome)
            item["baseline_outcome"] += int(max(baseline, key=baseline.get) == outcome)
    five,ten = stats["five"],stats["ten"]
    report = {
        # Stable legacy keys measure the original first five providers only.
        "complete_panels_with_verified_results":five["complete"],
        "paired_verified_scores":five["paired"],
        "panel_exact_hits":five["panel_exact"],
        "baseline_exact_hits":five["baseline_exact"],
        "panel_outcome_hits":five["panel_outcome"],
        "baseline_outcome_hits":five["baseline_outcome"],
        "panel_exact_rate":five["panel_exact"]/five["paired"] if five["paired"] else None,
        "baseline_exact_rate":five["baseline_exact"]/five["paired"] if five["paired"] else None,
        "ten_provider_complete_verified":ten["complete"],
        "ten_provider_paired_verified":ten["paired"],
        "ten_provider_exact_hits":ten["panel_exact"],
        "ten_provider_baseline_exact_hits":ten["baseline_exact"],
        "ten_provider_outcome_hits":ten["panel_outcome"],
        "ten_provider_baseline_outcome_hits":ten["baseline_outcome"],
        "ten_provider_exact_rate":ten["panel_exact"]/ten["paired"] if ten["paired"] else None,
        "ten_provider_baseline_exact_rate":ten["baseline_exact"]/ten["paired"] if ten["paired"] else None,
        "note": "First five and all ten are measured separately on complete, earlier-frozen "
                "panels for the same verified 90-minute fixtures. Not a live model."
    }
    write_json(Path(root)/"evaluation.json",report)
    return report


def main():
    status = capture()
    report = evaluate()
    print(json.dumps({"status": status, "evaluation": report},indent=2))


if __name__ == "__main__":
    main()
