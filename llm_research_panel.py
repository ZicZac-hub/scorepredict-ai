"""Optional five-provider pre-kickoff research panel; NEVER changes production scores.

Uses identical, saved evidence for independent OpenAI, Claude, Gemini, Grok
and DeepSeek analyses. No provider is connected without its own API credentials.
All evaluations require verified regulation-time final scores.
"""
import json
import math
import os
import re
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

from persist_prematch import load_json, verified_result, write_json
from v10_prospective import ROOT as V10_ROOT, read_latest, utc

ROOT = Path("data/llm_research")
PANEL_DIR = ROOT / "forecasts"
PROVIDERS = ("openai", "claude", "gemini", "grok", "deepseek")
MAX_MATCHES_PER_RUN = 2
MIN_LEAD = timedelta(minutes=45)
MAX_LEAD = timedelta(hours=36)


def settings(environ=None):
    """Credentials are read from secrets/env only. Never stored in a record."""
    env = os.environ if environ is None else environ
    result = {}
    for name in PROVIDERS:
        prefix = {"openai": "OPENAI", "claude": "ANTHROPIC",
                  "gemini": "GEMINI", "grok": "XAI", "deepseek": "DEEPSEEK"}[name]
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
    payload = response.json()
    if name == "claude":
        parts = payload.get("content") or []
        text = "".join(part.get("text", "") for part in parts if part.get("type") == "text")
    elif name == "gemini":
        parts = ((payload.get("candidates") or [{}])[0].get("content") or {}).get("parts") or []
        text = "".join(part.get("text", "") for part in parts)
    else:
        text = ((payload.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^\`\`\`(?:json)?\\s*|\\s*\`\`\`$", "", text, flags=re.I).strip()
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


def ask(name, key, model, prompt, request=None):
    post = request or requests.post
    if not re.fullmatch(r"[A-Za-z0-9._-]{1,100}", model):
        raise ValueError("Invalid configured model identifier")
    if name == "claude":
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
               "deepseek": "https://api.deepseek.com/chat/completions"}[name]
        headers = {"Authorization": "Bearer " + key}
        body = {"model": model, "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 650}
    response = post(url, headers=headers, json=body, timeout=40)
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
            "complete_five_provider_panel": len(votes) == 5,
            "note": "Independent AI interpretation of the same supplied evidence; "
                    "not measured exact-score accuracy or a substitute for the baseline."}


def capture(now=None, *, environ=None, root=ROOT, source=V10_ROOT, request=None,
            max_matches=MAX_MATCHES_PER_RUN):
    clock = utc(now or datetime.now(timezone.utc))
    env = os.environ if environ is None else environ
    configured = settings(env)
    status = {"checked_at_utc": clock.isoformat(), "enabled": env.get("LLM_RESEARCH_ENABLED") == "true",
              "configured_providers": list(configured), "captured": 0, "attempted": 0,
              "provider_error_types": {}, "reason": ""}
    if not status["enabled"] or not configured:
        status["reason"] = "disabled" if not status["enabled"] else "provider_keys_or_models_missing"
        write_json(Path(root)/"status.json", status)
        return status
    rows = list(read_latest(source).values())
    # Choose one valid frozen model per fixture, preferring newer near-kickoff evidence.
    eligible_rows = {}
    for row in rows:
        if not eligible(row, clock):
            continue
        identity = (row["provider"], str(row["fixture_id"]))
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
                votes[name] = ask(name, key, model, prompt, request=request)
                models[name] = model
            except (requests.RequestException, ValueError, KeyError, TypeError, RuntimeError) as exc:
                status["provider_error_types"][name] = type(exc).__name__
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
    """Observed comparisons only, with identity-matched final regulation scores."""
    results = load_json(Path(results_path), {})
    seen = set()
    complete = 0
    eligible_n = 0
    panel_exact = baseline_exact = panel_outcome = baseline_outcome = 0
    for path in sorted((Path(root)/"forecasts").glob("*.json")):
        row = load_json(path, {})
        fixture_id = str(row.get("fixture_id"))
        unique = (row.get("provider"), fixture_id)
        if unique in seen:
            continue
        actual = results.get(fixture_id)
        if not verified_result(row, actual, fixture_id):
            continue
        if utc(row["captured_at_utc"]) >= utc(row["kickoff_utc"]):
            continue
        seen.add(unique)
        consensus_row = row.get("consensus") or {}
        if not consensus_row.get("complete_five_provider_panel"):
            continue
        complete += 1
        home, away = actual["home_goals"],actual["away_goals"]
        score = consensus_row.get("suggested_score")
        if score is None:
            continue
        eligible_n += 1
        panel_exact += int(score == [home, away])
        baseline_exact += int(row.get("baseline_predicted_score") == [home, away])
        outcome = "home" if home > away else "away" if home < away else "draw"
        panel_outcome += int(consensus_row["outcome"] == outcome)
        baseline_outcome += int(max(row["baseline_probabilities"],key=row["baseline_probabilities"].get) == outcome)
    report = {"complete_panels_with_verified_results": complete,
              "paired_verified_scores": eligible_n,
              "panel_exact_hits": panel_exact, "baseline_exact_hits": baseline_exact,
              "panel_outcome_hits": panel_outcome, "baseline_outcome_hits": baseline_outcome,
              "panel_exact_rate": panel_exact/eligible_n if eligible_n else None,
              "baseline_exact_rate": baseline_exact/eligible_n if eligible_n else None,
              "note": "Same-fixture verified regulation scores only; partial panels and "
                      "panels without an aligned score are excluded. Not a production model."}
    write_json(Path(root)/"evaluation.json",report)
    return report


def main():
    status = capture()
    report = evaluate()
    print(json.dumps({"status": status, "evaluation": report},indent=2))


if __name__ == "__main__":
    main()
