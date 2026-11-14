"""Safe, session-only external AI research for the selected Deep Research fixture.

The GitHub Actions secret store is NOT available to a Streamlit server.
This module reads server-side Streamlit secrets/env only; never exposes keys.
One explicit button click runs a bounded, private, pre-kickoff research batch.
No immutable GitHub prediction is rewritten, and these unarchived opinions
are NEVER scored as verified prospective accuracy.
"""
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone

from llm_research_panel import (
    PROVIDERS, SECRETS, ask, classify_provider_error, consensus, settings
)

MIN_LEAD = timedelta(minutes=5)
MAX_PROVIDERS = 10


def utc(value):
    try:
        if isinstance(value, datetime):
            dt = value
        else:
            dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if dt.tzinfo is None:
            return None
        return dt.astimezone(timezone.utc)
    except (ValueError, TypeError):
        return None


def fixture_key(fixture):
    provider = str(fixture.get("provider") or "football-data.org")
    fixture_id = fixture.get("id")
    kickoff = utc(fixture.get("utcDate"))
    if fixture_id is None or kickoff is None:
        return None
    return (provider, str(fixture_id), kickoff.isoformat())


def _secret(st, name, environ):
    try:
        value = st.secrets.get(name, "")
    except (AttributeError, OSError, FileNotFoundError, KeyError, RuntimeError):
        value = ""
    return str(value or environ.get(name, "")).strip()


def server_settings(st, environ=None):
    """Names and keys only from the app server, NEVER from the public browser."""
    env = os.environ if environ is None else environ
    collected = {}
    for provider in PROVIDERS:
        prefix = SECRETS[provider]
        collected[prefix + "_API_KEY"] = _secret(st, prefix + "_API_KEY", env)
        collected[prefix + "_MODEL"] = _secret(st, prefix + "_MODEL", env)
    return settings(collected)


def server_qwen_url(st, environ=None):
    return _secret(st, "QWEN_API_URL", os.environ if environ is None else environ) or None


def _scores(stats, fallback):
    """Only observed model outputs; never invent a missing score probability."""
    result = []
    raw = stats.get("scores") if isinstance(stats, dict) else None
    if not raw and isinstance(fallback, dict):
        raw = fallback.get("scores")
    for item in (raw or [])[:6]:
        try:
            h,a,p = item[:3]
            if type(h) is int and type(a) is int and h >= 0 and a >= 0:
                p = float(p)
                if 0 <= p <= 1:
                    result.append([h,a,round(p,6)])
        except (TypeError,ValueError,IndexError):
            continue
    return result


def _verified_history(fixture, history, now):
    kickoff = utc(fixture.get("utcDate"))
    team_ids = {str((fixture.get(side+"Team") or {}).get("id"))
                for side in ("home","away")}
    team_ids.discard("None")
    seen, matches = set(), []
    for m in history or []:
        when = utc(m.get("utcDate"))
        if (m.get("status") != "FINISHED" or when is None or
                kickoff is None or when >= min(now, kickoff)):
            continue
        home = m.get("homeTeam") or {}
        away = m.get("awayTeam") or {}
        if team_ids and not ({str(home.get("id")), str(away.get("id"))} & team_ids):
            continue
        score = m.get("score") or {}
        # Some feeds use fullTime for 120-minute or penalty results.
        # Use explicitly saved regulation-time goals or skip the sample.
        if score.get("duration") in ("EXTRA_TIME","PENALTY_SHOOTOUT"):
            goals = score.get("regularTime") or {}
        else:
            goals = score.get("fullTime") or {}
        h,a = goals.get("home"), goals.get("away")
        if type(h) is not int or type(a) is not int or min(h,a)<0:
            continue
        identity = (when.isoformat(), str(home.get("id")), str(away.get("id")))
        if identity in seen:
            continue
        seen.add(identity)
        matches.append((when, {
            "date_utc": when.isoformat(),
            "home": str(home.get("name") or "")[:100],
            "away": str(away.get("name") or "")[:100],
            "full_time_90min": [h,a]
        }))
    matches.sort(key=lambda value: value[0], reverse=True)
    return [value for _,value in matches[:20]]


def briefing(fixture, history, statistical, fallback=None, *, now=None):
    """Identical limited evidence to all AI providers, strictly as available now."""
    clock = utc(now or datetime.now(timezone.utc))
    kickoff = utc(fixture.get("utcDate"))
    if clock is None or kickoff is None or kickoff - clock <= MIN_LEAD:
        raise ValueError("Match is too close to kickoff or already started")
    first = fixture.get("homeTeam") or {}
    second = fixture.get("awayTeam") or {}
    if not first.get("name") or not second.get("name") or fixture_key(fixture) is None:
        raise ValueError("Selected match identity is incomplete")
    result = statistical if isinstance(statistical, dict) else {}
    payload = {
        "source": "ZicZac selected-match on-demand research",
        "captured_evidence_at_utc": clock.isoformat(),
        "fixture_provider": fixture_key(fixture)[0],
        "fixture_id": fixture_key(fixture)[1],
        "kickoff_utc": kickoff.isoformat(),
        "home_team": str(first["name"])[:120],
        "away_team": str(second["name"])[:120],
        "competition": str((fixture.get("competition") or {}).get("name") or "")[:120],
        "statistical_model_version": str(result.get("model_version") or
                                         (fallback or {}).get("model_version") or "unavailable"),
        "statistical_score_estimates": _scores(result, fallback),
        "statistical_1x2": result.get("outcome_probabilities") or {},
        "model_expected_goals_not_verified_xg": {
            "home": result.get("home_xg"), "away": result.get("away_xg")
        },
        "completed_historical_matches": _verified_history(fixture, history, clock),
        "not_provided": ["confirmed lineups", "verified injuries", "live bookmaker odds",
                         "true event-level xG", "live web browsing"]
    }
    return ("You are one independent football research reviewer. Use ONLY the dated "
            "facts provided, not assumed injuries, up-to-the-minute web news, or "
            "fabricated player information. Statistical model estimates are context "
            "rather than proof. Return ONLY JSON with keys home_win, draw, away_win "
            "(numeric probabilities summing to 1), exact_home, exact_away "
            "(nonnegative whole goals), factors (short list), uncertainty (string). "
            "An exact-score opinion is NOT a guarantee or calibrated hit rate.\n"
            + json.dumps(payload, sort_keys=True, default=str))


def run_selected(fixture, history, statistical, fallback, provider_configs, *,
                 now=None, request=None, qwen_url=None):
    """One bounded research batch, isolated provider errors and no persistence."""
    clock = utc(now or datetime.now(timezone.utc))
    match = fixture_key(fixture)
    if match is None:
        return {"status": "invalid_fixture", "votes": {}, "errors": {}}
    if clock is None or utc(fixture["utcDate"]) - clock <= MIN_LEAD:
        return {"status": "kickoff_too_close", "votes": {}, "errors": {},
                "fixture": match}
    if not provider_configs:
        return {"status": "no_streamlit_api_credentials", "votes": {}, "errors": {},
                "fixture": match}
    prompt = briefing(fixture, history, statistical, fallback, now=clock)
    configured = [(name, provider_configs[name]) for name in PROVIDERS
                  if name in provider_configs][:MAX_PROVIDERS]
    votes, errors = {}, {}
    def one(name, pair):
        key, model = pair
        return ask(name, key, model, prompt, request=request,
                   endpoint=qwen_url if name=="qwen" else None)
    with ThreadPoolExecutor(max_workers=min(4,len(configured))) as executor:
        jobs = {executor.submit(one, name, creds):name for name,creds in configured}
        for future in as_completed(jobs):
            name=jobs[future]
            try:
                votes[name]=future.result()
            except Exception as exc:
                # Neither response bodies nor exception messages nor API keys
                # are ever saved or displayed.
                errors[name]=classify_provider_error(exc)
    completed=datetime.now(timezone.utc)
    return {
        "status": "completed" if completed < utc(fixture["utcDate"])
                  else "completed_after_kickoff_not_prospective",
        "fixture": match,
        "started_at_utc": clock.isoformat(),
        "completed_at_utc": completed.isoformat(),
        "attempted": [name for name,_ in configured],
        "votes": {k:votes[k] for k in PROVIDERS if k in votes},
        "errors": {k:errors[k] for k in PROVIDERS if k in errors},
        "consensus": consensus(votes),
        "ephemeral": True,
        "research_only": True,
        "original_predictions_overwritten": False,
        "evaluated_as_frozen_five_double": False
    }


def view(st, result):
    """Readable mobile presentation; do not present session-only votes as wins."""
    st.markdown("#### 🧠 External AI opinions for the selected match")
    state=result.get("status")
    if state=="no_streamlit_api_credentials":
        st.warning("No external AI API keys and model IDs are installed in Streamlit "
                   "Secrets. GitHub Actions secrets are separate from the Streamlit "
                   "website. Configure them securely in Streamlit Cloud.")
        return
    if state=="private_access_required":
        st.warning("External paid AI calls require the private ZicZac password gate. "
                   "Set ZICZAC_ACCESS_PASSWORD in Streamlit Secrets first.")
        return
    if state=="kickoff_too_close":
        st.info("No new AI opinions requested: this match has started or is within "
                "five minutes of kickoff.")
        return
    if state=="invalid_fixture":
        st.warning("Selected match has no verified fixture ID or kickoff.")
        return
    if state=="request_failed":
        st.warning("This on-demand research request could not complete. "
                   "Please check the server configuration and latest tests. "
                   "API keys are never displayed.")
        return
    if state=="disabled_by_owner":
        st.caption("External API research was not requested. Statistical research "
                   "above is still available without additional AI charges.")
        return
    if state=="already_requested":
        st.caption("Showing this session's earlier AI responses; no repeat paid calls.")
    if state=="completed_after_kickoff_not_prospective":
        st.warning("Responses completed after kickoff. They are NOT eligible as "
                   "frozen pre-match predictions or verified win-rate evidence.")
    st.caption("Session-only external research · "+
               str(len(result.get("votes") or {}))+"/"+str(len(result.get("attempted") or []))+
               " configured providers replied. It does not modify the original score.")
    for name,vote in (result.get("votes") or {}).items():
        score=vote.get("score") or []
        if len(score)!=2:
            continue
        st.write("**"+name.title()+"** · "+str(score[0])+"–"+str(score[1])+
                 " · H/D/A "+"/".join(f"{vote.get(k,0):.0%}"
                                   for k in ("home","draw","away")))
        facts=vote.get("factors") or []
        if facts:
            st.caption("; ".join(str(x)[:170] for x in facts[:2]))
    panel=result.get("consensus") or {}
    if panel:
        guess=panel.get("suggested_score")
        st.caption("Combined outcome: "+str(panel.get("outcome") or "unknown")+
                   ". Aligned exact-score consensus: "+
                   (str(guess[0])+"–"+str(guess[1]) if guess else "not established")+
                   ". Averaged 1X2 is not an exact-score probability.")
    for name,error in (result.get("errors") or {}).items():
        st.caption(name.title()+": "+str(error.get("reason") or "provider_error")+
                   (" (HTTP "+str(error["http_status"])+")"
                    if error.get("http_status") is not None else ""))
    if result.get("errors"):
        st.info("Provider errors may require a supported model ID, account credits, "
                "or API permissions. No key values or provider response bodies are shown.")
    st.caption("External model calls do not browse live football websites. "
               "These unarchived opinions are excluded from ZicZac's immutable "
               "pre-match accuracy and five-double win-rate statistics.")
