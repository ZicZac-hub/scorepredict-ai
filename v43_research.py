"""External research for the V43 score, using the existing provider adapters."""

from datetime import datetime, timezone
from hashlib import sha256

from llm_research_panel import PROVIDERS
from on_demand_ai import (MIN_LEAD, fixture_key, frozen_for_fixture,
                          private_gate_ready, run_selected, server_qwen_url,
                          server_settings, show_frozen, utc)


DISPLAY_NAMES = {"openai": "OpenAI", "claude": "Claude", "gemini": "Gemini",
                 "grok": "Grok", "deepseek": "DeepSeek", "perplexity": "Perplexity",
                 "mistral": "Mistral", "qwen": "Qwen", "groqcloud": "GroqCloud",
                 "cohere": "Cohere"}


def provider_status(st, environ=None):
    """Report available server connections without disclosing settings."""
    permitted = private_gate_ready(st, environ=environ)
    configs = server_settings(st, environ=environ) if permitted else {}
    return {
        "private_access": permitted,
        "providers": {
            name: ("ready" if name in configs else "not_configured"
                   if permitted else "access_required")
            for name in PROVIDERS
        },
    }


def run_v43_research(st, fixture, history, statistical, *, team_context=None,
                     now=None, environ=None, request=None):
    """Return one common-evidence batch of independent advisory opinions."""
    empty = {"votes": {}, "errors": {}, "attempted": [], "research_only": True,
             "ephemeral": True, "ai_numeric_weight": 0.0}
    if not private_gate_ready(st, environ=environ):
        return {**empty, "status": "private_access_required"}
    configs = server_settings(st, environ=environ)
    try:
        result = run_selected(fixture, history, statistical, {}, configs,
                              now=now, request=request,
                              qwen_url=server_qwen_url(st, environ=environ),
                              team_context=team_context)
    except (AttributeError, KeyError, TypeError, ValueError, OverflowError):
        return {**empty, "status": "invalid_research_evidence"}

    # Provider text is advisory and can echo unexpected input. Keep the
    # normalized numerical opinion and suppress any credential echoed in prose.
    keys = [key for key, _ in configs.values()]
    for vote in result["votes"].values():
        vote["factors"] = [factor for factor in vote["factors"]
                           if not any(key in factor for key in keys)]
        if any(key in vote["uncertainty"] for key in keys):
            vote["uncertainty"] = ""

    if result["status"] == "completed" and not result["votes"]:
        result["status"] = "providers_failed"
    return {**empty, **result}


def render_v43_research(st, fixture, history, statistical, *, team_context=None):
    """Show prior opinions and the single-click authenticated research action."""
    st.markdown("#### External AI match research")
    availability = provider_status(st)
    ready = [DISPLAY_NAMES[name] for name, status in availability["providers"].items()
             if status == "ready"]
    if not availability["private_access"]:
        st.caption("Sign in to Private Access to run OpenAI, Claude or other configured AI providers.")
    else:
        st.caption("Ready to request · " + (", ".join(ready) or "None"))
        missing = [DISPLAY_NAMES[name] for name in ("openai", "claude")
                   if availability["providers"][name] == "not_configured"]
        if missing:
            st.caption("Missing a Streamlit API key or model ID · " + ", ".join(missing))

    match = fixture_key(fixture) if fixture else None
    if match is None:
        st.caption("Select an upcoming match to view or request match-specific research.")
        return
    identity = (*match, str((fixture.get("homeTeam") or {}).get("id")),
                str((fixture.get("awayTeam") or {}).get("id")))

    saved = frozen_for_fixture(fixture)
    if saved:
        with st.expander("Earlier pre-match AI opinions"):
            show_frozen(st, saved)

    if not (availability["private_access"] and ready):
        return

    runs = st.session_state.setdefault("v43_research_runs", {})
    cached = runs.get(identity)
    if cached is None and utc(fixture["utcDate"]) - datetime.now(timezone.utc) <= MIN_LEAD:
        st.caption("New external research closes five minutes before kickoff.")
        return
    if cached is None:
        widget_suffix = sha256(repr(identity).encode()).hexdigest()[:16]
        allowed = st.checkbox("Allow external API calls for this match (charges may apply)",
                              key=f"v43_ai_consent_{widget_suffix}")
        if st.button("Run V43 AI research", disabled=not allowed,
                     key=f"v43_ai_run_{widget_suffix}") and allowed:
            with st.spinner("Requesting independent opinions from configured providers…"):
                cached = run_v43_research(st, fixture, history, statistical,
                                          team_context=team_context)
                runs[identity] = cached

    if cached is None:
        return
    attempted = cached["attempted"]
    votes = cached["votes"]
    if attempted:
        st.caption(f"{len(votes)}/{len(attempted)} providers returned opinions. "
                   "The statistical ensemble uses a 0% AI weight.")
    if cached["status"] == "completed_after_kickoff_not_prospective":
        st.warning("The providers finished after kickoff.")
    elif cached["status"] == "invalid_research_evidence":
        st.warning("Match identity or evidence could not be validated for external research.")
    elif cached["status"] == "kickoff_too_close":
        st.info("New external research closes five minutes before kickoff.")
    for name, vote in votes.items():
        home, away = vote["score"]
        outcome = " / ".join(f"{vote[key]:.0%}" for key in ("home", "draw", "away"))
        st.write(f"{DISPLAY_NAMES[name]} · {home}–{away} · Home / Draw / Away {outcome}")
        for factor in vote["factors"]:
            st.text(factor)
        if vote["uncertainty"]:
            st.text(vote["uncertainty"])
    for name, error in cached["errors"].items():
        st.caption(f'{DISPLAY_NAMES[name]} · {error["reason"].replace("_", " ")}')
    combined = (cached.get("consensus") or {}).get("suggested_score")
    if combined is not None and len(votes) >= 2:
        st.caption(f"Combined AI opinion · {combined[0]}–{combined[1]}")
