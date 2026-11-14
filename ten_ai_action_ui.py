"""Visible, match-specific Run & Predict controls for the Ten-AI research panel.

The Streamlit service is separate from scheduled GitHub Actions. Calls are
possible only with server-side Streamlit credentials and an authenticated
owner. Existing archived GitHub votes are read-only and cost nothing.
"""
from datetime import datetime, timezone

from on_demand_ai import (MIN_LEAD, fixture_key, frozen_for_fixture,
                          private_gate_ready, run_selected, server_qwen_url,
                          server_settings, show_frozen, utc, view)


def available(rows, *, now=None):
    """Only legitimate upcoming fixtures with a provider identity and teams."""
    clock = utc(now or datetime.now(timezone.utc))
    result = {}
    if clock is None:
        return result
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        match = row.get("fixture") or {}
        identity = fixture_key(match)
        if identity is None or utc(match["utcDate"]) - clock <= MIN_LEAD:
            continue
        home = match.get("homeTeam") or {}
        away = match.get("awayTeam") or {}
        if not home.get("name") or not away.get("name"):
            continue
        label = (str(home["name"]) + " vs " + str(away["name"]) + " · " +
                 utc(match["utcDate"]).strftime("%d %b %Y %H:%M UTC") +
                 " · " + str(identity[0]) + " #" + str(identity[1]))
        result[label] = row
    return result


def render(st, rows=None, histories=None, *, now=None):
    st.markdown("#### 🎯 Run & Predict · Choose a match")
    st.caption("Research the selected upcoming fixture with connected external AI "
               "models. Each provider makes an independent exact-score suggestion; "
               "these are not guaranteed results.")
    options = available(rows, now=now)
    if not options:
        st.info("No upcoming prediction-ready fixtures are available in your "
                "current league/date selection. Choose Tomorrow, Next 7 days, "
                "or Next 14 days above to find more.")
        return
    label = st.selectbox("Match for ten-AI research", list(options),
                         key="ziczac_ten_ai_match")
    row = options[label]
    fixture = row["fixture"]
    identity = fixture_key(fixture)
    if identity is None:
        st.warning("This match has no valid provider fixture identity.")
        return

    # Read-only published GitHub votes can be viewed with no API keys.
    saved = frozen_for_fixture(fixture)
    if saved:
        with st.expander("📁 Previously frozen AI prediction for this match",
                         expanded=False):
            show_frozen(st, saved)
    else:
        st.caption("No previously frozen GitHub ten-AI research matches this fixture.")

    configs = server_settings(st)
    permitted = private_gate_ready(st)
    if configs:
        st.caption("Streamlit AI providers ready to try: " +
                   ", ".join(name.title() for name in configs) +
                   " (" + str(len(configs)) + "/10).")
    else:
        st.warning("No AI provider credentials are configured in **Streamlit "
                   "Secrets**. The six provider keys you configured in GitHub "
                   "Actions are not automatically available to this website.")
    if not permitted:
        st.info("Before making paid AI requests, enable **Private Access** using "
                "ZICZAC_ACCESS_PASSWORD in Streamlit Secrets and sign in.")
    if not configs or not permitted:
        st.caption("See [on-demand AI setup instructions]"
                   "(https://github.com/ZicZac-hub/scorepredict-ai/blob/main/"
                   "ON_DEMAND_AI_SETUP.md). No API calls will be made until "
                   "the private setup is complete.")

    approved = st.checkbox(
        "Allow external AI API requests for this match (charges may apply)",
        value=False, key="ziczac_ten_ai_cost_consent")
    clicked = st.button("🧠 Run Ten-AI Research & Predict",
                        type="primary", use_container_width=True,
                        key="ziczac_ten_ai_run")
    prior = st.session_state.setdefault("ziczac_ten_ai_results", {})
    cached = prior.get(identity)
    if clicked:
        if cached is not None:
            st.info("Showing this session's previous AI answers. No extra paid "
                    "requests were made.")
        elif not permitted:
            st.warning("Sign in to the password-protected app first. The "
                       "Run button cannot use paid APIs on an unprotected site.")
        elif not configs:
            st.warning("Add AI API keys and matching model IDs to Streamlit "
                       "Cloud → Manage app → Settings → Secrets.")
        elif not approved:
            st.warning("Tick the API-charge consent box before requesting a "
                       "new AI prediction.")
        else:
            histories = histories or {}
            source = row.get("past") or histories.get(row.get("league"), [])
            provider = identity[0]
            known = [m for m in source if
                     str(m.get("provider") or "football-data.org") == provider]
            # No new inference is recorded as a prospective verified hit. A
            # completed batch is retained within this user's app session only.
            with st.spinner("Consulting connected AI models for this match..."):
                try:
                    response = run_selected(fixture, known, {},
                                            row.get("result") or {}, configs,
                                            qwen_url=server_qwen_url(st))
                except Exception:
                    response = {"status": "request_failed",
                                "votes": {}, "errors": {}}
            prior[identity] = response
            cached = response

    if cached is not None:
        view(st, cached)
    st.caption("New on-demand AI opinions remain separate from frozen "
               "pre-match accuracy results. The saved statistical model "
               "and five-double archive are unchanged.")
