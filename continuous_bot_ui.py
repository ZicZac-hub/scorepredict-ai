"""Read-only Streamlit dashboard for the scheduled five-double research bot."""
import json
from pathlib import Path


STATUS = Path("data/autobot/status.json")
ACTIONS = "https://github.com/ZicZac-hub/scorepredict-ai/actions/workflows/continuous-five-double-bot.yml"
REPO_SETTINGS = "https://github.com/ZicZac-hub/scorepredict-ai/settings/secrets/actions"


def _read():
    try:
        return json.loads(STATUS.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None


def _pct(x):
    return "—" if x is None else f"{x*100:.1f}%"


def render(st):
    st.subheader("🤖 24/7 Five-Double Research Bot")
    st.caption("Automated scheduled research · immutable pre-kickoff forecasts · independently verified full-time results")
    status = _read()
    if not status:
        st.warning("The bot has been added, but its first GitHub Actions status has not been published yet.")
        st.link_button("Open the bot workflow", ACTIONS)
        return
    st.caption("Latest saved run: " + status.get("last_run_utc", "unavailable") + " (UTC)")
    st.info("GitHub Actions runs the bot at scheduled intervals, not as a permanently running Streamlit process. Schedules depend on GitHub service availability, quotas, and repository settings.")
    baseline = status.get("original_v8_daily_five_doubles") or {}
    v25 = status.get("v8_v25_same_pair") or {}
    v35 = status.get("v35_challengers") or {}
    cols = st.columns(3)
    cols[0].metric("V8 verified five-double", f"{baseline.get('hits',0)}/{baseline.get('verified_sets',0)}", _pct(baseline.get("hit_rate")))
    cols[1].metric("V25 same-pair wins", f"{v25.get('v25_wins',0)}/{v25.get('verified_same_pair_sets',0)}")
    cols[2].metric("V35 frozen trial days", str(v35.get("frozen_days",0)))
    st.markdown("**Main target: 70% independently verified five-double hit rate**")
    st.progress(min(1.0, max(0.0, float(baseline.get("hit_rate") or 0)/.70)), text="V8 observed progress versus target (not a probability or guarantee)")
    st.caption("A model cannot claim 70% by treating pending results as losses, choosing different fixture pairs after the result, or counting overlapping candidate pairs as separate successful days.")
    st.markdown("#### Automated model comparisons")
    rows = []
    for name, result in (v35.get("methods") or {}).items():
        rows.append({
            "Method": name.replace("_"," "),
            "Finished same-pair days": int(result.get("verified_days") or 0),
            "Five-double hits": int(result.get("hits") or 0),
            "Hit rate": _pct(result.get("observed_hit_rate")),
            "Review gate": "Eligible" if result.get("review_gate_passed") else "Not met",
        })
    if rows:
        st.dataframe(rows, hide_index=True, use_container_width=True)
    else:
        st.caption("No qualified V35 pairs have been frozen and verified yet. Missing results are not failures.")
    st.write("**Next research priority:** " + str(status.get("next_priority", "Collect verified, frozen outcomes.")))
    st.caption("Source-integrity status: " + str(status.get("integrity", "Unknown")))
    st.caption("OpenAI research connection: " + str(status.get("ai_connection", "Not configured")))
    if status.get("ai_connection") == "api_http_401":
        st.warning("The last saved OpenAI authentication test returned 401. If the API key was updated after that test, use the bot workflow's Run workflow option with 'retry_openai_after_key_rotation' checked. Only one manual retry is allowed for a recorded 401 each UTC day.")
        st.caption("Never paste your API key into the website or this chat. A new key is managed in GitHub Actions secrets, not in the bot dashboard.")
    if status.get("ai_memo"):
        with st.expander("Daily ChatGPT/OpenAI research analysis", expanded=False):
            st.write(status["ai_memo"])
            st.caption("This is a text-only hypothesis. AI never changes code, historic scores, verified results, or production model weights.")
    if v35.get("ready_for_separate_production_review"):
        st.success("A prospective research method has met the preliminary statistical gate. Independent production review is still required.")
    c1, c2 = st.columns(2)
    with c1:
        st.link_button("View bot runs", ACTIONS, use_container_width=True)
    with c2:
        st.link_button("Configure API secrets", REPO_SETTINGS, use_container_width=True)
    st.caption("Optional AI connection uses OPENAI_API_KEY in GitHub Actions secrets. Set AI_RESEARCH_BOT_ENABLED=false in GitHub variables to disable paid calls. Usage is capped at one small API memo per UTC day; API charges are separate from a ChatGPT subscription.")
