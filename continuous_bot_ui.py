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
    v36 = status.get("v36_challengers") or {}
    v37 = status.get("v37_sparse_uefa_research") or {}
    v28 = status.get("v28_upstream_capture") or {}
    source = status.get("source_capture_health") or {}
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
    st.markdown("#### Pre-match V8 source capture readiness")
    st.caption("Source capture is separate from the research bot. Missing/old reports cannot prove the collector ran or that forecasts improved.")
    freshness=source.get("collector_freshness") or "unknown_missing_capture_timestamp"
    if freshness == "stale":
        st.warning("Pre-match source-capture report is over three hours old; check the separate capture workflow before trusting source coverage.")
    elif freshness != "recent":
        st.warning("Latest source-capture time is unavailable or invalid. Capture freshness is unverified.")
    else:
        st.caption("Verified timestamp for latest source report: " +
                   str(source.get("collector_started_at_utc")))
    st.write("Processed fixtures: " + str(source.get("processed_fixtures",0)) +
             " · V8 forecasts saved: " + str(source.get("v8_saved",0)) +
             " · Explicit V8 abstentions: " + str(source.get("v8_abstentions",0)) +
             " · V8 capture errors: " + str(source.get("v8_source_errors",0)))
    reasons = source.get("v8_abstention_reasons") or {}
    if reasons:
        st.write("Source-evidence abstentions (not lost tickets): " +
                 ", ".join(str(k) + ": " + str(v)
                           for k,v in sorted(reasons.items())))
    leagues=source.get("v8_abstention_reasons_by_league") or {}
    league_rows=[{"League": league,"Source gap": reason,"Abstentions":count}
                 for league, groups in sorted(leagues.items())
                 for reason,count in sorted(groups.items()) if count]
    if league_rows:
        st.dataframe(league_rows, hide_index=True, use_container_width=True)
    bins_by_league=source.get("v8_min_team_history_bins_by_league") or {}
    bins_rows=[
        {"Competition":league,
         "Minimum qualified games for either team":history_bin,
         "Fixtures abstaining":count}
        for league,bins in sorted(bins_by_league.items())
        for history_bin,count in sorted(bins.items()) if count
    ]
    if bins_rows:
        st.caption("Historical matches available to the weaker-evidence team in each fixture. Based only on original competition, provider and pre-kickoff result cutoffs; these are not model wins.")
        st.dataframe(bins_rows, hide_index=True, use_container_width=True)
    st.markdown("#### V37 sparse UEFA model — research only")
    st.caption("For European teams with 2–4 verified same-competition matches. This does NOT turn V8 abstentions into verified five-double predictions.")
    st.write(
        "New V37 frozen fixtures in latest source capture: " +
        str(source.get("v37_sparse_saved",0)) +
        " · Total V37 frozen fixtures: " +
        str(v37.get("frozen_unique_physical_fixtures",0)) +
        " · Verified matched fixtures: " +
        str(v37.get("verified_matched_physical_fixtures",0)) +
        " · Pending: " + str(v37.get("pending",0))
    )
    if v37.get("invalid_or_tampered_excluded"):
        st.warning("Some V37 research records failed integrity checks and are excluded.")
    arm_rows = [
        {"Research method": name.replace("_"," "),
         "Verified fixtures": method.get("verified_leg_slots"),
         "Mean exact-score log loss": method.get("mean_exact_score_log_loss"),
         "Mean multiclass Brier": method.get("mean_multiclass_score_brier")}
        for name,method in (v37.get("methods") or {}).items()
    ]
    if arm_rows:
        st.dataframe(arm_rows, hide_index=True, use_container_width=True)
    st.caption("These are standalone same-fixture probabilities, NOT V28/V35/V36 five-double accuracy or validated model promotion.")
    st.markdown("#### New V28 decision capture")
    st.caption("V28 source decisions must be issued and stored before the V35/V36 future-only 20-minute window expires. No retrospective tickets.")
    st.write(
        "New original decisions in latest bot run: " +
        str(v28.get("new_original_decisions_last_run", 0)) +
        " · V28 independently verified days: " +
        str(v28.get("verified_paired_days", 0)) +
        " · V28 pending days: " + str(v28.get("pending_days", 0))
    )
    if not v28.get("new_original_decisions_last_run") and not v36.get("frozen_days"):
        st.caption("No new eligible original decision was recorded in the latest run. Fresh authenticated V8 source receipts are required before V36 can issue new prospective forecasts.")
    st.markdown("#### V36 prospective score-tail research")
    st.caption("Research only. New tickets are frozen before kickoff from the same original V28 pair; missing finals are pending, never losses.")
    st.write(
        "Frozen days: " + str(v36.get("frozen_days", 0)) +
        " · Independently verified Sydney days: " +
        str(v36.get("verified_independent_sydney_days", 0)) +
        " · Pending: " + str(v36.get("pending_days", 0))
    )
    v36_rows = []
    for name, result in (v36.get("methods") or {}).items():
        scores = result.get("mean_proper_scores") or {}
        v36_rows.append({
            "Method": name.replace("_", " "),
            "Verified days": int(result.get("verified_days") or 0),
            "Five-double hits": int(result.get("hits") or 0),
            "Observed hit rate": _pct(result.get("observed_hit_rate")),
            "Mean exact-score log loss": scores.get("exact_score_log_loss"),
            "Mean multiclass Brier": scores.get("multiclass_score_brier"),
            "High-goal Brier": scores.get("over_3_5_brier"),
        })
    if v36_rows:
        st.dataframe(v36_rows, hide_index=True, use_container_width=True)
    else:
        st.caption("No independently verified future-frozen V36 outcomes yet.")
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
