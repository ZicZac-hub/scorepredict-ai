"""Streamlit entry point for the V43 forecast and its research evidence."""
from datetime import datetime, timedelta, timezone
from functools import lru_cache
import os

from ai_five_double_engine import SYDNEY
from ai_five_double_ui import archive_rows, repository_provider_packet
from ai_five_live_finder import research_day
from on_demand_ai import _secret, private_gate_ready, utc
from sydney_display import sydney_timestamp
from v25_upcoming import available, pair_catalog
from v43_prediction import forecast_pair, forecast_single
from v43_research import render_v43_research


@lru_cache(maxsize=3)
def _upcoming_archive(minute):
    return archive_rows(now=datetime.fromtimestamp(minute * 60, timezone.utc))


def _live_rows(st, clock):
    if not private_gate_ready(st):
        return []
    api_key = _secret(st, "API_FOOTBALL_KEY", os.environ)
    if not api_key:
        return []
    today = clock.astimezone(SYDNEY).date()
    stored = st.session_state.setdefault("v43_discovery", {})
    for day in list(stored):
        if day < today:
            del stored[day]
    with st.expander("Find and model fixtures from API-Football"):
        day = st.date_input("Sydney match date", today, min_value=today,
                            max_value=today + timedelta(days=7), key="v43_discovery_day")
        old = stored.get(day, {})
        cooldown = utc(st.session_state.get("v43_discovery_retry_after"))
        waiting = bool(cooldown and clock < cooldown)
        if st.button("Search and model up to four fixtures", key="v43_discover", disabled=waiting):
            with st.spinner("Loading fixtures and historical results…"):
                report = research_day(api_key, day, known_ids=old.get("checked_ids"), now=clock)
            old = {
                "rows": old.get("rows", []) + report["modeled"],
                "checked_ids": old.get("checked_ids", []) + report.get("checked_ids", []),
                "discovered": report["discovered"],
                "errors": bool(report["errors"]),
            }
            stored[day] = old
            if report["quota_limited"]:
                st.session_state["v43_discovery_retry_after"] = report["quota_retry_after_utc"]
                waiting = True
        if waiting:
            st.warning("API-Football reported a quota limit. Discovery will be available after the cooldown.")
        if old:
            st.caption(f"{old['discovered']} upcoming fixtures returned. {len(old['rows'])} have a full score model.")
            if old["errors"] and not waiting:
                st.caption("The provider or historical model could not supply all requested data.")
    return [row for entry in stored.values() for row in entry["rows"]]


def _score_table(model):
    return [
        {"Regulation score": _format_score(row["score"]),
         "Probability": f"{row['probability']:.2%}"}
        for row in model["top_scores"]
    ]


def _render_result(st, result, rows):
    if result["status"] != "issued":
        reason = str(result.get("reason") or "").replace("_", " ")
        st.warning(f"V43 could not issue a forecast for this selection. {reason.capitalize()}.")
        return

    if result.get("five_doubles"):
        st.subheader("Combined five exact-score doubles")
        st.metric("Estimated coverage of the five combinations", f"{result['estimated_coverage']:.2%}")
        st.caption("Joint probabilities multiply the two match probabilities, assuming independent match results.")
        st.dataframe([
            {"First match": _format_score(choice["score1"]),
             "Second match": _format_score(choice["score2"]),
             "Joint probability": f"{choice['joint_probability']:.2%}"}
            for choice in result["five_doubles"]
        ], hide_index=True, use_container_width=True)

    labels = [f"{model['home_team']} vs {model['away_team']}" for model in result["fixture_models"]]
    selected = 0
    if len(labels) > 1:
        selected = st.radio("Show match analysis for", range(len(labels)), format_func=labels.__getitem__,
                            horizontal=True, key="v43_detail_match")
    model = result["fixture_models"][selected]
    st.subheader(labels[selected])
    st.caption(f"Kickoff {sydney_timestamp(model['kickoff_utc'])} · Pre-match calculation {sydney_timestamp(result['decision_as_of_utc'])}")
    used = result["lineage"].get("evidence_used", {})
    additions = [name for key, name in (("xg", "shot xG"), ("market", "bookmaker markets")) if used.get(key)]
    st.caption("Forecast inputs · Statistical score grids" + (" + " + " + ".join(additions) if additions else ""))
    columns = st.columns(3)
    for column, key, label in zip(columns, ("home", "draw", "away"), ("Home win", "Draw", "Away win")):
        column.metric(label, f"{model['model_probabilities'][key]:.1%}")
    st.dataframe(_score_table(model), hide_index=True, use_container_width=True)
    with st.expander("Forecast inputs and mathematics"):
        st.write("V43 pools complete score distributions. The V41 statistical and qualified evidence model contributes 90%. The four related V42 shape hypotheses share the remaining 10%.")
        st.write("Outcome probabilities sum the score cells for home wins, draws and away wins. The goal expectations displayed here are means of the combined score grid.")
        st.json({"match_input": model, "blend": result["lineage"]})

    row = rows[selected]
    statistical = {
        "model_version": result.get("version", "v43-integrated-research-v1"),
        "scores": [[*map(int, item["score"].split("-")), item["probability"]]
                   for item in model["top_scores"]],
        "outcome_probabilities": model["model_probabilities"],
        "home_xg": model["expected_goals"]["home"],
        "away_xg": model["expected_goals"]["away"],
    }
    history = row.get("past") or []
    context = dict(row.get("research_context") or {})
    if private_gate_ready(st) and any(model.get("evidence_sources", {}).values()):
        packet = repository_provider_packet(row["fixture"], priority="research")
        if packet:
            context["v21_provider_snapshot"] = packet
    if not history:
        st.caption("This archived forecast includes the original score grid and source references. Match-by-match training rows are unavailable to the AI reviewers on this screen.")
    render_v43_research(st, row["fixture"], history, statistical, team_context=context)


def _format_score(score):
    if isinstance(score, str):
        return score.replace("-", "–")
    return f"{score[0]}–{score[1]}"


def _forecast_view(st, clock):
    live = _live_rows(st, clock)
    archived = _upcoming_archive(int(clock.timestamp() // 60))
    by_day, _ = available([*live, *archived], clock)
    if not by_day:
        st.info("No upcoming fixture currently has a qualifying pre-match score grid. Open Research and validation to review historical results.")
        return
    day = st.selectbox("Sydney match date", list(by_day), key="v43_archive_day")
    candidates = by_day[day]
    by_key = {item["key"]: item for item in candidates}

    def label(key):
        item = by_key[key]
        return f"{item['title']} · {sydney_timestamp(item['kickoff'])}"

    first_key = st.selectbox("First match", list(by_key), format_func=label, key="v43_first")
    first = by_key[first_key]
    partners = [key for key, item in by_key.items()
                if key != first_key and pair_catalog([first, item], clock)]
    second_key = st.selectbox("Second match", [None, *partners],
                              format_func=lambda key: "Single-match forecast" if key is None else label(key),
                              key="v43_second")
    selected = [first] + ([by_key[second_key]] if second_key else [])
    rows = [item["row"] for item in selected]
    signature = tuple((item["key"], item["captured_at_utc"]) for item in selected)
    if st.button("Generate V43 forecast", type="primary", key="v43_generate"):
        with st.spinner("Combining score distributions and qualifying match evidence…"):
            result = forecast_pair(*rows) if second_key else forecast_single(rows[0])
        st.session_state["v43_forecast"] = (signature, result)
    stored = st.session_state.get("v43_forecast")
    if stored and stored[0] == signature:
        _render_result(st, stored[1], rows)


@lru_cache(maxsize=1)
def _research_catalog(_hour):
    from v43_catalog import build_catalog
    return build_catalog()


@lru_cache(maxsize=1)
def _replay(_hour):
    from v43_replay import evaluate
    return evaluate()


def _evidence_view(st, clock):
    hour = int(clock.timestamp() // 3600)
    st.subheader("Historical replay")
    st.caption("Each comparison scores the same verified fixtures within its cohort. Cohorts can overlap. Retrospective results do not establish prospective accuracy.")
    replay = _replay(hour)
    with st.expander("Matched replay results", expanded=True):
        for label, cohort in replay["cohorts"].items():
            st.markdown(f"**{label.replace('_', ' ')}**")
            st.caption(f"{cohort['verified_pairs']} verified pairs · {cohort['pending_pairs']} pending · {cohort['invalid_pairs']} excluded · {cohort['abstained_pairs']} abstained")
            if cohort["methods"]:
                st.dataframe([{"Method": method, **metrics} for method, metrics in cohort["methods"].items()],
                             hide_index=True, use_container_width=True)

    catalog = _research_catalog(hour)
    summary = catalog["summary"]
    st.subheader("Historical data coverage")
    columns = st.columns(3)
    columns[0].metric("Verified fixtures", summary["verified_fixtures"])
    columns[1].metric("Archive families", summary["archive_families"])
    columns[2].metric("Archive files", summary["archive_files"])
    if summary["verified_fixture_start_utc"] and summary["verified_fixture_end_utc"]:
        st.caption(f"Verified result kickoff dates span {summary['verified_fixture_start_utc'][:10]} to {summary['verified_fixture_end_utc'][:10]} UTC.")
    if summary["openfootball_matches"]:
        st.caption(f"OpenFootball supplies {summary['openfootball_matches']:,} historical matches for the benchmark studies.")
    for label, key in (("V1–V42 methods and release history", "versions"),
                       ("Earlier project lineage", "legacy_lineage"),
                       ("Prediction methods and how V43 uses them", "methods"),
                       ("Data and research sources", "sources"),
                       ("Archive coverage", "archives"),
                       ("Existing validation studies", "benchmarks")):
        with st.expander(label):
            st.dataframe(catalog[key], hide_index=True, use_container_width=True)
    if catalog.get("metrics"):
        with st.expander("Previously measured scores by study"):
            st.dataframe(catalog["metrics"], hide_index=True, use_container_width=True)


def render(st):
    st.caption("V43 is a live research workspace. The combined forecast is experimental. OpenAI and Claude can review the available match context when configured.")
    pane = st.radio("V43 workspace", ("Forecast", "Research and validation"), horizontal=True, key="v43_pane")
    now = datetime.now(timezone.utc)
    if pane == "Forecast":
        _forecast_view(st, now)
    else:
        _evidence_view(st, now)
