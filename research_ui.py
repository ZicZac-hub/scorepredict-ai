"""ZicZac Deep Research workspace.

A match-specific UI for statistical forecasts and verified evidence.
Everything is evaluated strictly before the selected kickoff. Optional
team news is fetched only on request and is never fabricated.
"""
from datetime import datetime, timezone

from deep_research_ai import research_prediction

SECTIONS = (
    "📊 Match Analysis",
    "📅 Last 10 Matches",
    "🏟️ Home / Away",
    "🤝 Head to Head",
    "🩺 Injuries & Lineups",
    "♟️ Formations",
    "🎯 Score Probabilities",
)


def _utc(value):
    try:
        result = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return result.replace(tzinfo=timezone.utc) if result.tzinfo is None else result.astimezone(timezone.utc)
    except (ValueError, TypeError):
        return None


def _games(fixture, history, team_id=None, venue=None, h2h=False, limit=10):
    """Return verified, chronological pre-match evidence only."""
    cutoff = _utc(fixture.get("utcDate"))
    home_id = str((fixture.get("homeTeam") or {}).get("id"))
    away_id = str((fixture.get("awayTeam") or {}).get("id"))
    output = []
    for match in history or []:
        if match.get("status") != "FINISHED":
            continue
        date = _utc(match.get("utcDate"))
        if date is None or cutoff is None or date >= cutoff or date > datetime.now(timezone.utc):
            continue
        home = match.get("homeTeam") or {}
        away = match.get("awayTeam") or {}
        hid, aid = str(home.get("id")), str(away.get("id"))
        if h2h and {hid, aid} != {home_id, away_id}:
            continue
        if team_id is not None and str(team_id) not in (hid, aid):
            continue
        is_home = str(team_id) == hid
        if team_id is not None and venue == "home" and not is_home:
            continue
        if team_id is not None and venue == "away" and is_home:
            continue
        goals = (match.get("score") or {}).get("fullTime") or {}
        x, y = goals.get("home"), goals.get("away")
        if type(x) is not int or type(y) is not int:
            continue
        if team_id is not None:
            scored, conceded = (x, y) if is_home else (y, x)
            outcome = "W" if scored > conceded else ("D" if scored == conceded else "L")
            opponent = away.get("name") if is_home else home.get("name")
            output.append({"Date": date.date().isoformat(), "Opponent": opponent or "Unknown",
                           "Venue": "Home" if is_home else "Away",
                           "Scored": scored, "Conceded": conceded, "Result": outcome,
                           "_sort": date})
        else:
            output.append({"Date": date.date().isoformat(),
                           "Home": home.get("name", "Unknown"),
                           "Away": away.get("name", "Unknown"),
                           "Final score": f"{x}-{y}", "_sort": date})
    output.sort(key=lambda row: row["_sort"], reverse=True)
    return [{k: v for k, v in row.items() if k != "_sort"} for row in output[:limit]]


def _stats(records):
    n = len(records)
    wins = sum(r["Result"] == "W" for r in records)
    draws = sum(r["Result"] == "D" for r in records)
    return {"Played": n, "Wins": wins, "Draws": draws, "Losses": n - wins - draws,
            "Scored/game": round(sum(r["Scored"] for r in records) / n, 2) if n else None,
            "Conceded/game": round(sum(r["Conceded"] for r in records) / n, 2) if n else None}


def render(st, rows, histories, api_key, sydney_kickoff, pre_match_report):
    st.markdown('<div class="zz-research-hero"><div class="zz-research-kicker">🧠 DEEP RESEARCH LAB · EXPERIMENTAL</div><h2>Know the match. <span>See the evidence.</span></h2><p>Recent form, venue performance, opponent-adjusted models and honest score probabilities. No invented injuries or guaranteed scores.</p></div>', unsafe_allow_html=True)
    choices = {}
    for row in rows:
        if not row.get("result"):
            continue
        fixture = row.get("fixture") or {}
        if not fixture.get("id"):
            continue
        label = (str((fixture.get("homeTeam") or {}).get("name") or "Home") + " vs " +
                 str((fixture.get("awayTeam") or {}).get("name") or "Away") +
                 " · " + sydney_kickoff(row.get("kickoff")))
        choices[label] = row
    if not choices:
        st.info("No prediction-ready match is available in the selected leagues or dates. Select competitions in the fixture settings first.")
        return

    search = st.text_input("🔎 Search team or match", key="deep_match_search", placeholder="Type a team name...")
    matching = {name: row for name, row in choices.items() if search.strip().casefold() in name.casefold()}
    if not matching:
        st.info("No researched matches found. Try a different team or clear the search.")
        return
    selection = st.selectbox("Select the match to research", list(matching), key="deep_match_selector")
    selected = matching[selection]
    fixture = selected["fixture"]
    provider = str(fixture.get("provider") or "football-data.org")
    identifier = (provider, str(fixture.get("id")), str(fixture.get("utcDate")))
    st.caption(str(selected.get("league") or "Football") + " · " + sydney_kickoff(selected.get("kickoff")))
    method = st.radio("Prediction method",
        ["Evidence Ensemble v8 · new challenger", "Preserved Deep Research v2"],
        horizontal=True, key="deep_prediction_method",
        help="Compare the new opponent/Elo ensemble with the previous research model. Existing saved predictions are never changed.")
    cols = st.columns(2)
    with cols[0]:
        run = st.button("🧠 Run Deep Research AI", type="primary",
                        use_container_width=True, key="deep_execute")
    with cols[1]:
        st.caption("Runs only when requested. Your archived original predictions are not overwritten.")

    if run:
        prior = selected.get("past") or histories.get(selected.get("league"), [])
        compatible = [m for m in prior if str(m.get("provider") or "football-data.org") == provider]
        with st.spinner("Reviewing verified results and model probabilities..."):
            try:
                if method.startswith("Evidence Ensemble v8"):
                    from research_ensemble_v8 import predict as candidate_predict
                    league_history = histories.get(selected.get("league"), compatible)
                    candidate = candidate_predict(fixture, compatible, league_history=league_history)
                    if candidate:
                        probabilities = {
                            "home":candidate["home_win"],"draw":candidate["draw"],
                            "away":candidate["away_win"]}
                        result = {
                            "available":True,"model_version":candidate["model_version"],
                            "predicted_score":list(candidate["scores"][0][:2]),
                            "scores":candidate["scores"],"home_xg":candidate["home_xg"],
                            "away_xg":candidate["away_xg"],"outcome_probabilities":probabilities,
                            "variants_used":list(candidate["component_weights"]),
                            "component_weights":candidate["component_weights"],
                            "baseline_score":candidate["original_baseline_score"],
                            "baseline_probability":candidate["original_baseline_score_probability"],
                            "elo":candidate["elo"],
                            "agreement":f"{candidate['component_top_score_agreement']}/{candidate['component_count']}",
                            "difference":candidate["baseline_to_challenger_total_variation"],
                            "limitations":candidate["limitations"]}
                    else:
                        result = {"available":False,"reason":"Insufficient pre-kickoff verified match history for the v8 challenger."}
                else:
                    result = research_prediction(fixture, compatible)
            except (TypeError, KeyError, ValueError, ZeroDivisionError) as exc:
                result = {"available": False, "reason": "Research could not be completed: " + str(exc)[:160]}
        st.session_state["ziczac_deep_report"] = {"fixture": identifier, "method": method, "result": result}
    current = st.session_state.get("ziczac_deep_report") or {}
    if current.get("fixture") != identifier or current.get("method") != method:
        st.info("Choose this match and press **Run Deep Research AI** to open its report.")
        return
    report = current["result"]
    if not report.get("available"):
        st.warning(report.get("reason", "Not enough verified match history for this fixture."))
        return

    home = str((fixture.get("homeTeam") or {}).get("name") or "Home")
    away = str((fixture.get("awayTeam") or {}).get("name") or "Away")
    score = "-".join(map(str, report["predicted_score"]))
    best_prob = report["scores"][0][2]
    st.markdown("### " + home + " **vs** " + away)
    cards = st.columns(3)
    cards[0].metric("Model's top exact score", score)
    cards[1].metric("Estimated score probability", f"{best_prob:.1%}")
    cards[2].metric("Expected goals (H / A)", f"{report['home_xg']:.2f} / {report['away_xg']:.2f}")
    st.caption("Experimental statistical estimate, not a promised result. No original pre-kickoff predictions are changed.")
    if report.get("baseline_score"):
        st.info("Preserved v6 score: **" + "-".join(map(str,report["baseline_score"])) +
                "** (" + f"{report['baseline_probability']:.1%}" + " modal probability) · v8 challenger: **" +
                score + "**. A changed score is not proof of increased accuracy.")
        st.caption("Model agreement on v8 modal score: " + str(report["agreement"]) +
                   " contributors · Probability-distribution difference from original: " + f"{report['difference']:.1%}")
        st.write("**Statistical contributors / declared weights:**",
                 {name: f"{weight:.0%}" for name,weight in report["component_weights"].items()})
        if report.get("elo"):
            st.caption("Historical Elo signal · Home: " + str(report["elo"]["home"]) +
                       " · Away: " + str(report["elo"]["away"]) + " · Home-advantage adjustment included")

    section = st.selectbox("Explore this match", SECTIONS, key="ziczac_deep_section")
    hid = (fixture.get("homeTeam") or {}).get("id")
    aid = (fixture.get("awayTeam") or {}).get("id")
    prior = selected.get("past") or histories.get(selected.get("league"), [])
    compatible = [m for m in prior if str(m.get("provider") or "football-data.org") == provider]
    home10 = _games(fixture, compatible, hid)
    away10 = _games(fixture, compatible, aid)
    venue_home = _games(fixture, compatible, hid, venue="home")
    venue_away = _games(fixture, compatible, aid, venue="away")

    if section == SECTIONS[0]:
        st.subheader("📊 Match-specific probability analysis")
        probs = report["outcome_probabilities"]
        a, b, c = st.columns(3)
        a.metric(home + " win", f"{probs['home']:.1%}")
        b.metric("Draw", f"{probs['draw']:.1%}")
        c.metric(away + " win", f"{probs['away']:.1%}")
        st.markdown("**Recent verified form**")
        st.dataframe(__import__("pandas").DataFrame([
            {"Team": home, **_stats(home10)},
            {"Team": away, **_stats(away10)}
        ]), hide_index=True, use_container_width=True)
        st.caption("Models combined: " + ", ".join(report["variants_used"]))
        st.caption(report["limitations"])

    elif section == SECTIONS[1]:
        st.subheader("📅 Last 10 completed matches")
        st.markdown("**" + home + "**")
        if home10:
            st.dataframe(__import__("pandas").DataFrame(home10), hide_index=True, use_container_width=True)
        else:
            st.info("No verified results available for this team's recent games.")
        st.markdown("**" + away + "**")
        if away10:
            st.dataframe(__import__("pandas").DataFrame(away10), hide_index=True, use_container_width=True)
        else:
            st.info("No verified results available for this team's recent games.")
        st.caption("Only finished games recorded before kickoff are included.")

    elif section == SECTIONS[2]:
        st.subheader("🏟️ Home and away performance")
        st.dataframe(__import__("pandas").DataFrame([
            {"Team / split": home + " · overall", **_stats(home10)},
            {"Team / split": home + " · home", **_stats(venue_home)},
            {"Team / split": away + " · overall", **_stats(away10)},
            {"Team / split": away + " · away", **_stats(venue_away)}
        ]), hide_index=True, use_container_width=True)
        st.caption("Split samples may contain fewer than 10 matches; blank averages indicate no verified results.")

    elif section == SECTIONS[3]:
        st.subheader("🤝 Head-to-head results")
        games = _games(fixture, compatible, h2h=True)
        if games:
            st.dataframe(__import__("pandas").DataFrame(games), hide_index=True, use_container_width=True)
        else:
            st.info("No verified previous meetings were returned by the connected provider.")
        st.caption("Past head-to-head records are context, not a guarantee of future results.")

    elif section == SECTIONS[4] or section == SECTIONS[5]:
        if section == SECTIONS[4]:
            st.subheader("🩺 Verified injuries and lineups")
        else:
            st.subheader("♟️ Verified formations and tactical context")
        if provider != "API-Football" or not api_key:
            st.info("Provider-verified injury and lineup data is available only for supported API-Football fixtures. No lineup or absence has been assumed.")
        else:
            if st.button("🔄 Check provider team news", key="deep_check_news", use_container_width=True):
                with st.spinner("Checking player availability and lineup records..."):
                    try:
                        news = pre_match_report(api_key, fixture)
                    except (TypeError, KeyError, ValueError, RuntimeError) as exc:
                        news = {"available": False, "reason": "Provider check failed: " + str(exc)[:180]}
                st.session_state["ziczac_deep_news"] = {"fixture": identifier, "report": news}
            stored = st.session_state.get("ziczac_deep_news") or {}
            if stored.get("fixture") != identifier:
                st.caption("Select **Check provider team news** to request the latest available pre-match records.")
            else:
                news = stored["report"]
                if not news.get("available"):
                    st.info(news.get("reason", "No verified team news available."))
                elif section == SECTIONS[4]:
                    injuries = news.get("injuries") or []
                    if injuries:
                        st.dataframe(__import__("pandas").DataFrame(injuries), hide_index=True, use_container_width=True)
                    else:
                        st.info("No injury reports returned. This does NOT establish that everyone is fit.")
                    lineups = news.get("lineups") or []
                    if lineups:
                        for lineup in lineups:
                            st.write("**" + str(lineup.get("team")) + "** · " +
                                     ("Confirmed starters" if lineup.get("confirmed") else "Partial lineup"))
                            st.write(", ".join(lineup.get("starters") or []))
                    for warning in news.get("warnings") or []:
                        st.warning(str(warning))
                else:
                    lineups = news.get("lineups") or []
                    if lineups:
                        for lineup in lineups:
                            st.markdown("**" + str(lineup.get("team")) + "**")
                            st.write("Provider formation: " + str(lineup.get("formation") or "Not verified"))
                            st.caption("Confirmed starting XI" if lineup.get("confirmed") else "Not a confirmed full starting XI")
                    else:
                        st.info("Confirmed or verified formations are not available yet. No tactical setup has been invented.")
        st.caption("Player and formation records are context only; this challenger has not validated numerical adjustments for them.")

    elif section == SECTIONS[6]:
        st.subheader("🎯 Exact-score probability distribution")
        table = [{"Exact score": f"{h}-{a}", "Model probability (%)": round(float(prob) * 100, 2)}
                 for h, a, prob in report["scores"]]
        st.dataframe(__import__("pandas").DataFrame(table), hide_index=True, use_container_width=True)
        st.bar_chart(__import__("pandas").DataFrame(table).set_index("Exact score"))
        st.caption("The top scores are alternatives for the SAME game, not five independent bets. Probabilities are uncalibrated challenger estimates.")

