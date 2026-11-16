"""Read-only ten-provider AI research status for the ZicZac Streamlit app."""
from sydney_display import sydney_timestamp
from pathlib import Path
import json

ROOT = Path("data/llm_research")


def _json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return {}


def _score_text(value):
    if (isinstance(value, (list, tuple)) and len(value) == 2
            and all(type(v) is int and v >= 0 for v in value)):
        return str(value[0]) + "–" + str(value[1])
    return "Not available"


def _pct(value):
    if type(value) not in (int, float) or not 0 <= value <= 1:
        return "Not available"
    return f"{value:.1%}"


def _outcome_of(score):
    if not isinstance(score, (list, tuple)) or len(score) != 2:
        return None
    if not all(type(x) is int and x >= 0 for x in score):
        return None
    return "home" if score[0] > score[1] else "away" if score[0] < score[1] else "draw"


def _score_explanation(panel, votes):
    """Explain abstention without changing immutable votes or inventing scores."""
    selected = panel.get("suggested_score")
    if _score_text(selected) != "Not available":
        return "Aligned research score: " + _score_text(selected) + " (not verified)."
    if not votes:
        return "No validated AI score suggestions were saved."
    top = str(panel.get("outcome") or "")
    if top and any(_outcome_of(v.get("score")) != top for v in votes.values()):
        return ("No consensus exact score: the suggested scoreline differs from "
                "the leading home/draw/away probability. Individual AI picks "
                "appear below; nothing has been discarded.")
    return "Not enough agreeing evidence for a consensus exact score."


def render(st, root=ROOT):
    st.markdown("### 🧠 Ten-Provider Independent AI Research Panel")
    st.caption("OpenAI · Claude · Gemini · Grok · DeepSeek · Perplexity · Mistral · "
               "Qwen · GroqCloud (hosted models) · Cohere. Same frozen pre-match evidence; "
               "not a proven score-accuracy upgrade.")
    context = st.session_state.get("ziczac_ten_ai_context") or {}
    from ten_ai_action_ui import render as render_ten_ai_action
    render_ten_ai_action(st, rows=context.get("rows"),
                         histories=context.get("histories"))
    st.divider()
    st.markdown("#### 📁 Automatically frozen research archive")
    st.caption("The records below are saved by the separate GitHub schedule. "
               "They cannot be changed by an on-demand Run & Predict request.")
    status = _json(root/"status.json")
    if not status.get("enabled"):
        st.info("Ten-provider research is installed but not activated. Enter API keys, "
                "model IDs, and the opt-in GitHub Actions variable to enable real calls. "
                "Existing ZicZac predictions are unchanged.")
    elif not status.get("configured_providers"):
        st.warning("Research panel enabled, but no provider keys and model IDs "
                   "are configured. No external AI predictions have been fetched.")
    else:
        st.caption("Configured AI services: "+", ".join(status["configured_providers"])+
                   ". A full ten-provider consensus requires all ten; partial results are marked.")
        errors = status.get("provider_error_types") or {}
        details = status.get("last_attempt_errors") or status.get("provider_error_details") or {}
        if details:
            last = status.get("last_attempt_at_utc") or status.get("checked_at_utc")
            st.warning("Last attempted provider check ("+sydney_timestamp(last)+") had "
                       +str(len(details))+" provider error(s). These are not football losses.")
            for provider, error in sorted(details.items()):
                reason = str(error.get("reason") or "provider_error")
                code = error.get("http_status")
                st.caption(provider.upper()+": "+reason+
                           ((" · HTTP "+str(code)) if code is not None else ""))
            st.caption("HTTP 429 can mean rate limits or missing credits; 400 usually "
                       "requires checking the provider model ID and request format. "
                       "API keys are never shown.")
        elif errors:
            st.warning("Some providers did not return valid research: "+
                       ", ".join(sorted(errors))+". The system does not invent votes.")
    report = _json(root/"evaluation.json")
    n = int(report.get("paired_verified_scores") or 0)
    if n:
        st.write({"Matched completed matches": n,
                  "Five-AI suggested exact hits": report.get("panel_exact_hits"),
                  "Original statistical exact hits": report.get("baseline_exact_hits")})
        st.caption("Both rows use identical completed fixtures; there is no automatic model promotion.")
    else:
        st.caption("No verified complete-panel results yet; no multi-AI exact-score win rate can be claimed.")
    ten_n = int(report.get("ten_provider_paired_verified") or 0)
    if ten_n:
        st.write({"All ten AI providers, paired verified matches": ten_n,
                  "Ten-provider exact hits": report.get("ten_provider_exact_hits"),
                  "Same-fix original ZicZac exact hits":report.get("ten_provider_baseline_exact_hits")})
        st.caption("The original 5-AI group and the new 10-AI group are measured separately.")
    else:
        st.caption("A complete ten-AI panel does not yet have verified final results.")
    saved = []
    for file in (root/"forecasts").glob("*.json"):
        entry = _json(file)
        if entry.get("research_only") and entry.get("captured_at_utc") and entry.get("consensus"):
            saved.append(entry)
    saved.sort(key=lambda entry: str(entry["captured_at_utc"]), reverse=True)
    if not saved:
        st.info("No frozen external-AI research yet. A valid pre-kickoff "
                "fixture must be collected before any AI predictions appear.")
    for entry in saved[:3]:
        panel = entry["consensus"]
        votes = entry.get("votes") or {}
        actual_count = len(votes)
        configured = entry.get("configured_provider_count") or len(status.get("configured_providers") or [])
        label = (str(entry.get("home_team") or "Home")+" vs "+
                 str(entry.get("away_team") or "Away"))
        with st.expander(label + " · " + str(actual_count) + " AI vote(s) saved"):
            st.caption("Frozen: "+sydney_timestamp(entry.get("captured_at_utc"))+
                       " · Source: +str(entry.get("source_model") or "unknown")+
                       " · Original forecast preserved")
            st.write("**AI responses:** "+str(actual_count)+
                     " of "+str(configured)+" configured at capture")
            probabilities = panel.get("outcome_probabilities") or {}
            st.write("**Independent research — home/draw/away:** "
                     + " / ".join(_pct(probabilities.get(k)) for k in
                                  ("home","draw","away")))
            st.write("**Most likely match outcome in AI panel:** "
                     + str(panel.get("outcome") or "Not available").title())
            st.write("**Original ZicZac exact score:** "
                     + _score_text(entry.get("baseline_predicted_score")))
            st.caption(_score_explanation(panel, votes))
            if votes:
                st.markdown("**Individual AI suggestions (unvalidated)**")
                for name,vote in sorted(votes.items()):
                    st.write("**"+name.title()+"** · Score "+
                             _score_text(vote.get("score"))+
                             " · H/D/A "+
                             "/".join(_pct(vote.get(k)) for k in
                                      ("home","draw","away")))
                    facts=vote.get("factors") or []
                    if isinstance(facts,list) and facts:
                        st.caption("Evidence notes: "+"; ".join(str(item)[:140]
                                   for item in facts[:2]))
            st.caption("Only pre-kickoff archived AI opinions are shown. "
                       "These are not real-time updates or verified hits. "
                       "A missing vote is not a zero-percent prediction.")
    st.divider()
    st.markdown("#### ⚽ V15 Two-Match AI-Assisted Five Doubles")
    st.caption("Requires two genuine pre-match statistical forecasts AND at "
               "least two successful independent AI API responses per match. "
               "Six configured keys do not mean six successful votes. "
               "Existing V8 predictions are preserved.")
    if status.get("two_match_research_enabled"):
        st.caption("Two-match GitHub AI research mode: enabled by its "
                   "additional request budget. Same-day pair selected: "+
                   ("yes" if status.get("same_day_pair_selected") else "not this run"))
        if status.get("selected_original_frozen_pair_day"):
            st.success("Exact original frozen five-double pair selected for research: "+
                       str(status["selected_original_frozen_pair_day"])+
                       ". Both model sources are matched to their historical forecast IDs.")
        elif status.get("same_day_pair_selected"):
            st.caption("A same-day pair was researched, but there was no eligible exact "
                       "match to the original frozen five-double pair. This cannot "
                       "be counted as a successful V15 frozen-pair comparison.")
        st.caption("AI API requests planned in the latest eligible batch: "+
                   str(status.get("planned_model_requests") or 0)+
                   " · Successfully returned votes in that batch: "+
                   str(status.get("actual_successful_ai_votes") or 0)+
                   ". A configured provider is not a successful response.")
    else:
        st.info("Scheduled AI research is currently limited to one fixture "
                "per heavy run. For two-match research, set GitHub Actions "
                "variable LLM_RESEARCH_MAX_MATCHES to 2 after reviewing "
                "provider costs. This can approximately double calls per "
                "heavy run, so it stays opt-in.")
    v15root=Path("data/v15_ai_double")
    v15status=_json(v15root/"status.json")
    v15report=_json(v15root/"evaluation.json")
    source_audit=_json(v15root/"source_audit.json")
    if source_audit:
        st.caption(
            "V16 source audit · "+str(source_audit.get("frozen_original_double_sets_checked",0))+
            " original double sets checked · "+
            str(source_audit.get("original_sets_with_at_least_two_ai_votes_on_both_legs",0))+
            " with >=2 genuine source-matched AI votes on BOTH matches."
        )
        if source_audit.get("source_rows"):
            with st.expander("V16: source-matched AI coverage by double"):
                for item in source_audit["source_rows"][-8:]:
                    st.write(str(item.get("sydney_day"))+" · Match 1 / Match 2: "+
                             " / ".join(str(n) for n in item.get("actual_provider_votes_per_leg",[]))+
                             " valid AI responses")
                    st.caption(" / ".join(item.get("leg_reasons",[])))
    attempted=v15status.get("existing_original_days")
    if attempted is not None:
        st.caption("Original daily pairs inspected: "+str(attempted)+
                   " · New eligible AI-assisted pairs frozen: "+
                   str(v15status.get("frozen_new") or 0)+
                   " · Existing frozen AI-assisted pairs: "+
                   str(v15status.get("previously_frozen") or 0))
        if v15status.get("abstentions"):
            with st.expander("Why multi-AI five doubles may be unavailable"):
                st.write(v15status["abstentions"])
    if v15report.get("verified_pairs"):
        st.write({"Verified AI-assisted pairs":v15report["verified_pairs"],
                  "AI-assisted exact double wins":v15report["ai_assisted_hits"],
                  "Original V8 wins on those identical pairs":
                      v15report["original_v8_hits_same_pairs"]})
        st.caption("95% uncertainty interval for V15 verified set win rate: "+
                   ("–".join(f"{value:.1%}" for value in
                             v15report["v15_win_rate_wilson_95"])
                    if v15report.get("v15_win_rate_wilson_95")
                    else "Awaiting newer V16 comparison records"))
        st.caption("Paired comparison on the same two fixtures: "+
                   str(v15report.get("v15_only_wins") or 0)+
                   " AI-only wins vs "+
                   str(v15report.get("v8_only_wins") or 0)+
                   " V8-only wins. No automatic production promotion.")
        bounds=v15report.get("v15_win_rate_wilson_95")
        if isinstance(bounds,list) and len(bounds)==2:
            st.caption("AI-assisted observed win-rate uncertainty (95% Wilson): "+
                       _pct(bounds[0])+"–"+_pct(bounds[1])+
                       ". Only genuinely settled same-pair predictions count; "
                       "this is not a guarantee of future accuracy.")
        st.caption("Paired disagreement · V15-only wins: "+
                   str(v15report.get("v15_only_wins",0))+
                   " · V8-only wins: "+str(v15report.get("v8_only_wins",0))+
                   ". Original V8 remains production.")
    else:
        st.caption("No verified two-match multi-AI win rate yet. "
                   "No matched paired results means no accuracy claim.")
    archives=[]
    for filepath in (v15root/"forecasts").glob("*.json"):
        entry=_json(filepath)
        if (entry.get("mode")=="v15_ai_supported_double_shadow" and
            entry.get("status")=="frozen_pre_kickoff"):
            archives.append(entry)
    archives.sort(key=lambda row:str(row.get("captured_at_utc") or ""),
                  reverse=True)
    for item in archives[:2]:
        with st.expander("AI-assisted frozen five doubles · "+
                         str(item.get("sydney_date") or ""),expanded=True):
            st.caption(str(item["first"]["home_team"])+" vs "+
                       str(item["first"]["away_team"])+" | "+
                       str(item["second"]["home_team"])+" vs "+
                       str(item["second"]["away_team"]))
            st.caption("Verified provider votes: "+str(len(item["ai_provider_names"]["first"]))+
                       " on match 1 and "+
                       str(len(item["ai_provider_names"]["second"]))+
                       " on match 2. These models used the same briefing and "
                       "are not independent calibrated predictors.")
            for i,p in enumerate(item["five_doubles"],1):
                st.write("**Double "+str(i)+"** · "+
                         p["score1"]+" & "+p["score2"])
                st.caption("Statistical joint estimate: "+
                           f"{p['statistical_joint_estimate']:.2%}"+
                           " · AI endorsements "+
                           str(p["supporting_ai_votes_match1"])+"/"+
                           str(p["supporting_ai_votes_match2"]))
            st.caption("This is a separately frozen research challenger. "
                       "AI endorsements do not make the joint probability "
                       "larger or prove greater accuracy.")
    st.divider()
    from sportmonks_probability_ui import render as render_sportmonks
    render_sportmonks(st)
