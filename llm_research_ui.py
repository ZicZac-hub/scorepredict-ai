"""Read-only ten-provider AI research status for the ZicZac Streamlit app."""
from pathlib import Path
import json

ROOT = Path("data/llm_research")


def _json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return {}


def render(st, root=ROOT):
    st.markdown("### 🧠 Ten-Provider Independent AI Research Panel")
    st.caption("OpenAI · Claude · Gemini · Grok · DeepSeek · Perplexity · Mistral · "
               "Qwen · GroqCloud (Llama) · Cohere. Same frozen pre-match evidence; "
               "not a proven score-accuracy upgrade.")
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
        if errors:
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
    archived = []
    for path in sorted((root/"forecasts").glob("*.json"), reverse=True):
        entry = _json(path)
        if entry.get("research_only") and entry.get("captured_at_utc") and entry.get("consensus"):
            archived.append(entry)
        if len(archived) >= 3:
            break
    for entry in archived:
        panel = entry["consensus"]
        label = (str(entry.get("home_team") or "Home")+" vs "+
                 str(entry.get("away_team") or "Away"))
        with st.expander(label+" · "+str(entry.get("provider_count"))+"/10 independent models"):
            st.caption("Frozen at "+str(entry.get("captured_at_utc"))+
                       " UTC; matchup kickoff "+str(entry.get("kickoff_utc"))+
                       "; source "+str(entry.get("source_model")))
            st.write({"Outcome": panel.get("outcome"),
                      "Outcome probabilities": panel.get("outcome_probabilities"),
                      "AI suggested score (unvalidated)": panel.get("suggested_score"),
                      "Original saved score (unchanged)": entry.get("baseline_predicted_score")})
            st.caption("AI scores are research candidates only; do not treat these "
                       "as independently calibrated exact-score probabilities.")
    st.divider()
    from sportmonks_probability_ui import render as render_sportmonks
    render_sportmonks(st)
