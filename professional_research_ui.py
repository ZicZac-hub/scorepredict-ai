"""Keep historical model research visibly separate from live performance."""
import json
from pathlib import Path


def render(st, root=Path("data")):
    path=Path(root)/"source_quality_sensitivity_v1"/"summary.json"
    if not path.exists():
        path=Path(root)/"research_model_comparison_v1"/"summary.json"
    if not path.exists():
        return
    try:
        report=json.loads(path.read_text(encoding="utf-8"))
        thresholds=report["development_five_double_thresholds"]
        metrics=report["development_single_match_metrics"]
        if not isinstance(thresholds,dict) or not isinstance(metrics,dict):
            return
        rows=[]
        for model,values in thresholds.items():
            base=values[0]
            single=metrics[model]
            rows.append({"Research model":model,"Five-double hits":base["hits"],
                         "Issued sets":base["issued"],"Abstained dates":base["abstained"],
                         "Set hit rate":base["rate_per_issued"],
                         "Single-score log loss":single["exact_score_log_loss"],
                         "Single-outcome Brier":single["outcome_brier"]})
    except (OSError,ValueError,KeyError,TypeError,IndexError):
        return
    with st.expander("Professional model research · development retrospective"):
        st.warning("These historical dates have already been explored. This comparison "
                   "cannot establish independent prospective improvement or 70% accuracy.")
        st.caption("CC0 results across five leagues; missing final scores can omit fixtures "
                   "or dates. Different model policies may choose different pairs. "
                   "The original frozen live model remains unchanged.")
        st.caption("The source-quality sensitivity excludes a training file with verified "
                   "calendar errors. The earlier, unchanged experiment remains archived "
                   "and is not treated as independent proof of accuracy.")
        st.dataframe(rows,hide_index=True,use_container_width=True)
        st.caption("xG, bookmaker and AI variants require genuine matched historical inputs; "
                   "unavailable inputs are not synthesized. Betting ROI is not measurable "
                   "without executable prices, stakes and returns.")
        st.download_button("Download model research summary",path.read_bytes(),
                           "professional-model-research.json","application/json",
                           key="professional_research_summary_download")
