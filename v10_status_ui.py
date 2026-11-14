"""Read-only v10 observed-performance panel; never manufacture hit rates."""
from pathlib import Path
import json


def render(st,path="data/v10_prospective/performance.json",compact=False):
    st.markdown("### 🔬 Verified Hybrid v10 · Research Performance")
    st.caption("Frozen pre-kickoff research predictions only. Older historical replay is a separate experiment; no retroactively revised scores count as wins.")
    try:
        report=json.loads(Path(path).read_text(encoding="utf-8"))
    except (FileNotFoundError,OSError,ValueError,TypeError):
        st.info("The new v10 research collector has not published a verified performance report yet. Original predictions remain available.")
        return
    models=report.get("models") or {}
    cols=st.columns(3)
    for column,key,label in zip(cols,("v8","v9"),("v8 · Statistical","v9 · Verified market")):
        item=models.get(key) or {}
        with column:
            st.metric(label+" frozen",int(item.get("frozen_fixtures") or 0))
            n=int(item.get("verified_fixtures") or 0)
            hits=int(item.get("exact_score_hits") or 0)
            st.caption(f"Exact hits: {hits}/{n}" + (
                f" · {item['exact_score_rate']:.1%}" if n and item.get("exact_score_rate") is not None
                else " · No verified outcomes yet"))
    doubles=report.get("five_doubles") or {}
    with cols[2]:
        st.metric("Five doubles frozen",int(doubles.get("frozen_pairs") or 0))
        n=int(doubles.get("verified_pairs") or 0)
        wins=int(doubles.get("wins") or 0)
        st.caption(f"Verified wins: {wins}/{n}"+(
            f" · {doubles['hit_rate']:.1%}" if n and doubles.get("hit_rate") is not None
            else " · No verified pairs yet"))
    common=(report.get("paired_v8_v9") or {}).get("verified_same_fixture_count",0)
    st.caption(f"Same-fixture v8 versus v9 verified comparison: {common} pairs. Pending games are neither hits nor losses.")
    if not compact:
        for name in ("v8","v9"):
            m=models.get(name) or {}
            with st.expander(name.upper()+" · Full probability-quality breakdown"):
                if not m.get("verified_fixtures"):
                    st.info("No finished matches with independently verified regulation scores yet.")
                else:
                    st.write({"Evaluated":m["verified_fixtures"],
                              "Exact-score accuracy":f"{m['exact_score_rate']:.2%}",
                              "Top-three coverage":f"{m['top3_hit_rate']:.2%}",
                              "Score log loss (lower better)":round(m["mean_score_log_loss"],3),
                              "1X2 Brier score (lower better)":round(m["mean_outcome_brier"],3)})
        st.caption("Market-informed v9 is a challenger, not a proven superior model. Better historical-looking predictions do not justify changing the recorded champion without prospective evidence.")
