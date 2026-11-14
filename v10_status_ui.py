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
    cols=st.columns(4)
    for column,key,label in zip(cols,("v8","v9","v10"),
                                ("v8 · Statistical","v9 · Market","v10 · Multi-book")):
        item=models.get(key) or {}
        with column:
            st.metric(label+" frozen",int(item.get("frozen_fixtures") or 0))
            n=int(item.get("verified_fixtures") or 0)
            hits=int(item.get("exact_score_hits") or 0)
            st.caption(f"Exact hits: {hits}/{n}" + (
                f" · {item['exact_score_rate']:.1%}" if n and item.get("exact_score_rate") is not None
                else " · No verified outcomes yet"))
    near=models.get("v8_near") or {}
    near_count=int(near.get("frozen_fixtures") or 0)
    near_checked=int(near.get("verified_fixtures") or 0)
    near_hits=int(near.get("exact_score_hits") or 0)
    st.caption("Fresh verified pre-kickoff research: "+str(near_count)+
               " separately frozen v8 snapshots · "+str(near_hits)+"/"+
               str(near_checked)+" exact-score hits on completed matches. "+
               "Original v8 predictions are never rewritten.")
    doubles=report.get("five_doubles") or {}
    with cols[3]:
        st.metric("Five doubles frozen",int(doubles.get("frozen_pairs") or 0))
        n=int(doubles.get("verified_pairs") or 0)
        wins=int(doubles.get("wins") or 0)
        st.caption(f"Verified wins: {wins}/{n}"+(
            f" · {doubles['hit_rate']:.1%}" if n and doubles.get("hit_rate") is not None
            else " · No verified pairs yet"))
    near_pair=report.get("paired_original_v8_vs_near_v8") or {}
    near_verified=int(near_pair.get("same_fixture_verified") or 0)
    if near_verified:
        st.caption("Original vs fresh v8 · "+str(near_verified)+
                   " independently verified same-fixture matches · fresh-only hits: "+
                   str(near_pair.get("fresh_research_only_exact_hits") or 0)+
                   " · original-only hits: "+
                   str(near_pair.get("original_only_exact_hits") or 0)+
                   ". These forecasts used different available information.")
    common=(report.get("paired_v8_v9") or {}).get("verified_same_fixture_count",0)
    common_v10=(report.get("paired_v8_v10") or {}).get("verified_same_fixture_count",0)
    st.caption(f"Fair same-fixture verification: v8 vs v9 = {common} matches · v8 vs v10 = {common_v10} matches. Pending games are neither hits nor losses.")
    comparisons=doubles.get("same_pair_challengers") or {}
    if any((data.get("same_pair_verified_days") or 0)>0 for data in comparisons.values()):
        st.caption("Same frozen match-pair five-double comparisons: "+
                   " · ".join(name.upper()+": "+str(item["challenger_wins"])+"/"+
                              str(item["same_pair_verified_days"])+ " hits"
                              for name,item in comparisons.items() if item.get("same_pair_verified_days")) )
    if not compact:
        for name in ("v8","v8_near","v9","v10"):
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
        with st.expander("⚙️ Automatic forecast & odds-refresh diagnostics"):
            try:
                status=json.loads(Path("data/prospective_ledger/capture_coverage.json").read_text(encoding="utf-8"))
            except (OSError,ValueError,TypeError):
                status={}
            if status:
                st.write({
                    "v8 saved by latest collection":int(status.get("v10_shadow_v8_saved") or 0),
                    "v9 saved by latest collection":int(status.get("v10_shadow_v9_saved") or 0),
                    "v10 saved by latest collection":int(status.get("v10_shadow_v10_saved") or 0),
                    "new near-kickoff research forecasts":int(status.get("v10_near_research_saved") or 0),
                    "new near-kickoff market checks":int(status.get("v10_market_refresh_checked") or 0),
                    "new near-kickoff v9 forecasts":int(status.get("v10_market_refresh_v9_saved") or 0),
                    "provider market calls":int(status.get("v10_market_calls") or 0),
                    "shadow errors":len(status.get("v10_shadow_errors") or []),
                    "market refresh errors":len(status.get("v10_market_refresh_errors") or [])
                })
                st.caption("Latest collector report only. The near-kickoff refresh is enabled on six-hour heavy runs; a light run normally shows zero refresh checks.")
            else:
                st.caption("No collector diagnostics have been published in this deployment yet.")
        st.caption("Accuracy fixes v10.1: near-kickoff odds refresh · same-evidence market controls · matchday double freeze · timestamp-based bookmaker selection.")
        st.caption("Market-informed v9 and multi-book v10 are challengers, not proven superior models. Better historical-looking predictions do not justify changing the recorded champion without prospective evidence.")
