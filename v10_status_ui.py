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
            st.caption((f"Verified exact hits: {hits}/{n} · {item['exact_score_rate']:.1%}"
                        if n and item.get("exact_score_rate") is not None
                        else "⏳ Pending verification — accuracy not yet measured"))
    near=models.get("v8_near") or {}
    near_count=int(near.get("frozen_fixtures") or 0)
    near_checked=int(near.get("verified_fixtures") or 0)
    near_hits=int(near.get("exact_score_hits") or 0)
    st.caption("Fresh verified pre-kickoff research: "+str(near_count)+
               " separately frozen v8 snapshots · "+str(near_hits)+"/"+
               str(near_checked)+" exact-score hits on completed matches. "+
               "Original v8 predictions are never rewritten.")
    v12=models.get("v12") or {}
    v12_count=int(v12.get("frozen_fixtures") or 0)
    v12_checked=int(v12.get("verified_fixtures") or 0)
    v12_hits=int(v12.get("exact_score_hits") or 0)
    st.caption("🧪 V12 league score calibration (shadow only): "+
               str(v12_count)+" separate frozen forecasts · "+
               str(v12_hits)+"/"+str(v12_checked)+
               " verified exact-score hits. This experiment has not been "+
               "promoted into the main predictions.")
    v13=models.get("v13") or {}
    v13_count=int(v13.get("frozen_fixtures") or 0)
    v13_verified=int(v13.get("verified_fixtures") or 0)
    v13_hits=int(v13.get("exact_score_hits") or 0)
    st.caption("🧠 V13 Hybrid (experimental, not promoted): "+
               str(v13_count)+" immutable pre-match forecasts · "+
               str(v13_hits)+"/"+str(v13_verified)+
               " verified exact-score hits. Unfinished games are pending.")
    doubles=report.get("five_doubles") or {}
    with cols[3]:
        st.metric("Five doubles frozen",int(doubles.get("frozen_pairs") or 0))
        n=int(doubles.get("verified_pairs") or 0)
        wins=int(doubles.get("wins") or 0)
        st.caption((f"Verified five-double wins: {wins}/{n} · {doubles['hit_rate']:.1%}"
                    if n and doubles.get("hit_rate") is not None
                    else "⏳ Pending both match results — not zero accuracy"))
    if not compact:
        with st.expander("🧮 V25 Professional Analyst · Research-only five doubles"):
            try:
                professional = json.loads(
                    Path("data/pro_analyst_shadow/status.json").read_text(
                        encoding="utf-8"))
            except (OSError, ValueError, TypeError):
                professional = {}
            if not professional:
                st.info("No V25 shadow selection report yet. This independent "
                        "experiment never changes your existing five doubles.")
            else:
                n = int(professional.get("verified_sets") or 0)
                w = int(professional.get("robust_wins") or 0)
                b = int(professional.get("baseline_wins_on_same_pairs") or 0)
                st.write("Original frozen V8 vs a four-distribution robustness "
                         "challenger, on the exact same two-match pairs.")
                st.write({
                    "Frozen sets": int(professional.get("total_frozen_sets") or 0),
                    "Verified sets": n,
                    "Pending sets": int(professional.get("pending_sets") or 0),
                    "Robust five-double hits": w,
                    "Original V8 same-pair hits": b,
                    "Actual robust accuracy": (
                        f"{w/n:.1%}" if n else "Unmeasured"),
                })
                st.caption("Robust score ranking uses fixed V8, independent "
                           "Poisson, negative-binomial and bivariate-Poisson "
                           "sensitivity models, not four independent AI votes. "
                           "All source forecasts must be frozen before both "
                           "kickoffs. It may abstain. No method is promoted "
                           "without substantial future verification.")
                selected = [
                    packet for packet in
                    ((professional.get("candidate_coverage") or {}).get("days") or {}).values()
                    if isinstance(packet, dict) and
                    packet.get("decision")=="FROZEN_RESEARCH_ONLY"
                ]
                if selected:
                    latest=selected[-1]
                    st.caption("Latest source-verified human-style analysis "
                               "(descriptive only; unverified xG, lineups and "
                               "tactical effects have zero numeric weight):")
                    st.json({
                        "first_fixture": latest["first"].get("human_research_note"),
                        "second_fixture": latest["second"].get("human_research_note"),
                        "model_estimated_robust_mass": latest.get(
                            "robust_lower_envelope_model_mass"),
                        "not_a_proven_hit_rate": True,
                    })
                try:
                    matched = json.loads(Path("data/accuracy_diagnosis/v8_v25_matched.json")
                                         .read_text(encoding="utf-8"))
                except (OSError, ValueError, TypeError):
                    matched = {}
                mixture = matched.get("matched_mixture_five") or {}
                if mixture:
                    verified = mixture["verified_pairs"]
                    pending = mixture["pending_pairs"]
                    st.caption(f"Normalized V25 five-double trial · {verified} verified pairs · {pending} pending")
                    if verified:
                        st.dataframe([
                            {"Policy": label, "Wins": mixture[key]["five_double_hits"],
                             "Verified pairs": verified}
                            for key, label in (("v8", "V8"), ("v25_robust", "V25 robust"),
                                               ("v25_mixture", "V25 probability mixture"))
                        ], use_container_width=True, hide_index=True)
                try:
                    freeze_health = json.loads(
                        Path("data/accuracy_diagnosis/v25_freeze_monitor.json")
                        .read_text(encoding="utf-8")
                    )
                except (OSError, ValueError, TypeError):
                    freeze_health = {}
                if freeze_health:
                    st.markdown("#### V25 prospective probability freezing")
                    st.write({
                        "Older sets (not backfilled)": int(
                            freeze_health.get("legacy_sets_not_backfilled") or 0),
                        "New sets with both verified frozen probability grids": int(
                            freeze_health.get(
                                "new_sets_with_both_valid_grids_and_mixture") or 0),
                        "New completed pairs": int(
                            freeze_health.get("new_verified_pairs") or 0),
                        "New pending pairs": int(
                            freeze_health.get("new_pending_pairs") or 0),
                    })
                    if not freeze_health.get("check_passed", False):
                        st.error("V25 freeze integrity check failed. New score "
                                 "probability results must not be promoted.")
                    elif not freeze_health.get("prospective_sets_since_rollout"):
                        st.info("The archived V25 five-double selections predate the "
                                "new frozen-probability research cohort. "
                                "Waiting for genuinely new write-once predictions; "
                                "old V25 probabilities are never reconstructed.")
                    else:
                        goal = freeze_health.get("initial_100_pair_evaluation") or {}
                        st.caption(
                            f"Independent V25 score-probability evaluation: "
                            f"{goal.get('verified', 0)}/{goal.get('goal', 100)} "
                            "verified matched pairs towards an initial evidence "
                            "review, not a guaranteed accuracy target."
                        )
                try:
                    feasibility = json.loads(
                        Path("data/accuracy_diagnosis/v25_five_double_frontier.json")
                        .read_text(encoding="utf-8"))
                except (OSError, ValueError, TypeError):
                    feasibility = {}
                if feasibility and feasibility.get("original_archived_sets"):
                    st.markdown("#### V25 70% goal · model feasibility")
                    least = feasibility.get(
                        "min_number_of_combinations_to_70pct_model_mass")
                    most = feasibility.get(
                        "max_number_of_combinations_to_70pct_model_mass")
                    st.write({
                        "Source-verified archived pairs": feasibility["original_archived_sets"],
                        "Best-five minimum model mass":
                            f"{feasibility['five_model_coverage_min']:.1%}",
                        "Best-five maximum model mass":
                            f"{feasibility['five_model_coverage_max']:.1%}",
                        "Combinations for 70% estimated coverage":
                            f"{least}–{most}" if least is not None else "Unknown",
                        "Five-ticket sets meeting 70% model mass":
                            feasibility["sets_with_70pct_model_mass_in_only_five"],
                    })
                    st.caption(
                        "Model-estimated coverage is NOT measured accuracy. "
                        "These numbers apply only to the saved V8 score grids. "
                        "No V25 improvement or 70% hit rate has been verified."
                    )
                try:
                    v26 = json.loads(Path("data/v26_temporal_calibration/status.json")
                                     .read_text(encoding="utf-8"))
                except (OSError, ValueError, TypeError):
                    v26 = {}
                if v26 and v26.get("version") == "v26-temporal-tail-calibration-v1-research-only":
                    st.markdown("#### V26 · Chronological goal calibration trial")
                    retrospective = v26.get("chronological_research_replay") or {}
                    future = v26.get("prospective_verified") or {}
                    readiness = retrospective.get("training_readiness") or {}
                    st.write({
                        "Verified original V8 fixtures available as past training":
                            readiness.get("verified_original_v8_fixtures_known_now", 0),
                        "Most prior results available before a historical forecast":
                            readiness.get(
                                "max_prior_verified_before_any_historical_source_capture", 0),
                        "Historical chronological test fixtures":
                            retrospective.get("verified_holdout_fixtures", 0),
                        "Future-frozen V26 verified match pairs":
                            future.get("verified_pairs", 0),
                        "Future-frozen V26 pending pairs":
                            future.get("pending_pairs", 0),
                        "V26 verified five-double wins":
                            (future.get("same_pair_five_hits") or {}).get(
                                "v26_calibrated", 0),
                        "V25 robust wins on those same future pairs":
                            (future.get("same_pair_five_hits") or {}).get(
                                "v25_robust", 0),
                    })
                    retrospective_delta = retrospective.get("v26_minus_v8_log_loss")
                    if retrospective_delta is not None:
                        st.caption(
                            "Exploratory chronological historical V26-minus-V8 "
                            f"exact-score log loss: {retrospective_delta:+.4f} "
                            "(negative favours V26). The method was developed "
                            "after these games, so this is NOT a prospective win."
                        )
                    if future.get("verified_pairs"):
                        st.caption(
                            "Frozen prospective V26-minus-V8 exact-score log loss "
                            f"on identical verified fixtures: "
                            f"{future['v26_minus_v8_log_loss']:+.4f}. "
                            "Too few results can be inconclusive."
                        )
                    else:
                        st.info(
                            "V26 has not yet completed a genuinely frozen "
                            "prospective two-match set. It remains research-only. "
                            "Historical replay results cannot justify promotion."
                        )
                try:
                    v27 = json.loads(
                        Path("data/v27_pair_selection/status.json")
                        .read_text(encoding="utf-8"))
                except (OSError, ValueError, TypeError):
                    v27 = {}
                if v27.get("version") == "v27-frozen-full-slate-pair-quality-v1-research-only":
                    st.markdown("#### V27 · Smart full-slate match-pair selection")
                    day_study = v27.get("prospective_comparison") or {}
                    same_day_wins = day_study.get("wins_on_same_completed_days") or {}
                    st.write({
                        "Original-slate decisions frozen before kickoff":
                            day_study.get("frozen_decision_days", 0),
                        "Research abstentions": day_study.get("abstained_days", 0),
                        "V27 completed pairs": day_study.get(
                            "v27_verified_pair_results", 0),
                        "Still pending": day_study.get("v27_pending_pair_results", 0),
                        "Both policies verified on same days":
                            day_study.get("both_policies_verified_same_days", 0),
                        "V27 five-double hits on those days":
                            same_day_wins.get("v27_selected_robust", 0),
                        "V25 original hits on those days":
                            same_day_wins.get("v25_original_robust", 0),
                        "Same exact fixtures selected by both":
                            day_study.get("same_fixture_pair_comparisons", 0),
                    })
                    if day_study.get("invalid_or_tampered_decisions_excluded"):
                        st.error(
                            "Some V27 frozen selections failed provenance "
                            "verification and are excluded from results."
                        )
                    if day_study.get("both_policies_verified_same_days"):
                        st.caption(
                            "V27 versus V25 is a SAME-DAY match-pair SELECTION "
                            "comparison. The models may have selected different "
                            "fixtures. It is not a same-match probability-calibration "
                            "comparison or evidence of a 70% win rate."
                        )
                    else:
                        st.info(
                            "V27 is configured to freeze complete qualifying "
                            "match slates on genuinely NEW V25 decision days. "
                            "There are not yet matched verified selection results."
                        )
                    st.caption(
                        "V26 separately evaluates calibration on the original "
                        "V25 pair when at least 30 earlier verified sources qualify. "
                        "Do not combine these different denominators."
                    )
                try:
                    v28 = json.loads(Path("data/v28_independent_pair_trial/status.json")
                                     .read_text(encoding="utf-8"))
                except (OSError, ValueError, TypeError):
                    v28 = {}
                if v28.get("version") == "v28-independent-paired-pair-policy-v1-research-only":
                    st.markdown("#### V28 · Independent prospective pair selection")
                    comparison = v28.get("verified_comparison") or {}
                    recent_freeze = v28.get("freeze") or {}
                    st.write({
                        "Days frozen without needing a new V25 daily selection":
                            comparison.get("original_frozen_decisions", 0),
                        "Both policies fully verified on the same day":
                            comparison.get("verified_paired_days", 0),
                        "Pending pairs (not losses)":
                            comparison.get("pending_days", 0),
                        "Explicit abstention days":
                            comparison.get("abstained_days", 0),
                        "Quality/freshness policy five-double wins":
                            comparison.get("quality_wins", 0),
                        "V25-style robustness control five-double wins":
                            comparison.get("robustness_control_wins", 0),
                        "Different pairs selected on the same day":
                            comparison.get("different_pair_days", 0),
                    })
                    skipped = recent_freeze.get("skipped") or {}
                    if skipped:
                        st.caption("Source-readiness and no-freeze reasons: "+
                                   ", ".join(str(k)+": "+str(v)
                                             for k, v in skipped.items()))
                    if comparison.get("excluded_invalid_or_tampered_files"):
                        st.error("Some V28 frozen policy comparisons failed "
                                 "immutable-source validation and were excluded.")
                    if comparison.get("verified_paired_days"):
                        probability = comparison.get(
                            "exact_paired_test_two_sided_p")
                        st.caption(
                            "Both policies are issued from the same original "
                            "pre-match slate. They may select DIFFERENT "
                            "fixtures; this tests the pair-choice policy, "
                            "not identical-fixture model calibration. "
                            "Paired exact-test p: "+(
                                f"{probability:.3g}" if probability is not None
                                else "not estimable from current results")
                        )
                    else:
                        st.info(
                            "V28 can freeze its own forward-only pairs without "
                            "waiting for a newly created V25 set. Accuracy is "
                            "not measurable until independently verified "
                            "future matches finish."
                        )
                    st.caption(
                        "At least 100 future verified policy-comparison days "
                        "are requested for an initial statistical review. "
                        "No 70% accuracy claim or live model promotion."
                    )
                research = matched.get("v25_score_accuracy_research") or {}
                if research:
                    st.markdown("#### V25 score accuracy · Verified evidence")
                    baseline_shape = research.get("same_original_v8_verified_legs") or {}
                    slots = int(baseline_shape.get("verified_leg_slots") or 0)
                    st.caption(
                        f"{slots} original V8 leg results from matched V25 research sets. "
                        "The alternative score-shape replay is retrospective, "
                        "NOT frozen V25 accuracy."
                    )
                    if slots:
                        st.write({
                            "Original V8 top-one exact hits": baseline_shape.get("top_one_exact_hits"),
                            "Original V8 top-three exact hits": baseline_shape.get("top_three_exact_hits"),
                            "Original V8 top-five exact hits": baseline_shape.get("top_five_exact_hits"),
                            "Actual 4+ goal games": baseline_shape.get("high_goal_4_plus_observed"),
                            "Mean predicted 4+ goal chance": round(
                                baseline_shape.get("mean_over_3_5_probability") or 0, 3),
                            "Mean actual 4+ goal frequency": round(
                                baseline_shape.get("mean_over_3_5_observed") or 0, 3),
                        })
                    paired_shape = (research.get("prospective_same_leg_probability_comparison")
                                    or {})
                    paired_slots = int(paired_shape.get("verified_leg_slots") or 0)
                    if paired_slots:
                        st.caption(
                            f"True frozen V8–V25 probability comparison on "
                            f"{paired_slots} identical verified match legs "
                            "(lower loss is better)."
                        )
                        st.dataframe([
                            {
                                "Frozen model": label,
                                "Exact top-one hits": (paired_shape.get(key) or {}).get(
                                    "top_one_exact_hits"),
                                "Exact log loss": round((paired_shape.get(key) or {}).get(
                                    "mean_exact_score_log_loss"), 3),
                                "Full-score Brier": round((paired_shape.get(key) or {}).get(
                                    "mean_multiclass_score_brier"), 3),
                            }
                            for key, label in (
                                ("v8_on_exact_same_legs", "V8"),
                                ("v25_frozen_on_exact_same_legs", "V25"),
                            )
                        ], use_container_width=True, hide_index=True)
                    else:
                        st.info(
                            "No independently verified V25 score-probability pair "
                            "has both originally frozen distributions yet. "
                            "Historical V25 probabilities are never reconstructed."
                        )
                    replay = research.get("retrospective_same_source_sensitivity_replays") or {}
                    comparison_rows = [
                        {
                            "Retrospective score shape": key,
                            "Matched leg results": value.get("verified_leg_slots"),
                            "Top-one exact hits": value.get("top_one_exact_hits"),
                            "Log loss": round(value["mean_exact_score_log_loss"], 3),
                            "4+ goal probability": round(
                                value["mean_over_3_5_probability"], 3),
                        }
                        for key, value in replay.items()
                        if value.get("verified_leg_slots")
                    ]
                    if comparison_rows:
                        st.caption(
                            "Exploratory fixed mathematical sensitivity on archived "
                            "V8 grids; NOT a new future-tested V25 win rate."
                        )
                        st.dataframe(comparison_rows, use_container_width=True,
                                     hide_index=True)
                examples = (professional.get("outcomes") or [])[-8:]
                if examples:
                    st.dataframe(examples, use_container_width=True,
                                 hide_index=True)
    robust_auto=report.get("v14_independently_selected_five_doubles") or {}
    robust_frozen=int(robust_auto.get("frozen_pairs") or 0)
    robust_checked=int(robust_auto.get("verified_pairs") or 0)
    robust_wins=int(robust_auto.get("wins") or 0)
    st.caption("🔬 V14 independently selected two-match five doubles "
               "(research shadow only): "+str(robust_frozen)+
               " future pairs frozen · "+str(robust_wins)+
               "/"+str(robust_checked)+" verified hits. "
               "Both selected matches must finish and be independently verified; "
               "pending results are never counted as losses.")
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
    common_v12=(report.get("paired_v8_v12") or {}).get("verified_same_fixture_count",0)
    common_v13=(report.get("paired_v8_v13") or {}).get("verified_same_fixture_count",0)
    st.caption(f"Fair same-fixture verification: v8 vs v9 = {common} matches · v8 vs v10 = {common_v10} matches · v8 vs v12 = {common_v12} matches · v8 vs v13 = {common_v13} matches. Pending games are neither hits nor losses.")
    comparisons=doubles.get("same_pair_challengers") or {}
    if any((data.get("same_pair_verified_days") or 0)>0 for data in comparisons.values()):
        st.caption("Same frozen match-pair five-double comparisons: "+
                   " · ".join(name.upper()+": "+str(item["challenger_wins"])+"/"+
                              str(item["same_pair_verified_days"])+ " hits"
                              for name,item in comparisons.items() if item.get("same_pair_verified_days")) )
    if not compact:
        for name in ("v8","v8_near","v9","v10","v12","v13"):
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
                    "V12 score calibration shadows saved":int(status.get("v10_shadow_v12_saved") or 0),
                    "V13 hybrid shadows saved":int(status.get("v10_shadow_v13_saved") or 0),
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
        st.caption("Market-informed v9, multi-book v10, league-calibration v12, hybrid v13 and robust agreement v14 are experimental challengers, not proven superior models. V14's envelope is not a calibrated probability; only independently verified results can establish improvement.")
        st.divider()
        from llm_research_ui import render as render_five_ai
        render_five_ai(st)
