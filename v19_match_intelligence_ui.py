"""V19 market and match intelligence: only source-validated research results."""
from datetime import datetime, timezone
from pathlib import Path

from persist_prematch import load_json
from v10_prospective import utc
from v19_match_intelligence import ROOT, VERSION, readiness
from v17_confidence import SYDNEY


def render(st, root=ROOT):
    st.markdown("#### 🧩 V19 Verified Match Intelligence")
    st.caption("Market-informed challengers are generated only when both "
               "originally selected matches have authentic archived, timestamped "
               "pre-kickoff bookmaker probability models. No fictional odds, "
               "shot-based xG or independent AI opinions are supplied.")
    info=load_json(Path(root)/"readiness.json",{})
    if not info:
        st.info("No V19 scheduled report is published yet. Research will abstain "
                "until the original pair and genuine two-leg market evidence "
                "have been collected before kickoff.")
        return
    quality=info.get("archived_prematch_market_models") or {}
    cols=st.columns(3)
    cols[0].metric("Archived V9 market models",str(quality.get("v9",0)))
    cols[1].metric("Archived V10 multi-book models",str(quality.get("v10",0)))
    cols[2].metric("Market-evidenced pair forecasts",
                   str(info.get("qualified_source_matched_original_double_sets",0)))
    performance=load_json(Path(root)/"performance.json",{})
    done=int(performance.get("verified_sets") or 0)
    pending=int(performance.get("pending_sets") or 0)
    st.caption("Genuinely verified two-match market challenger: "+
               (f"{performance.get('market_wins',0)}/{done} hits"
                if done else "not measurable")+
               f" · {pending} pending. Compare only with V8 on identical fixtures.")
    if done:
        st.caption("Original V8 on exactly the same pairs: "+
                   str(performance.get("original_same_pair_wins",0))+"/"+str(done)+
                   " · Market 95% Wilson interval: "+
                   (f"{performance['market_95_wilson'][0]:.1%}–"
                    f"{performance['market_95_wilson'][1]:.1%}"
                    if performance.get("market_95_wilson") else "not available"))
    with st.expander("Evidence readiness and missing sources", expanded=True):
        st.write({
            "Two-leg valid AI consensus records":
                info.get("qualified_two_match_ai_pairs",0),
            "Licensed shot-quality xG integrated":
                "No" if not info.get("true_shot_quality_xg_feed_integrated") else "Yes",
            "Trained team-news and lineup coefficients":
                "No" if not info.get("timestamped_lineup_adjustment_trained") else "Yes",
            "Archived odds metadata independently replayable from raw HTTP":
                "No" if not info.get("market_quotes_raw_http_response_audited") else "Yes",
            "Sportmonks prediction requests enabled":
                bool(info.get("sportmonks_predictions_enabled")),
            "V8 independently verified single outcomes":
                info.get("original_v8_verified_singles",0)
        })
        st.caption("The repository can verify saved market model metadata, "
                   "but no V19 result proves the raw odds response has been "
                   "independently archived. Paid provider access is unchanged.")
    available=[]
    now=datetime.now(timezone.utc)
    for path in sorted((Path(root)/"doubles").glob("*.json")):
        row=load_json(path,{})
        if row.get("version")!=VERSION:
            continue
        try:
            if all(utc(row[side]["kickoff_utc"])>now for side in ("first","second")):
                available.append(row)
        except (KeyError,TypeError,ValueError):
            continue
    if not available:
        st.info("No upcoming V19 two-leg market-evidenced exact-score forecast. "
                "This is correct when one or both legs have no qualifying odds. "
                "The original V8 five doubles remain available separately.")
    else:
        chosen=available[0]
        st.markdown("**Experimental V19 five exact-score doubles — "+
                    str(chosen["sydney_date"])+" Sydney**")
        st.caption("Same two fixtures as frozen original; "+
                   chosen["market_model"].upper()+
                   " source used on BOTH legs; research only.")
        for side in ("first","second"):
            leg=chosen[side]
            st.write(str(leg.get("home_team") or "Team "+str(leg.get("home_team_id"))) +
                     " vs " +str(leg.get("away_team") or "Team "+str(leg.get("away_team_id"))))
            provenance=chosen[side+"_market_provenance"]
            st.caption(str(provenance.get("bookmaker"))+" · archived market quote "+
                       utc(provenance["snapshot_utc"]).astimezone(SYDNEY).strftime("%d %b, %I:%M %p"))
        st.metric("V19 estimated exact five-double coverage",
                  f"{float(chosen['estimated_coverage']):.2%}")
        st.caption("The modelled mass of five mutually exclusive exact outcomes. "
                   "This is not empirical win accuracy, or evidence of 70% success.")
        for i,opt in enumerate(chosen["five_doubles"],1):
            st.write(f"{i}. {opt['score1']} / {opt['score2']}"+
                     f" · estimated {float(opt['joint_probability']):.2%}")
    with st.expander("Remaining actions for genuine prediction improvement"):
        for item in info.get("data_actions_required") or []:
            st.write("• "+str(item))
        st.caption("V19 has not changed production predictions, purchased "
                   "subscriptions, raised AI API spending, or promoted "
                   "a less-tested challenger.")
