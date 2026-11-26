"""Display the same evidence-gated final selection on both doubles pages."""


_STATUS = {
    "no_decision": "Waiting for a frozen pre-match slate comparison.",
    "awaiting_validation": "Final selection withheld while prospective results accumulate.",
    "incomplete_slate": "Final selection withheld because the full eligible slate could not be compared.",
    "expired": "The latest frozen selection has passed its pre-match release window.",
    "evidence_unavailable": "The saved comparison or verification evidence could not be authenticated.",
}


def render(st):
    from final_five_doubles import presentation

    st.subheader("Final Five Doubles")
    report = presentation()
    state, decision, validation = (report[key] for key in ("state", "decision", "validation"))
    available = state == "available" and validation["ready"]
    if not available:
        st.info(_STATUS.get(state, _STATUS["awaiting_validation"]))
    st.caption("A set hits when one of its five rows matches both regulation-time scores.")

    verified = validation["verified"]
    if verified:
        wins = validation["wins"]
        st.metric("Verified prospective hit rate", f"{wins / verified:.1%}")
        st.caption(f"{wins} wins across {verified} verified sets.")
    else:
        st.caption("No independently verified final-policy sets have settled yet.")
    interval = validation["wilson_95"]
    if interval is not None:
        st.caption(f"95% Wilson interval {interval[0]:.1%}–{interval[1]:.1%}.")
    st.caption(f"Initial review · {validation['first_cohort_verified']}/{validation['minimum_verified']} "
               "of the first issued sets verified.")
    cohort = validation["first_cohort_wilson_95"]
    all_issued = validation["publication_wilson_95"]
    if cohort is not None and all_issued is not None:
        st.caption(f"95% Wilson lower bounds · {cohort[0]:.1%} for the initial cohort · "
                   f"{all_issued[0]:.1%} for all issued sets, including pending sets in the denominator.")
    st.caption(f"Release requires a 95% Wilson lower bound of {validation['target']:.0%} for both the "
               f"initial cohort and all issued sets, issuance on at least "
               f"{validation['minimum_issued_fraction']:.0%} of reviewed slates, and zero failed evidence checks.")
    st.caption(f"{validation['pending']} pending · {validation['abstained']} abstained · "
               f"{validation['invalid']} failed evidence checks.")

    if decision is None or decision["eligible_pairs"] is None:
        return
    st.caption(f"Sydney matchday {decision['sydney_day']} · {decision['eligible_fixtures']} eligible fixtures · "
               f"{decision['compared_pairs']}/{decision['eligible_pairs']} eligible fixture pairs compared.")
    if decision["methods"]:
        st.caption("Methods compared · " + ", ".join(method.replace("_", " ") for method in decision["methods"]))
    if decision["evidence_counts"]:
        counts = decision["evidence_counts"]
        st.caption(f"Evidence used on both legs · shot xG in {counts['xg']} pairs · "
                   f"bookmaker markets in {counts['market']} pairs.")
    if decision["estimate"] is not None:
        st.metric("Selected set’s model-estimated coverage", f"{decision['estimate']:.2%}")
        st.caption("Model coverage adds the five joint score probabilities under match independence.")

    if available:
        first, second = decision["names"]
        st.write(f"**Match 1** · {first}")
        st.write(f"**Match 2** · {second}")
        st.dataframe([
            {"Double": index, "Match 1 score": row["score1"], "Match 2 score": row["score2"],
             "Estimated probability": f"{row['joint_probability']:.2%}"}
            for index, row in enumerate(decision["five_doubles"], 1)
        ], hide_index=True, use_container_width=True)
