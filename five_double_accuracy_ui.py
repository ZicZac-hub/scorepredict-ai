"""Display the scheduled, source-validated exact-score double diagnosis."""
import json
import math
from pathlib import Path

from five_double_accuracy import OUTPUT


def coverage_research_rows(report, limit=12):
    """Read-only summary of original pair probability geometry, including pending sets.

    These numbers describe *model-estimated* coverage; they cannot be
    interpreted as historical or future betting win rates.
    """
    if not isinstance(report, dict) or not isinstance(report.get("sets"), list):
        return []
    if type(limit) is not int or limit < 1:
        return []
    rows = []
    for item in report["sets"]:
        if not isinstance(item, dict):
            continue
        geometry = item.get("full_grid_diagnosis") or {}
        if not isinstance(geometry, dict):
            continue
        selected = item.get("selected_five_model_coverage")
        optimal = geometry.get("optimal_five_model_coverage")
        if not all(type(v) in (int, float) and math.isfinite(v) and 0 <= v <= 1
                   for v in (selected, optimal)):
            continue
        if selected > optimal + 1e-9:
            continue
        required = geometry.get("minimum_combinations_for_target_model_coverage")
        if required is not None and (type(required) is not int or required < 1):
            continue
        gaps = item.get("source_feature_gaps") or []
        gaps = [leg for leg in gaps if isinstance(leg, dict)]
        rows.append({
            "Sydney date": str(item.get("sydney_date") or ""),
            "Result": item.get("status") if item.get("status") in ("hit", "miss", "pending") else "unknown",
            "Selected five model coverage": round(selected * 100, 2),
            "Best five model coverage": round(optimal * 100, 2),
            "Combos for 70% model coverage": required,
            "Legs missing true xG": sum(not g.get("verified_shot_xg_available", False) for g in gaps),
            "Legs without validated lineup adjustment": sum(
                not g.get("validated_lineup_adjustments_applied", False) for g in gaps),
        })
        if len(rows) >= limit:
            break
    return rows


def render(st, report_path=OUTPUT):
    try:
        raw = Path(report_path).read_text(encoding="utf-8")
        report = json.loads(raw)
        if not isinstance(report, dict) or report.get("schema_version") != 1:
            return
        observed = report["observed"]
        verified, hits, pending = (observed[k] for k in
                                   ("verified_sets", "hits", "pending_sets"))
        if (any(type(n) is not int or n < 0 for n in (verified, hits, pending))
                or hits > verified):
            return
        evaluated = str(report["evaluated_at_utc"])
        diagnosis = report["diagnosis"]
        if not isinstance(diagnosis, dict):
            return
    except (OSError, ValueError, KeyError, TypeError):
        return
    with st.expander("Why five exact-score doubles miss · verified evidence"):
        st.caption("Scheduled audit: " + evaluated +
                   " · original frozen V8 sets with matched regulation-time results.")
        st.write(f"Verified sets: {hits} hits / {verified} completed; "
                 f"{pending} pending.")
        if verified:
            st.caption(f"Observed set accuracy: {hits}/{verified} = {hits/verified:.2%}. "
                       "Pending sets are excluded from this denominator.")
            if verified < 100:
                st.caption(f"Small sample: {verified} completed sets cannot establish "
                           "a dependable future hit rate.")
        else:
            st.caption("No verified completed sets; observed accuracy is unavailable.")
        st.write("A hit requires both exact scores in the same one of five combinations. "
                 "Single-match double-chance percentages measure a different prediction.")
        coverage = diagnosis.get("model_coverage_range")
        if (isinstance(coverage, list) and len(coverage) == 2
                and all(type(n) in (int, float) and math.isfinite(n)
                        and 0 <= n <= 1 for n in coverage)
                and coverage[0] <= coverage[1]):
            st.write(f"These saved sets cover {coverage[0]:.2%}–{coverage[1]:.2%} "
                     "of their source models' outcomes under the independent-match assumption. "
                     "Estimated coverage is separate from observed accuracy.")
        st.write("70% exact-score-double accuracy has not been established. "
                 "Changing the order of five guesses cannot add probability to the source grids.")
        evidence_rows = coverage_research_rows(report)
        if evidence_rows:
            st.caption("Research comparison for original saved pairs, including upcoming/pending games. "
                       "All coverage percentages are MODEL ESTIMATES, not verified wins.")
            st.dataframe(evidence_rows, use_container_width=True, hide_index=True)
            st.caption("The 70% column reports how many distinct exact-score doubles would "
                       "be needed for 70% of the ORIGINAL MODEL'S probability mass. "
                       "It is not a betting recommendation, a promise, or proof of 70% real accuracy. "
                       "Missing xG and lineup evidence identifies potential research work, "
                       "not permission to invent adjustments.")
        sets = report.get("sets")
        for item in sets if isinstance(sets, list) else []:
            try:
                attribution = item.get("verified_result_attribution")
                if not attribution:
                    continue
                scores = " / ".join(leg["actual_score"] for leg in attribution["legs"])
                minimum = item["full_grid_diagnosis"].get(
                    "minimum_combinations_for_target_model_coverage")
                day, status = item["sydney_date"], item["status"]
                rank = attribution["actual_joint_score_rank_min"]
            except (AttributeError, KeyError, TypeError):
                continue
            st.caption(f"{day}: {status} · actual {scores} · "
                       f"joint score rank {rank}. " +
                       (f"The saved model needs {minimum} combinations to cover 70% estimated mass. "
                        if minimum is not None and report.get('target') == .70 else "") +
                       "This does not guarantee future success.")
        st.download_button("Download full accuracy diagnosis", raw.encode("utf-8"),
                           "five-double-accuracy-diagnosis.json", "application/json",
                           key="five_double_accuracy_diagnosis_download")
