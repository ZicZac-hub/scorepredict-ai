"""Display only actually captured Sportmonks Predictions add-on probabilities."""
import json
from pathlib import Path

ROOT=Path("data/independent_football")


def _read(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError,ValueError,TypeError):
        return {}


def render(st, root=ROOT):
    st.markdown("### ⚽ Sportmonks · Independent Correct-Score Research")
    st.caption("Different provider's licensed probabilities, matched by teams and kickoff. "
               "A Sportmonks fixture ID is not interchangeable with API-Football's ID.")
    status=_read(Path(root)/"status.json")
    if not status.get("enabled"):
        st.info("Sportmonks prediction add-on is not activated in GitHub Actions. "
                "Existing Sportmonks fixture access does not automatically include predictions.")
    elif not status.get("token_present"):
        st.warning("Sportmonks prediction collector requires a GitHub Actions API secret.")
    elif status.get("error_type"):
        st.warning("Sportmonks predictions not available or request failed ("+
                   str(status.get("error_type"))+"). Check the plan/add-on and API quota.")
    else:
        st.caption("Latest check: "+str(status.get("checked_at_utc","unknown"))+
                   " UTC · frozen reports: "+str(status.get("captured",0))+
                   " · unmatched fixture identities: "+str(status.get("unmatched",0)))
    report=_read(Path(root)/"evaluation.json")
    n=int(report.get("verified_matched_fixtures") or 0)
    if n:
        st.write({"Same-fixture verified sample":n,
                  "Sportmonks exact hits":report.get("sportmonks_top_score_hits"),
                  "Original ZicZac exact hits":report.get("original_statistical_hits")})
    else:
        st.caption("No verified same-fixture Sportmonks-vs-ZicZac results yet. "
                   "There is no evidence of higher exact-score accuracy.")
    paths=sorted((Path(root)/"forecasts").glob("*.json"),reverse=True)
    shown=0
    for path in paths:
        row=_read(path)
        if not row.get("research_only"):
            continue
        with st.expander(str(row.get("home_team"))+" vs "+str(row.get("away_team"))+
                         " · Sportmonks score probabilities"):
            st.caption("Frozen before kickoff: "+str(row.get("captured_at_utc"))+
                       ". Sportmonks fixture ID: "+str(row.get("sportmonks_fixture_id")))
            st.write({"Source top-5 exact scores":[
                         {"Score":score,"Probability":round(100*row["score_probabilities"][score],2)}
                         for score in row.get("top_scores",[])],
                      "ZicZac original saved score":row.get("original_statistical_score")})
            st.caption("Probability estimates, not observed win percentages. "
                       "The original frozen prediction is unchanged.")
        shown+=1
        if shown>=3:
            break
