"""Read-only display of truly frozen automated ZicZac five-double picks.

The automatic capture job runs outside Streamlit and archives predictions
before kickoff. This view never invents a new pair or overwrites one after
the game has started. Displayed coverage is a model estimate, not a win rate.
"""
from datetime import datetime,timezone
from pathlib import Path
from zoneinfo import ZoneInfo
import json

SYDNEY=ZoneInfo("Australia/Sydney")
ROOT=Path("data/v10_prospective")
RESULTS=ROOT/"performance.json"


def _time(value):
    try:
        d=datetime.fromisoformat(str(value).replace("Z","+00:00"))
        return d.replace(tzinfo=timezone.utc) if d.tzinfo is None else d.astimezone(timezone.utc)
    except (ValueError,TypeError,OverflowError):
        return None


def read_upcoming(root=ROOT,now=None):
    now=_time(now or datetime.now(timezone.utc))
    if now is None:
        return None
    today=now.astimezone(SYDNEY).date().isoformat()
    options=[]
    for path in sorted((Path(root)/"doubles").glob("*.json")):
        if path.stem<today:
            continue
        try:
            packet=json.loads(path.read_text(encoding="utf-8"))
            if packet.get("mode")!="prospective" or packet.get("status")!="frozen_pre_kickoff":
                continue
            captured=_time(packet["captured_at_utc"])
            first=_time(packet["first"]["kickoff_utc"])
            second=_time(packet["second"]["kickoff_utc"])
            if (not captured or not first or not second
                    or captured>=min(first,second)
                    or max(first,second)<=now
                    or packet.get("sydney_date")!=path.stem):
                continue
            scores=packet["five_doubles"]
            if (len(scores)!=5 or len({(x["score1"],x["score2"]) for x in scores})!=5
                    or not 0<float(packet.get("estimated_coverage",0))<1):
                continue
            options.append((packet["sydney_date"],first,packet))
        except (OSError,TypeError,ValueError,KeyError):
            continue
    return sorted(options,key=lambda item:(item[0],item[1]))[0][2] if options else None


def result_status(packet,report_path=RESULTS):
    try:
        report=json.loads(Path(report_path).read_text(encoding="utf-8"))
        for result in (report.get("five_doubles") or {}).get("rows") or []:
            if result.get("sydney_date")==packet["sydney_date"]:
                return result
    except (OSError,ValueError,TypeError):
        pass
    return {"status":"awaiting_verified_results"}


def render(st,root=ROOT):
    st.subheader("⚡ Automatically Frozen Five Doubles")
    st.caption("Created by ZicZac's pre-match research workflow; no manual research button needed.")
    packet=read_upcoming(root)
    if not packet:
        st.info("No qualified upcoming automatic five-double selection has been frozen yet. ZicZac can abstain when the match data or modeled coverage is insufficient.")
        return
    day=packet["sydney_date"]
    today=datetime.now(timezone.utc).astimezone(SYDNEY).date().isoformat()
    status="Today's research" if day==today else "Next scheduled research"
    st.markdown("**"+status+" · "+day+" (Sydney)**")
    st.caption("Frozen at "+_time(packet["captured_at_utc"]).astimezone(SYDNEY).strftime("%d %b · %I:%M %p")+" Sydney · original exact scores cannot be edited after kickoff.")
    c1,c2=st.columns(2)
    for col,label,leg in ((c1,"Match 1",packet["first"]),(c2,"Match 2",packet["second"])):
        with col:
            st.markdown("**"+label+"**")
            st.write(str(leg.get("home_team") or "Home")+" vs "+str(leg.get("away_team") or "Away"))
            t=_time(leg["kickoff_utc"])
            st.caption(t.astimezone(SYDNEY).strftime("%a %d %b · %I:%M %p"))
            source=leg.get("model_key","v8")
            st.caption("Research source: "+("Latest verified pre-kickoff form" if source=="v8_near" else "Original frozen v8 research"))
    st.metric("Five-double estimated coverage",f"{float(packet['estimated_coverage']):.2%}")
    st.caption("The figure above is the mathematical probability assigned to five distinct combinations assuming both matches are independent. It is NOT measured accuracy or a guaranteed win.")
    for index,item in enumerate(packet["five_doubles"],1):
        with st.container(border=True):
            a,b,c=st.columns([1,2,2])
            a.markdown("**#"+str(index)+"**")
            b.metric("Match 1",item["score1"])
            c.metric("Match 2",item["score2"])
            st.caption("Estimated chance both exact scores occur: "+f"{float(item['joint_probability']):.2%}")
    result=result_status(packet,Path(root)/"performance.json")
    if result.get("status")=="verified":
        if result.get("hit"):
            st.success("Verified: at least one of the five doubles matched BOTH final 90-minute scores.")
        else:
            st.warning("Verified: none of the five doubles matched both exact scores.")
        st.caption("Verified final scores: "+", ".join(result.get("actual_scores") or []))
    else:
        st.caption("Status: awaiting independent verification of both completed matches. Pending is not a loss.")
    if packet.get("challenger_exclusions"):
        with st.expander("Research model comparison availability"):
            st.caption("Other models are shown only when their pre-kickoff evidence is genuinely comparable.")
            st.write(packet["challenger_exclusions"])
    st.caption("Original frozen selection · every result is graded after completion · alternative models remain experimental.")
