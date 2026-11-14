"""Sydney-dated, pre-kickoff Prediction of the Day for the Daily page.

The selection is independent of the sidebar's optional legacy model.
Never invents injuries, odds, or a fixture. The first view freezes a
pre-match candidate IN THE CURRENT SESSION only; it is not yet a
cross-device verified prospective record.
"""
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
import html
import math

SYDNEY=ZoneInfo("Australia/Sydney")
MODEL="ziczac-evidence-ensemble-v8-experimental"
STATE_KEY="ziczac_daily_research_pick_v1"
ODDS_KEY="ziczac_daily_market_comparison_v1"


def _when(raw):
    try:
        d=datetime.fromisoformat(str(raw).replace("Z","+00:00"))
        return d.replace(tzinfo=timezone.utc) if d.tzinfo is None else d.astimezone(timezone.utc)
    except (ValueError,TypeError,OverflowError):
        return None


def _quality(row):
    model=row.get("result") or {}
    recent=min(len(model.get("home_last10") or []),
               len(model.get("away_last10") or []))
    venue=min(len(model.get("home_home10") or []),
              len(model.get("away_away10") or []))
    # On v8 the model explicitly reports this; on v6 count available methods.
    contributors=int(model.get("component_count") or 1)
    p=float(model.get("scores",[(-1,-1,0)])[0][2])
    outcome=max(float(model.get(k,0)) for k in ("home_win","draw","away_win"))
    return (min(recent,10)>=5, min(venue,10)>=3, contributors>=2,
            min(recent,10),min(venue,10),contributors,outcome,p)


def daily_candidates(rows,as_of):
    """Choose today's strongest researched upcoming match, else nearest date."""
    now=_when(as_of)
    if now is None:
        raise ValueError("as_of must be an aware timestamp")
    today=now.astimezone(SYDNEY).date()
    suitable=[]
    for row in rows:
        game=row.get("fixture") or {}
        result=row.get("result") or {}
        kickoff=_when(row.get("kickoff") or game.get("utcDate"))
        if (not kickoff or kickoff<=now or kickoff.astimezone(SYDNEY).date()<today
                or not game.get("id") or not result.get("scores")):
            continue
        top=result["scores"][0]
        try:
            if (len(top)<3 or any(type(v) is not int or v<0 for v in top[:2])
                    or not math.isfinite(float(top[2])) or float(top[2])<=0):
                continue
        except (ValueError,TypeError,IndexError):
            continue
        suitable.append((row,kickoff))
    if not suitable:
        return []
    nearest=min(k.astimezone(SYDNEY).date() for _,k in suitable)
    return [r for r,k in sorted(suitable,
                                key=lambda item:(_quality(item[0]),
                                  -item[1].timestamp()),reverse=True)
            if k.astimezone(SYDNEY).date()==nearest]


def build_snapshot(row,as_of,forecast=None):
    """Freeze a genuine model prediction computed strictly before kickoff."""
    now=_when(as_of)
    game=row["fixture"]
    kickoff=_when(row.get("kickoff") or game.get("utcDate"))
    result=forecast or row.get("result")
    if not now or not kickoff or now>=kickoff or not result or not result.get("scores"):
        return None
    top=result["scores"][0]
    home=game.get("homeTeam") or {}
    away=game.get("awayTeam") or {}
    if not all((home.get("name"),away.get("name"),home.get("id"),away.get("id"))):
        return None
    scores=[[int(h),int(a),float(p)] for h,a,p in result["scores"][:5]]
    if any(p<0 or p>1 or not math.isfinite(p) for _,_,p in scores):
        return None
    outcomes={name:float(result[key]) for name,key in
              (("home","home_win"),("draw","draw"),("away","away_win"))}
    if any(not math.isfinite(p) or p<0 or p>1 for p in outcomes.values()):
        return None
    snapshot={
        "fixture_identity":(str(game.get("provider") or "football-data.org"),str(game["id"])),
        "fixture":game,
        "home":str(home["name"]),"away":str(away["name"]),
        "league":str(row.get("league") or (game.get("competition") or {}).get("name") or "Football"),
        "kickoff_utc":kickoff.isoformat(),"captured_at_utc":now.isoformat(),
        "prediction_date":kickoff.astimezone(SYDNEY).date().isoformat(),
        "model":str(result.get("model_version") or "unknown"),
        "exact_score":scores[0][:2],"top_scores":scores,"probability":scores[0][2],
        "home_xg":float(result["home_xg"]),"away_xg":float(result["away_xg"]),
        "outcomes":outcomes,"home_recent":len(result.get("home_last10") or []),
        "away_recent":len(result.get("away_last10") or []),
        "home_venue":len(result.get("home_home10") or []),
        "away_venue":len(result.get("away_away10") or []),
        "h2h":len(result.get("h2h_last10") or []),
        "components":list((result.get("component_weights") or {}).keys()),
        "agreement":result.get("component_top_score_agreement"),
        "component_count":result.get("component_count") or 1,
        "market_comparison_available":game.get("provider")=="API-Football",
        "status":"frozen_in_session",
        }
    return snapshot


def choose_daily_snapshot(rows,session,as_of=None):
    """Do not recalculate an already shown candidate within this user session."""
    now=_when(as_of or datetime.now(timezone.utc))
    if now is None:
        return None
    today=now.astimezone(SYDNEY).date().isoformat()
    previous=session.get(STATE_KEY)
    if (previous and previous.get("prediction_date")>=today
            and _when(previous.get("captured_at_utc"))
            and _when(previous.get("kickoff_utc"))
            and _when(previous["captured_at_utc"])<_when(previous["kickoff_utc"])):
        return previous
    for row in daily_candidates(rows,now):
        try:
            from research_ensemble_v8 import predict as research_v8
            packet=row.get("evidence") or {}
            prior=packet.get("history") or row.get("past") or []
            league=packet.get("league_history") or row.get("past") or prior
            forecast=research_v8(row["fixture"],prior,league_history=league,as_of=now)
            if forecast is None:
                continue
            # Require actual observed last-five evidence on BOTH clubs.
            if min(len(forecast.get("home_last10") or []),
                   len(forecast.get("away_last10") or []))<5:
                continue
            frozen=build_snapshot(row,now,forecast)
            if frozen:
                session[STATE_KEY]=frozen
                session.pop(ODDS_KEY,None)
                return frozen
        except (ValueError,TypeError,KeyError,ZeroDivisionError,OverflowError):
            continue
    return None


def _date_format(stamp,fmt):
    return _when(stamp).astimezone(SYDNEY).strftime(fmt)


def _label(snapshot,now):
    as_of=_when(now).astimezone(SYDNEY).date()
    date=datetime.fromisoformat(snapshot["prediction_date"]).date()
    return "TODAY’S HIGHLY RESEARCHED PICK" if date==as_of else "NEXT RESEARCHED MATCH PICK"


def render(st,rows,session,api_key=None,now=None):
    now=_when(now or datetime.now(timezone.utc))
    st.markdown("""<style>
    .zz-pod{border:1px solid rgba(124,109,255,.72);border-radius:20px;
       background:linear-gradient(120deg,#101b3e 0%,#20194a 72%,#132a49 100%);
       padding:clamp(16px,3vw,30px);box-shadow:0 7px 30px #141b44aa;margin:6px 0 16px}
    .zz-pod-kicker{color:#b9abff;font-size:.8rem;font-weight:800;letter-spacing:.07em}
    .zz-pod-title{color:#fff;font-weight:900;font-size:clamp(1.35rem,3vw,2.1rem);line-height:1.26;margin:13px 0}
    .zz-pod-score{color:#fff;font-size:clamp(2.1rem,7vw,4rem);font-weight:950;
       letter-spacing:-.06em;line-height:1.2}
    .zz-pod-meta{color:#c6d2f8;font-size:clamp(.8rem,1.75vw,1rem);margin:6px 0}
    </style>""",unsafe_allow_html=True)
    st.subheader("🏆 Highly Researched Prediction of the Day")
    st.caption("One pre-match exact-score forecast selected using verified form and venue evidence · Enhanced v8 research · Optional market-informed v9 comparison.")
    snapshot=choose_daily_snapshot(rows,session,now)
    if not snapshot:
        st.info("No sufficiently researched upcoming match is available from the connected football feeds. Expand your fixture dates or monitored competitions; a score will never be invented.")
        return
    title=_label(snapshot,now)
    score="-".join(map(str,snapshot["exact_score"]))
    league=html.escape(snapshot["league"])
    homes=html.escape(snapshot["home"])
    aways=html.escape(snapshot["away"])
    matchdate=html.escape(_date_format(snapshot["kickoff_utc"],"%a %d %b %Y · %I:%M %p Sydney"))
    st.markdown(
        '<div class="zz-pod"><div class="zz-pod-kicker">🏆 '+title+
        ' · ENHANCED MODEL v8</div><div class="zz-pod-meta">'+league+' · '+matchdate+
        '</div><div class="zz-pod-title">'+homes+' <span style="color:#9489ff">VS</span> '+aways+
        '</div><div class="zz-pod-score">'+score+'</div><div class="zz-pod-meta">Most likely single exact score · Estimated chance '+
        f'{snapshot["probability"]:.1%}'+
        ' · Original forecast frozen before kickoff</div></div>',unsafe_allow_html=True)
    a,b,c=st.columns(3)
    a.metric("Home win",f'{snapshot["outcomes"]["home"]:.1%}')
    b.metric("Draw",f'{snapshot["outcomes"]["draw"]:.1%}')
    c.metric("Away win",f'{snapshot["outcomes"]["away"]:.1%}')
    st.caption(f'Estimated goals: {snapshot["home_xg"]:.2f} home · {snapshot["away_xg"]:.2f} away · '
               f'Model components: {int(snapshot["component_count"])} · '
               f'Selected {_date_format(snapshot["captured_at_utc"],"%d %b %I:%M %p")} Sydney.')
    if now>=_when(snapshot["kickoff_utc"]):
        st.warning("Kickoff has passed. This original estimate is unchanged; check verified match results in Performance.")
    else:
        st.caption("Locked within this browser session before kickoff. A new browser session may make a different selection; this card is not yet a globally archived Prediction of the Day.")
    with st.expander("🔬 Why this match? Verified research and alternative scores"):
        st.write("**Research coverage:** "+snapshot["home"]+" last "+str(snapshot["home_recent"])+" matches; "+
                 snapshot["away"]+" last "+str(snapshot["away_recent"])+" matches.")
        st.write("**Venue evidence:** "+str(snapshot["home_venue"])+" home matches and "+
                 str(snapshot["away_venue"])+" away matches; "+str(snapshot["h2h"])+" recorded head-to-head matches.")
        if snapshot["components"]:
            st.write("**Contributors:** "+", ".join(snapshot["components"]))
        if snapshot["agreement"] is not None:
            st.caption("Top-score agreement: "+str(snapshot["agreement"])+"/"+str(snapshot["component_count"]))
        st.dataframe([{"Exact score":f"{h}-{a}","Estimated probability":f"{p:.1%}"}
                      for h,a,p in snapshot["top_scores"]],hide_index=True,use_container_width=True)
        st.caption("Top-score probability measures one particular result, not overall certainty. Missing injuries, confirmed lineups, true xG and other inputs are not invented.")
    if not snapshot["market_comparison_available"]:
        st.caption("Live v9 market check is available only for supported API-Football fixtures.")
        return
    if not api_key:
        st.caption("Add your API-Football key in Streamlit Secrets for an optional v9 market check.")
        return
    if now>=_when(snapshot["kickoff_utc"]):
        return
    if st.button("📈 Compare with Market-Informed v9",key="ziczac_daily_check_v9",
                 use_container_width=True):
        item={"fixture_identity":snapshot["fixture_identity"]}
        try:
            from api_football import _request
            from market_score_v9 import select_market_response
            packet=_request(api_key,"/odds",{"fixture":snapshot["fixture"]["id"]},with_metadata=True)
            metadata=packet.get("metadata") or {}
            if metadata.get("stale") or metadata.get("freshness_known") is False:
                item["reason"]="The provider odds response is stale or its retrieval time is unknown."
            else:
                market=select_market_response(packet.get("data") or [],snapshot["fixture"],as_of=datetime.now(timezone.utc))
                if market:
                    # Market projection on the original pre-match score grid is
                    # shown as a separate comparison, without changing the
                    # already frozen v8 score above.
                    from market_score_v9 import _project, MARKET_WEIGHT
                    import numpy as np
                    from research_ensemble_v8 import predict as research_predict
                    row=next((r for r in rows if (str(r["fixture"].get("provider")),str(r["fixture"].get("id")))==
                              tuple(snapshot["fixture_identity"])),None)
                    if row:
                        evidence=row.get("evidence") or {}
                        past=evidence.get("history") or row.get("past") or []
                        league=evidence.get("league_history") or row.get("past") or past
                        base=research_predict(row["fixture"],past,league_history=league,
                                              as_of=_when(snapshot["captured_at_utc"]))
                        if base is not None:
                            adjusted=_project(base["matrix"],market)
                            mixture=(1-MARKET_WEIGHT)*base["matrix"]+MARKET_WEIGHT*adjusted
                            mixture/=mixture.sum()
                            rank=int(np.argmax(mixture))
                            i,j=np.unravel_index(rank,mixture.shape)
                            item.update({"score":[int(i),int(j)],
                                         "probability":float(mixture[i,j]),
                                         "market":market})
                        else:
                            item["reason"]="The original pre-kickoff model evidence could not be replayed."
                    else:
                        item["reason"]="This match is not present in the current verified research dataset."
                else:
                    item["reason"]="No complete recent pre-match 1X2 odds were returned. The v8 forecast above remains unchanged."
        except (RuntimeError,TypeError,ValueError,KeyError,ZeroDivisionError) as exc:
            item["reason"]="The market comparison is unavailable ("+type(exc).__name__+")."
        session[ODDS_KEY]=item
    outcome=session.get(ODDS_KEY) or {}
    if tuple(outcome.get("fixture_identity",()))!=tuple(snapshot["fixture_identity"]):
        return
    if outcome.get("market"):
        m=outcome["market"]
        st.success("v9 comparison: "+str(outcome["score"][0])+"–"+str(outcome["score"][1])+
                   " · "+f'{outcome["probability"]:.1%}'+
                   " · Provider: "+str(m["bookmaker"])+" · Snapshot "+str(m["snapshot_utc"]))
        st.caption("This is a separate experimental market-adjusted estimate, NOT an update to the frozen Prediction of the Day.")
    elif outcome.get("reason"):
        st.info(str(outcome["reason"]))
