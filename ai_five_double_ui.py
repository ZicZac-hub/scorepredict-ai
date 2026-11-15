"""Standalone manual AI Five-Double Studio for ZicZac Streamlit.

Use the SAME original model score tables as Five Doubles, optionally adding
previously frozen AI opinions plus owner-authorised on-demand AI providers.
No scraping, no fabricated data, no overwrite of original match forecasts.
"""
from datetime import datetime, timedelta, timezone
import csv
import io
from functools import lru_cache

from ai_five_double_engine import eligible, identity, merged_five, pair_check, teams, SYDNEY
from on_demand_ai import (frozen_for_fixture, private_gate_ready, run_selected,
                          server_qwen_url, server_settings, utc)
from llm_research_panel import PROVIDERS
from smart_double_ai import _scores
from goal_research_v20 import goal_profile, goal_form_challenger
from v21_provider_evidence import valid_snapshot


def archive_rows(now=None, root=None):
    """Read immutable pre-match v8 research for future fixtures if live API fails."""
    from v10_prospective import ROOT, read_latest
    root = ROOT if root is None else root
    clock = utc(now or datetime.now(timezone.utc))
    output = {}
    try:
        originals=read_latest(root)
    except (OSError,ValueError,TypeError):
        return []
    for (provider,fid,model),saved in originals.items():
        if model not in ("v8", "v8_near"):
            continue
        if not saved.get("home_team") or not saved.get("away_team"):
            continue
        try:
            kickoff=utc(saved["kickoff_utc"])
            captured=utc(saved["captured_at_utc"])
            if not kickoff or not captured or captured>=kickoff or kickoff<=clock:
                continue
            row={
                "fixture":{
                    "id":saved["fixture_id"],"provider":provider,
                    "utcDate":kickoff.isoformat(),
                    "homeTeam":{"id":saved["home_team_id"],"name":saved["home_team"]},
                    "awayTeam":{"id":saved["away_team_id"],"name":saved["away_team"]},
                    "competition":{"code":saved.get("competition") or "",
                                   "name":saved.get("competition") or ""}
                },
                "kickoff":kickoff.isoformat(),
                "result":{"scores":saved["top10_scores"],
                          "model_version":saved["model_version"],
                          "outcome_probabilities":saved.get("outcome_probabilities") or {},
                          "home_xg":(saved.get("expected_goals") or {}).get("home"),
                          "away_xg":(saved.get("expected_goals") or {}).get("away")},
                "league":saved.get("competition") or "",
                "past":[],"archive_frozen":True,
                "frozen_at_utc":captured.isoformat(),
                "history_matches_reported":saved.get("history_matches")
            }
        except (KeyError,ValueError,TypeError):
            continue
        if not eligible(row,now=clock):
            continue
        key=identity(row)
        other=output.get(key)
        if other is None or (model=="v8_near" and
                              other.get("source_model")!="v8_near"):
            row["source_model"]=model
            output[key]=row
    return sorted(output.values(),key=lambda row:row["kickoff"])


@lru_cache(maxsize=4)
def _minute_archived_candidates(minute):
    """One read per minute rather than rescanning hundreds of JSON files
    for every Streamlit button or selectbox rerun."""
    return archive_rows()


def options(rows, *, now=None, archived=None):
    """Prefer valid current modelled fixtures; fill missing fixtures from archive."""
    chosen={}
    for row in rows or []:
        if isinstance(row,dict) and eligible(row,now=now):
            chosen.setdefault(identity(row),row)
    for row in archived if archived is not None else archive_rows(now=now):
        if eligible(row,now=now):
            chosen.setdefault(identity(row),row)
    return sorted(chosen.values(),key=lambda row:(utc(row["kickoff"]),identity(row)))


def label(row):
    fixture=row["fixture"]
    kickoff=utc(row["kickoff"]).astimezone(SYDNEY)
    home=fixture["homeTeam"]["name"]
    away=fixture["awayTeam"]["name"]
    source="Saved v8" if row.get("archive_frozen") else "Current model"
    return (home+" vs "+away+" · "+kickoff.strftime("%d %b, %I:%M %p")+
            " Sydney · "+source+" · "+identity(row)[1])


def _history(row,histories):
    """Modelled fixtures provide a restricted, dated source history."""
    past=row.get("past") or (histories or {}).get(row.get("league"),[]) or []
    return [m for m in past if
            (m.get("provider") or "football-data.org")==identity(row)[0]]


def _as_research(archive, fresh):
    votes={}
    source={}
    # A prior write-once validated snapshot remains visible even when the
    # provider cannot be called today.
    for name,vote in (archive or {}).get("votes",{}).items():
        if isinstance(vote,dict) and isinstance(vote.get("score"),list):
            votes[name]=vote
            source[name]="frozen GitHub pre-match"
    if fresh and fresh.get("status")=="completed":
        for name,vote in (fresh.get("votes") or {}).items():
            if isinstance(vote,dict) and isinstance(vote.get("score"),list):
                votes[name]=vote
                source[name]="new Streamlit API request"
    return {"status":"completed","votes":votes,"sources":source}


def analyze_pair(first, second, histories, *, configs=None, use_live=False,
                 request=None, now=None, qwen_url=None, archiver=frozen_for_fixture,
                 selection_mode="probability", provider_packets=None):
    """Evidence-backed, session-only research for one pre-match pair."""
    now = utc(now or datetime.now(timezone.utc))
    issue = pair_check(first, second, now=now)
    if issue:
        return {"status": "invalid_pair", "reason": issue}
    pair_rows = (first, second)
    provider_packets = list(provider_packets or (None,None))
    if len(provider_packets)!=2:
        return {"status":"invalid_provider_packet","reason":"Need exactly two evidence packets"}
    provider_packets = [packet if valid_snapshot(packet,row["fixture"],now=now)
                        else None for row,packet in zip(pair_rows,provider_packets)]
    dated_history = [_history(row, histories) for row in pair_rows]
    profiles = [goal_profile(row["fixture"], dated_history[i], now)
                for i, row in enumerate(pair_rows)]
    # The goal-form shadow only runs when two authentic full model grids and
    # sufficient dated team+venue results exist. No guessed high-score boost.
    research_models = [goal_form_challenger(row, dated_history[i], now)
                       for i, row in enumerate(pair_rows)]
    effective_rows = pair_rows
    if selection_mode == "goal_form_research":
        missing = [str(i+1)+": "+str(m.get("reason") or "insufficient evidence")
                   for i, m in enumerate(research_models) if not m.get("available")]
        if missing:
            return {"status": "insufficient_research_evidence",
                    "reason": "Cannot responsibly produce research-adjusted scores for both matches. "
                              + " | ".join(missing)
                              + " Search & model the fixtures to obtain full dated history, "
                                "or select the original baseline."}
        effective_rows = []
        for row, model in zip(pair_rows, research_models):
            adjusted = dict(row.get("result") or {})
            adjusted["scores"] = model["scores"]
            adjusted["matrix"] = model["matrix"]
            adjusted["home_xg"], adjusted["away_xg"] = model["research_expected_goals"]
            adjusted["model_version"] = model["model"]
            effective_rows.append({**row, "result": adjusted})
    archives = [archiver(row["fixture"]) for row in pair_rows]
    fresh = [None, None]
    errors = [{}, {}]
    if use_live and configs:
        for i, row in enumerate(pair_rows):
            already = (archives[i] or {}).get("votes") or {}
            remaining = {p: cfg for p, cfg in configs.items() if p not in already}
            if not remaining:
                continue
            try:
                result = run_selected(
                    row["fixture"], dated_history[i], row.get("result") or {}, {},
                    remaining, now=now, request=request, qwen_url=qwen_url,
                    team_context={**(row.get("research_context") or {}),
                                  "v21_provider_snapshot":provider_packets[i]})
                fresh[i] = result
                errors[i] = result.get("errors") or {}
            except Exception:
                errors[i] = {"research": {"reason": "provider_request_failed"}}
    evidence = [_as_research(archives[i], fresh[i]) for i in (0, 1)]
    original = merged_five(first, second, evidence[0], evidence[1], now=now,
                           selection_mode="probability")
    # Until a source-matched AI panel beats statistical-only picks on
    # independent future games, V21 never promotes AI opinions to numerical
    # weighting. The validated votes remain visible for review.
    applied_evidence = ([{},{}] if selection_mode=="goal_form_research"
                        else evidence)
    combined = merged_five(effective_rows[0], effective_rows[1],
                           applied_evidence[0], applied_evidence[1], now=now,
                           selection_mode=("probability" if
                                           selection_mode == "goal_form_research"
                                           else selection_mode))
    if combined["status"] != "ready":
        return combined
    if selection_mode == "goal_form_research":
        # The two values use different distributions; never imply the
        # challenger coverage is an observed or directly comparable win rate.
        combined["selection_mode"] = "goal_form_research"
        combined["research_model_coverage"] = combined["selected_model_coverage"]
        combined["baseline_coverage"] = original.get("selected_model_coverage")
        combined["baseline_diversity"] = original.get("selected_diversity")
        combined["research_coverage_not_comparable_without_validation"] = True
    return {
        "status": "ready", "generated_at_utc": now.isoformat(),
        "fixtures": [identity(first), identity(second)],
        "teams": [teams(first), teams(second)],
        "statistical_models": [(r.get("result") or {}).get("model_version")
                               for r in pair_rows],
        "data_source": ["archived statistical forecast" if r.get("archive_frozen")
                        else "current statistical forecast" for r in pair_rows],
        "past_matches_available": [len(h) for h in dated_history],
        "archived_history_counts": [r.get("history_matches_reported")
                                    for r in pair_rows],
        "historical_profiles": profiles, "research_models": research_models,
        "v21_provider_packets": provider_packets,
        "research_ai_votes_weighted": selection_mode!="goal_form_research",
        "provider_context": [r.get("research_context") or {} for r in pair_rows],
        "ai_research": evidence, "api_errors": errors,
        "model_and_ai_result": combined,
        "new_ai_requested": bool(use_live and configs),
        "session_only": True, "original_forecasts_overwritten": False,
        "independent_prospective_accuracy_record": False,
        "provider_disclosure": "Verified provider data and available named AI APIs only; no unlimited web browsing"
    }


def _csv_bytes(data):
    stream=io.StringIO()
    writer=csv.writer(stream)
    writer.writerow(["Double","Match 1 exact score","Match 2 exact score",
                     "Model-only joint estimate","AI votes for Match 1 score",
                     "AI votes for Match 2 score"])
    for i,d in enumerate(data["doubles"],1):
        writer.writerow([i,d["Score 1"],d["Score 2"],format(d["Model joint"],".6f"),
                         d["AI endorsements 1"],d["AI endorsements 2"]])
    return stream.getvalue().encode("utf-8")


def _display(st,report,first,second):
    calc=report["model_and_ai_result"]
    challenger = calc.get("selection_mode") == "balanced_challenger"
    form_research = calc.get("selection_mode") == "goal_form_research"
    method = ("research goal-form challenger" if form_research else
              "balanced challenger" if challenger else "original baseline")
    st.markdown("### ✅ Final five exact-score doubles · " + method)
    st.caption("Match 1: "+label(first))
    st.caption("Match 2: "+label(second))
    st.caption("Research compiled "+str(report["generated_at_utc"])+
               " UTC · Original forecasts unchanged · session-only comparison.")
    a,b=st.columns(2)
    a.metric("Original coverage estimate", f"{calc['baseline_coverage']:.2%}")
    b.metric("Selected model coverage estimate", f"{calc['selected_model_coverage']:.2%}")
    if form_research:
        st.warning("Research form model is a separate, unvalidated probability "
                   "distribution. Original and research coverage estimates are "
                   "not directly comparable evidence of improved accuracy. "
                   "No historic forecast has been replaced.")
    st.caption("Both coverage values are statistical estimates under an "
               "independent-matches assumption, NOT observed accuracy. "
               "External AI votes affect candidate order conservatively; "
               "they are NOT added as probabilities.")
    old=calc.get("baseline_diversity") or {}
    new=calc.get("selected_diversity") or {}
    st.caption("Different scores covered (match 1 / match 2): original "+
               str(old.get("match_1_scores","?"))+" / "+str(old.get("match_2_scores","?"))+
               " · selected "+str(new.get("match_1_scores","?"))+" / "+
               str(new.get("match_2_scores","?")))
    if challenger:
        st.warning("Experimental diversity challenger: it may spread five guesses across "
                   "more model-supported scorelines, but normally reduces the "
                   "probability-maximising estimated coverage. This is NOT a proven "
                   "accuracy gain. The original frozen records and automated "
                   "Five Doubles remain unchanged.")
    if form_research:
        st.caption("V21 AI verdicts are recorded as advisory only; they do NOT "
                   "change research score probabilities until verified future "
                   "performance establishes an improvement.")
    packets=report.get("v21_provider_packets") or [None,None]
    with st.expander("🔎 Verified pre-match provider source availability",expanded=True):
        for i,packet in enumerate(packets):
            st.markdown("**Match "+str(i+1)+"**")
            if not packet:
                st.caption("No source-matched timestamped provider packet. "
                           "Missing lineup, shot xG and market values were not guessed.")
                continue
            st.caption("Captured "+str(packet["captured_at_utc"])+
                       " UTC · source SHA-256 "+str(packet["snapshot_sha256"])[:16])
            available=packet.get("available_sources") or {}
            st.caption("Previous-fixture expected goals: "+str(len(
                packet.get("verified_historical_fixture_xg") or []))+
                " qualifying fixture(s) · Provider-reported starting XIs: "+
                str(len(packet.get("confirmed_starting_xi") or []))+
                " · Injury reports: "+str(len(packet.get("reported_injuries") or []))+
                " · Timestamped 1X2 odds: "+("yes" if available.get("timestamped_1x2") else "no"))
    with st.expander("📋 Actual team research: last 10, venue, H2H, goals, injuries", expanded=True):
        for i, row in enumerate((first, second)):
            st.markdown("**Match " + str(i+1) + ": " +
                        row["fixture"]["homeTeam"]["name"] + " vs " +
                        row["fixture"]["awayTeam"]["name"] + "**")
            p = report["historical_profiles"][i]
            if not p.get("available"):
                st.warning("Dated match history is unavailable for this fixture.")
                continue
            st.caption("Provider-matched verified past results: " +
                       str(p["verified_historical_fixtures"]) + " · "
                       "H2H is descriptive, not a fitted player-strength adjustment.")
            for title, key in (("Home last 10", "home_last10"),
                               ("Home at home", "home_at_home_last10"),
                               ("Away last 10", "away_last10"),
                               ("Away away", "away_away_last10")):
                sample = p[key]
                if sample["matches"]:
                    st.write("**" + title + "** · " + str(sample["matches"]) +
                             " played · " + str(sample["scored"]) +
                             " scored/game · " + str(sample["conceded"]) +
                             " conceded/game · 2+ scored " +
                             f"{sample['scored_2plus']:.0%}" +
                             " · 3+ total " + f"{sample['match_over_2_5']:.0%}")
                    st.caption("Most recent scored–conceded results: " +
                               " | ".join(str(x["gf"])+"–"+str(x["ga"]) +
                                          " ("+x["venue"]+")"
                                          for x in sample["last_scores"][:5]))
                else:
                    st.caption(title + ": no verified result available")
            h2h = p.get("h2h_last6") or []
            st.caption("Previous H2H 90-minute scores: " +
                       (", ".join(str(x["home_goals"])+"–"+str(x["away_goals"])
                                  for x in h2h) if h2h else "none in source"))
            model = report["research_models"][i]
            if model.get("available"):
                original = model["original_expected_goals"]
                adjusted = model["research_expected_goals"]
                st.caption("Base goal means " + f"{original[0]:.2f} / {original[1]:.2f}" +
                           " · Experimental form means " +
                           f"{adjusted[0]:.2f} / {adjusted[1]:.2f}" +
                           " · P(total 4+) " +
                           f"{model['four_plus_goal_probability_research']:.1%}")
                for side in ("home", "away"):
                    bands = model["goal_count_probabilities"][side]
                    st.caption(side.title() + " possible goals · 0: " +
                               f"{bands['0']:.0%}" + " · 1: " +
                               f"{bands['1']:.0%}" + " · 2: " +
                               f"{bands['2']:.0%}" + " · 3+: " +
                               f"{bands['3+']:.0%}")
            else:
                st.caption("Goal-form model abstained: " + str(model.get("reason")))
            ctx = report["provider_context"][i]
            if ctx.get("standings"):
                st.caption("Verified provider table: " +
                           "; ".join(str(x.get("team")) + " rank " +
                                     str(x.get("rank")) + ", points " +
                                     str(x.get("points"))
                                     for x in ctx["standings"][:2]))
            for key, title in (("home_season","Home season"),
                               ("away_season","Away season")):
                season = ctx.get(key) or {}
                if season:
                    st.caption(title + " goals, home/away splits and team form are "
                               "available from the provider; see detailed source packet.")
                    st.caption("Provider-reported " + title + " goals: " +
                               str(season.get("goals") or "unavailable")[:450] +
                               " · form " + str(season.get("form") or "unavailable")[:50])
            if ctx.get("injuries"):
                st.caption("Provider-reported injuries: " +
                           "; ".join(str(x.get("team")) + " — " + str(x.get("player"))
                                     for x in ctx["injuries"][:8]))
            if ctx.get("confirmed_lineups"):
                st.caption("Provider lineup/formation records: " +
                           "; ".join(str(x.get("team")) + " " +
                                     str(x.get("formation"))
                                     for x in ctx["confirmed_lineups"][:3]))
            st.caption("No injury report or lineups supplied does NOT mean "
                       "every player is fit. Player effects are context only.")
    with st.expander("🔎 All top-ten model scorelines for each match",expanded=False):
        for i,row in enumerate((first,second),1):
            st.markdown("**Match "+str(i)+" · existing statistical model**")
            for home,away,p in _scores(row)[:10]:
                st.caption(str(home)+"–"+str(away)+" · "+f"{float(p):.2%}")
    for num,pick in enumerate(calc["doubles"],1):
        with st.container(border=True):
            st.markdown("**Double "+str(num)+"**")
            left,right=st.columns(2)
            left.metric("Match 1 score",pick["Score 1"])
            right.metric("Match 2 score",pick["Score 2"])
            st.caption("Original model joint estimate: "+
                       f"{pick['Model joint']:.2%}"+
                       " · AI endorsements: "+
                       str(pick["AI endorsements 1"])+" / "+
                       str(pick["AI endorsements 2"]))
    st.download_button("⬇️ Download five AI-assisted doubles (.csv)",
                       _csv_bytes(calc),"ziczac_ai_five_doubles.csv","text/csv",
                       key="ziczac_ai_five_download",use_container_width=True)
    with st.expander("🔬 Source-by-source research and AI opinions",expanded=True):
        for i,row in enumerate((first,second)):
            st.markdown("**Match "+str(i+1)+": "+row["fixture"]["homeTeam"]["name"]+
                        " vs "+row["fixture"]["awayTeam"]["name"]+"**")
            st.caption("Model: "+str(report["statistical_models"][i])+
                       " · Source: "+report["data_source"][i]+
                       " · Dated past match records available: "+
                       str(report["past_matches_available"][i]))
            if row.get("archive_frozen"):
                st.caption("Frozen forecast captured "+str(row.get("frozen_at_utc"))+
                           " UTC · Historical count in saved evidence: "+
                           str(row.get("history_matches_reported") or "unknown")+
                           " · Raw historical match list unavailable in this session.")
            votes=report["ai_research"][i]["votes"]
            if not votes:
                st.info("No validated external AI opinion for this match. "
                        "The statistical model is still used.")
            for provider,vote in sorted(votes.items()):
                score=vote.get("score") or []
                if len(score)!=2:
                    continue
                st.write("**"+provider.title()+"** · "+str(score[0])+"–"+str(score[1])+
                         " · "+report["ai_research"][i]["sources"].get(provider,""))
                facts=vote.get("factors") or []
                if isinstance(facts,list) and facts:
                    st.caption("Evidence notes: "+"; ".join(str(f)[:160] for f in facts[:2]))
            for provider,error in sorted(report["api_errors"][i].items()):
                st.caption(provider.title()+": "+
                           str(error.get("reason") or "provider_error")+
                           (" · HTTP "+str(error.get("http_status"))
                            if error.get("http_status") is not None else ""))
    st.info("No AI model automatically browses every football website. "
            "The result uses available dated fixture/history feeds and "
            "configured provider opinions. Missing team news, market odds, "
            "injury or lineup data is not invented. These five doubles are "
            "experimental research, not guaranteed wins or independently "
            "verified historical performance.")


def render(st,rows,histories=None):
    st.subheader("🤖 AI Five-Double Studio")
    st.write("Choose **two matches**. ZicZac uses its existing statistical "
             "predictions and any validated saved AI research, with optional "
             "fresh calls to connected ChatGPT/OpenAI, Gemini and other providers.")
    st.caption("The original Five Doubles page and frozen prediction history "
               "remain unchanged.")
    minute=int(datetime.now(timezone.utc).timestamp()//60)
    from ai_five_live_finder import research_day
    from on_demand_ai import utc
    import os

    clock=datetime.now(timezone.utc)
    local_today=clock.astimezone(SYDNEY).date()
    st.caption("Sydney dates: Today "+local_today.strftime("%d %b %Y")+
               " · Tomorrow "+(local_today+timedelta(days=1)).strftime("%d %b %Y"))
    date_scope=st.selectbox(
        "Show matches for",
        ["Today & Tomorrow", "Today only", "Tomorrow only", "Next 7 days", "All upcoming"],
        key="zz_ai_five_day_scope")
    search_day=st.selectbox(
        "Search fresh matches on",
        ["Today", "Tomorrow"],
        key="zz_ai_five_target_day",
        help="Choose a Sydney date and request real fixtures from API-Football. "
             "This does not invent scores or change archived predictions.")
    target_day=local_today+(timedelta(days=1) if search_day=="Tomorrow" else timedelta())
    with st.expander("🔎 Find today's or tomorrow's missing matches", expanded=True):
        st.caption("If the saved forecast list starts on 19 November, the "
                   "collector may not have modelled two eligible earlier fixtures. "
                   "Search this date directly, then build score predictions "
                   "only from sufficient verified historical results. "
                   "Up to four matches are researched per click to limit API use.")
        request_today=st.button(
            "🔎 Search & model "+search_day+"'s matches",
            key="zz_ai_five_search_near", use_container_width=True)
    fetched=st.session_state.setdefault("zz_ai_five_targeted_modelled",{})
    day_key=target_day.isoformat()
    if request_today:
        try:
            api_key=str(st.secrets.get("API_FOOTBALL_KEY") or
                        os.environ.get("API_FOOTBALL_KEY") or "").strip()
        except (AttributeError, OSError, KeyError, RuntimeError):
            api_key=str(os.environ.get("API_FOOTBALL_KEY") or "").strip()
        previous=fetched.get(day_key) or {"rows":[],"checked_ids":[]}
        with st.spinner("Searching verified fixtures and modelling up to four "
                        "upcoming matches on "+day_key+"..."):
            result=research_day(api_key,target_day,
                                known_ids=previous["checked_ids"],now=clock)
        existing={identity(row):row for row in previous["rows"]}
        for new_row in result["modeled"]:
            existing.setdefault(identity(new_row),new_row)
        fetched[day_key]={
            "rows":list(existing.values()),
            "checked_ids":list(dict.fromkeys(previous["checked_ids"]+
                                              result["checked_ids"])),
            "last":{k:v for k,v in result.items()
                    if k not in ("modeled","checked_ids")}
        }
    for day,entry in sorted(fetched.items()):
        if day>=local_today.isoformat():
            status=entry.get("last") or {}
            if status:
                if status.get("quota_limited"):
                    st.error("⚠️ API-Football quota/rate limit stopped the "+
                             day+" search. **0 returned fixtures is NOT a "
                             "confirmed empty football schedule.** Do not "
                             "keep retrying: check your API-Sports account "
                             "usage or wait for the provider allowance to reset. "
                             "Previously saved forecasts remain available.")
                else:
                    st.caption("Direct search "+day+": "+str(status.get("discovered",0))+
                               " eligible upcoming fixtures found, "+
                               str(len(entry.get("checked_ids") or []))+
                               " checked, "+str(len(entry.get("rows") or []))+
                               " fully modelled; "+
                               str(status.get("remaining",0))+
                               " more unprocessed. No missing results were fabricated.")
                for e in (status.get("errors") or [])[:2]:
                    st.caption("Provider/model limitation: "+str(e)[:160])
    targeted=[row for day,entry in fetched.items()
              if day>=local_today.isoformat() for row in entry.get("rows",[])]
    found_all=options(rows+targeted, archived=_minute_archived_candidates(minute))
    until=local_today+timedelta(days=7)
    def in_scope(row):
        day=utc(row["kickoff"]).astimezone(SYDNEY).date()
        if date_scope=="Today only":
            return day==local_today
        if date_scope=="Tomorrow only":
            return day==local_today+timedelta(days=1)
        if date_scope=="Today & Tomorrow":
            return local_today<=day<=local_today+timedelta(days=1)
        if date_scope=="Next 7 days":
            return local_today<=day<until
        return True
    found=[row for row in found_all if in_scope(row)]
    st.caption("Eligible modelled matches in selected range: "+str(len(found))+
               " · All upcoming eligible matches: "+str(len(found_all))+
               ". Same-Sydney-day pairing and five distinct scores are mandatory.")
    if len(found)<2:
        next_label=(label(found_all[0]) if found_all else "none yet")
        st.info("Fewer than two eligible matches are available in this "
                "Sydney-date range. This does not mean football games are "
                "absent; they may lack a verified pre-match score model. "
                "Use Search & model for Today or Tomorrow above. "
                "You can also select All upcoming for existing later dates.")
        if found_all and not found:
            st.caption("First available eligible saved/modelled fixture: "+next_label)
        return
    ids={label(r):r for r in found}
    first_label=st.selectbox("1. Select Match 1",list(ids),key="ziczac_ai_five_first")
    first=ids[first_label]
    second_options={k:v for k,v in ids.items() if v is not first and
                    pair_check(first,v) is None}
    if not second_options:
        st.info("No eligible second match on the same Sydney date with four "
                "different teams. Choose a different first match or a wider "
                "fixture window.")
        return
    second_label=st.selectbox("2. Select Match 2",list(second_options),
                              key="ziczac_ai_five_second")
    second=second_options[second_label]
    st.caption("Selected: "+first["fixture"]["homeTeam"]["name"]+" / "+
               first["fixture"]["awayTeam"]["name"]+" and "+
               second["fixture"]["homeTeam"]["name"]+" / "+
               second["fixture"]["awayTeam"]["name"])
    base=merged_five(first,second)
    if base["status"]!="ready":
        st.warning(base.get("reason") or "The statistical model cannot form five valid doubles.")
        return
    research_ready = all(goal_form_challenger(
        row, _history(row, histories), clock).get("available")
        for row in (first, second))
    mode_label=st.selectbox(
        "Choose five-double scoring method",
        ["Research goal-form challenger (experimental)", "Original top-probability baseline",
         "Balanced challenger (experimental)"],
        index=0 if research_ready else 1,key="ziczac_ai_five_selection_mode",
        help="Research goal-form recomputes a shadow score grid from verified "
             "last-ten home/away scoring and opposition concessions, only if "
             "both fixtures have complete original score grids and enough "
             "dated results. Original and balanced methods remain available.")
    if not research_ready:
        st.caption("Experimental goal-form mode needs more verified past home/away "
                   "matches for both teams plus complete score grids. "
                   "Original five doubles remain available without inventing research.")
    selection_mode = ("goal_form_research" if mode_label.startswith("Research") else
                      "balanced_challenger" if mode_label.startswith("Balanced")
                      else "probability")
    st.caption("Research mode tests real goal strength and wider score scenarios; "
               "it never forces a win or a 3-goal score. No result is shown "
               "when essential evidence is missing. Existing Five Doubles "
               "and frozen historical forecasts remain unchanged.")
    configs=server_settings(st)
    private=private_gate_ready(st)
    st.markdown("#### Connected research sources")
    st.caption("Statistical model: "+str((first.get("result") or {}).get("model_version"))+
               " and "+str((second.get("result") or {}).get("model_version")))
    st.caption("External AI services configured on this Streamlit server: "+
               (", ".join(p.title() for p in configs) if configs else "none")+
               ". Saved GitHub research is also checked for both fixtures.")
    selected=st.multiselect("Which connected AI services to consult?",
        [p for p in PROVIDERS if p in configs],
        default=[p for p in PROVIDERS if p in configs],
        key="ziczac_ai_five_providers")
    consent=st.checkbox(
        "Use fresh external AI API calls for BOTH matches (charges may apply)",
        value=False,key="ziczac_ai_five_cost_consent",
        help="Click only if you consent to one request per selected model per "
             "match. Models can have billing quotas.")
    if consent and not private:
        st.warning("Fresh paid API calls require private ZicZac login. "
                   "Set ZICZAC_ACCESS_PASSWORD in Streamlit Secrets and sign in.")
    if consent and not configs:
        st.warning("GitHub Actions secrets cannot be read by Streamlit. "
                   "Add API keys and model IDs in Streamlit Cloud → Settings → Secrets.")
    st.caption("Without fresh API access, Run still compares the baseline and "
               "any valid previously frozen AI suggestions. It never pretends "
               "that unavailable providers were consulted.")
    clicked=st.button("🚀 Run AI Five-Double Prediction",type="primary",
                      use_container_width=True,key="ziczac_ai_five_run")
    cache=st.session_state.setdefault("ziczac_ai_five_session",{})
    pair=(identity(first),identity(second),
          str((first.get("result") or {}).get("model_version")),
          str((second.get("result") or {}).get("model_version")),
          selection_mode)
    if clicked:
        if pair not in cache:
            permitted=bool(consent and private and selected and configs)
            with st.spinner("Comparing statistical scores and available AI research..."):
                report=analyze_pair(first,second,histories or {},
                    configs={k:configs[k] for k in selected} if permitted else {},
                    use_live=permitted,qwen_url=server_qwen_url(st),
                    selection_mode=selection_mode)
            cache[pair]=report
            # Preserve this five-option user selection and all validated AI
            # opinions *as they were returned before kickoff*. A later rerun
            # cannot alter the original score combinations.
            if report.get("status")=="ready":
                import os
                from ziczac_interactive_capture import (
                    pair as _audit_pair, single as _audit_single,
                    store as _audit_store, store_ui as _audit_ui)
                try:
                    _choices=[{"score1":x["Score 1"],"score2":x["Score 2"]}
                              for x in report["model_and_ai_result"]["doubles"]]
                    _audit_ui(st,_audit_pair(
                        first["fixture"],second["fixture"],_choices,
                        method=(selection_mode if selection_mode != "probability" else "")))
                except (ValueError,TypeError,KeyError):
                    st.warning("This five-double result was not eligible for "
                               "permanent pre-match tracking.")
                try:
                    _db=st.secrets.get("DATABASE_URL","") or os.environ.get("DATABASE_URL","")
                except (OSError,KeyError,RuntimeError,AttributeError):
                    _db=os.environ.get("DATABASE_URL","")
                _ai_votes_saved=0
                if private:
                    for _leg,_research in zip((first,second),report["ai_research"]):
                        for _provider,_vote in (_research.get("votes") or {}).items():
                            try:
                                _ai_votes_saved+=int(_audit_store(_audit_single(
                                    _leg["fixture"],_vote["score"],"ten_ai",
                                    ai_provider=_provider),_db).get("durable",False))
                            except (ValueError,TypeError,KeyError):
                                continue
                if _ai_votes_saved:
                    st.caption(str(_ai_votes_saved)+" individual model votes "
                               "preserved alongside this five-double selection.")
        else:
            st.info("Showing the existing session result for this match pair. "
                    "No duplicate paid API calls were made.")
    report=cache.get(pair)
    if report:
        if report.get("status")=="ready":
            _display(st,report,first,second)
        else:
            st.warning(report.get("reason") or "This pair could not be analysed.")
    else:
        st.info("Select two matches and press **Run AI Five-Double Prediction**.")
    st.caption("Set up website API access in [Streamlit Secrets instructions]"
               "(https://github.com/ZicZac-hub/scorepredict-ai/blob/main/"
               "ON_DEMAND_AI_SETUP.md). Never share your API keys in chat.")
