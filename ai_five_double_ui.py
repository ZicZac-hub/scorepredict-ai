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


def analyze_pair(first,second,histories,*,configs=None,use_live=False,
                 request=None,now=None,qwen_url=None,archiver=frozen_for_fixture):
    """Return one session-only report for an explicitly chosen pair."""
    now=utc(now or datetime.now(timezone.utc))
    issue=pair_check(first,second,now=now)
    if issue:
        return {"status":"invalid_pair","reason":issue}
    archives=[archiver(row["fixture"]) for row in (first,second)]
    fresh=[None,None]
    errors=[{},{}]
    if use_live and configs:
        for i,row in enumerate((first,second)):
            already=(archives[i] or {}).get("votes") or {}
            # Do not pay again for an already-frozen provider opinion.
            remaining={p:cfg for p,cfg in configs.items() if p not in already}
            if not remaining:
                continue
            try:
                result=run_selected(
                    row["fixture"],_history(row,histories),
                    row.get("result") or {},{},
                    remaining,now=now,request=request,qwen_url=qwen_url)
                fresh[i]=result
                errors[i]=result.get("errors") or {}
            except Exception:
                errors[i]={"research":{"reason":"provider_request_failed"}}
    evidence=[_as_research(archives[i],fresh[i]) for i in (0,1)]
    combined=merged_five(first,second,evidence[0],evidence[1],now=now)
    if combined["status"]!="ready":
        return combined
    return {
        "status":"ready","generated_at_utc":now.isoformat(),
        "fixtures":[identity(first),identity(second)],
        "teams":[teams(first),teams(second)],
        "statistical_models":[(first.get("result") or {}).get("model_version"),
                              (second.get("result") or {}).get("model_version")],
        "data_source":["archived statistical forecast" if r.get("archive_frozen")
                       else "current statistical forecast" for r in (first,second)],
        "past_matches_available":[len(_history(r,histories)) for r in (first,second)],
        "archived_history_counts":[r.get("history_matches_reported") for r in (first,second)],
        "ai_research":evidence,
        "api_errors":errors,
        "model_and_ai_result":combined,
        "new_ai_requested":bool(use_live and configs),
        "session_only":True,
        "original_forecasts_overwritten":False,
        "independent_prospective_accuracy_record":False,
        "provider_disclosure":"Only provider APIs and dated data connected to ZicZac; no unlimited web browsing"
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
    st.markdown("### ✅ Final five exact-score doubles")
    calc=report["model_and_ai_result"]
    st.caption("Match 1: "+label(first))
    st.caption("Match 2: "+label(second))
    st.caption("Research compiled "+str(report["generated_at_utc"])+
               " UTC · Original forecasts unchanged · session-only comparison.")
    a,b=st.columns(2)
    a.metric("Baseline model coverage",f"{calc['baseline_coverage']:.2%}")
    b.metric("Final model-only coverage",f"{calc['selected_model_coverage']:.2%}")
    st.caption("Both coverage values are statistical estimates under an "
               "independent-matches assumption, NOT observed accuracy. "
               "External AI votes affect candidate order conservatively; "
               "they are NOT added as probabilities.")
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
          str((second.get("result") or {}).get("model_version")))
    if clicked:
        if pair not in cache:
            permitted=bool(consent and private and selected and configs)
            with st.spinner("Comparing statistical scores and available AI research..."):
                report=analyze_pair(first,second,histories or {},
                    configs={k:configs[k] for k in selected} if permitted else {},
                    use_live=permitted,qwen_url=server_qwen_url(st))
            cache[pair]=report
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
