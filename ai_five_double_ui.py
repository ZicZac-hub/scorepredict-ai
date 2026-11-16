"""Standalone manual AI Five-Double Studio for ZicZac Streamlit.

Use the SAME original model score tables as Five Doubles, optionally adding
previously frozen AI opinions plus owner-authorised on-demand AI providers.
No scraping, no fabricated data, no overwrite of original match forecasts.
"""
from datetime import datetime, timedelta, timezone
import csv
import io
from functools import lru_cache
from pathlib import Path

from ai_five_double_engine import eligible, identity, merged_five, pair_check, teams, SYDNEY
from on_demand_ai import (frozen_for_fixture, private_gate_ready, run_selected,
                          server_qwen_url, server_settings, utc)
from llm_research_panel import PROVIDERS
from smart_double_ai import _scores
from goal_research_v20 import goal_profile, goal_form_challenger
from v21_provider_evidence import valid_snapshot
from v22_shadow import shadow as v22_shadow
from v23_market_form import shadow as v23_shadow
from source_capture_clock import capture_clock


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
                          # Do not discard the verified full probability
                          # grid: V20–V23 cannot produce genuine score
                          # probabilities from a top-ten shortlist alone.
                          "matrix":saved.get("score_matrix"),
                          "model_version":saved["model_version"],
                          "outcome_probabilities":saved.get("outcome_probabilities") or {},
                          "home_xg":(saved.get("expected_goals") or {}).get("home"),
                          "away_xg":(saved.get("expected_goals") or {}).get("away")},
                "league":saved.get("competition") or "",
                "past":[],"archive_frozen":True,
                "frozen_at_utc":captured.isoformat(),
                "history_matches_reported":saved.get("history_matches"),
                "league_history_matches":saved.get("league_history_matches")
            }
            # Only advertise source provenance when it was genuinely saved.
            source_keys=("provider","fixture_id","forecast_id","input_digest",
                         "home_team_id","away_team_id","kickoff_utc")
            if all(saved.get(key) is not None for key in source_keys):
                row["source_forecast"]={key:saved[key] for key in source_keys}
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
def _minute_repository_originals(minute):
    """Read source identities once per minute, not for each Streamlit rerun."""
    from v10_prospective import read_latest
    return list(read_latest().values())


def repository_provider_packet(fixture, *, now=None, originals=None,
                               packet_root=None, priority="market"):
    """Use the same genuine GitHub-scheduled provider evidence in the Studio.

    The source packet must match an immutable original forecast's provider,
    fixture/team IDs and kickoff, plus its exact forecast ID/input digest.
    This never adds paid API requests or invents data. For private databases
    the existing owner-authorized database path remains an alternative.
    """
    from v21_capture_sources import best_packet
    clock = utc(now or datetime.now(timezone.utc))
    kickoff = utc(fixture.get("utcDate"))
    if kickoff <= clock:
        return None
    provider = str(fixture.get("provider") or "football-data.org")
    fixture_id = str(fixture.get("id"))
    home = str((fixture.get("homeTeam") or {}).get("id"))
    away = str((fixture.get("awayTeam") or {}).get("id"))
    if any(value in ("None", "", "null") for value in (fixture_id, home, away)):
        return None
    if originals is None:
        originals = _minute_repository_originals(int(clock.timestamp() // 60))
    if isinstance(originals, dict):
        originals = originals.values()
    packet_root = packet_root if packet_root is not None else Path("data/v21_sources")
    found = []
    for source in originals or []:
        try:
            if source.get("model_key") not in ("v8", "v8_near"):
                continue
            if (str(source["provider"]) != provider or
                str(source["fixture_id"]) != fixture_id or
                str(source["home_team_id"]) != home or
                str(source["away_team_id"]) != away or
                utc(source["kickoff_utc"]) != kickoff or
                utc(source["captured_at_utc"]) >= kickoff or
                utc(source["captured_at_utc"]) > clock):
                continue
            record = best_packet(packet_root,source,now=clock,priority=priority)
            packet = (record or {}).get("provider_snapshot")
            if not valid_snapshot(packet,fixture,now=clock):
                continue
            if priority == "market":
                markets = packet.get("multi_market") or {}
                if "1x2" not in markets or not ({"over_2_5","over_3_5"} & set(markets)):
                    continue
                clocks = [utc(markets[k]["last_source_update_utc"]) for k in
                          ("1x2","over_2_5","over_3_5") if k in markets]
                if not clocks or any(t > clock or t > kickoff or
                                     clock-t > timedelta(hours=12) for t in clocks):
                    continue
                quality = (max(clocks),utc(packet["captured_at_utc"]))
            else:
                quality = (utc(packet["captured_at_utc"]),utc(source["captured_at_utc"]))
            found.append((quality,packet))
        except (KeyError,TypeError,ValueError,OSError):
            continue
    return max(found,key=lambda pair:pair[0])[1] if found else None


@lru_cache(maxsize=4)
def _minute_archived_candidates(minute):
    """One read per minute rather than rescanning hundreds of JSON files
    for every Streamlit button or selectbox rerun."""
    return archive_rows()


def _has_complete_score_grid(row):
    """Separate ten score suggestions from a real normalized joint score grid."""
    try:
        import numpy as np
        raw=(row.get("result") or {}).get("matrix")
        if raw is None:
            return False
        matrix=np.asarray(raw,dtype=float)
        return (matrix.ndim==2 and min(matrix.shape)>=5 and
                bool(np.isfinite(matrix).all()) and bool((matrix>=0).all()) and
                abs(float(matrix.sum())-1.0)<1e-5)
    except (TypeError,ValueError,AttributeError):
        return False


def options(rows, *, now=None, archived=None):
    """Prefer current forecasts when equally complete; recover archived grids.

    A current ten-score-only record cannot support researched V23 market
    adjustments. Prefer a fully saved original grid over silently discarding
    a genuine, pre-match, source-matched archival model.
    """
    chosen={}
    for row in rows or []:
        if isinstance(row,dict) and eligible(row,now=now):
            chosen.setdefault(identity(row),row)
    for row in archived if archived is not None else archive_rows(now=now):
        if eligible(row,now=now):
            key=identity(row)
            current=chosen.get(key)
            if current is None or (
                    not _has_complete_score_grid(current) and
                    _has_complete_score_grid(row)):
                chosen[key]=row
    return sorted(chosen.values(),key=lambda row:(utc(
        row.get("kickoff") or row["fixture"]["utcDate"]),identity(row)))


def label(row):
    fixture=row["fixture"]
    kickoff=utc(row.get("kickoff") or fixture["utcDate"]).astimezone(SYDNEY)
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
    return {"status":"completed","votes":votes,"sources":source,
            "source_snapshot_digest":(fresh or {}).get("source_snapshot_digest")
            if fresh and fresh.get("status")=="completed" else None}


def analyze_pair(first, second, histories, *, configs=None, use_live=False,
                 request=None, now=None, qwen_url=None, archiver=frozen_for_fixture,
                 selection_mode="probability", provider_packets=None, clock=None):
    """Evidence-backed, session-only research for one pre-match pair."""
    now, observed = capture_clock(now, clock)
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
    v22_models = []
    if selection_mode == "v22_xg_market":
        v22_models = [v22_shadow(row,provider_packets[i],as_of=now)
                      for i,row in enumerate(pair_rows)]
        blocked = [f"Match {i+1}: {x.get('reason','unavailable')}"
                   for i,x in enumerate(v22_models) if not x.get("ready")]
        if blocked:
            return {"status":"insufficient_v22_evidence",
                    "reason":"V22 cannot create verified xG + market doubles: " +
                             " | ".join(blocked) +
                             ". Use the original baseline when any source is missing; "
                             "V22 never invents lineups or historical xG."}
        effective_rows = []
        for row,model in zip(pair_rows,v22_models):
            adjusted = dict(row.get("result") or {})
            adjusted.update({"scores":model["scores"],"matrix":model["matrix"],
                             "home_xg":model["home_xg"],"away_xg":model["away_xg"],
                             "model_version":model["prediction_method"]})
            effective_rows.append({**row,"result":adjusted})
    v23_models = []
    if selection_mode == "v23_market_form":
        v23_models=[v23_shadow(row,provider_packets[i],as_of=now,
                              history=dated_history[i])
                    for i,row in enumerate(pair_rows)]
        blocked=[f"Match {i+1}: {x.get('reason','unavailable')}"
                 for i,x in enumerate(v23_models) if not x.get("ready")]
        if blocked:
            return {"status":"insufficient_v23_evidence",
                    "reason":"V23 cannot produce researched doubles: "+
                             " | ".join(blocked)+
                             ". Both legs need source-verified 1X2 and goal-total odds. "
                             "True shot-based xG is optional, not fabricated."}
        effective_rows=[]
        for row,model in zip(pair_rows,v23_models):
            adjusted=dict(row.get("result") or {})
            adjusted.update({"scores":model["scores"],"matrix":model["matrix"],
                             "home_xg":model["home_xg"],"away_xg":model["away_xg"],
                             "model_version":model["prediction_method"]})
            effective_rows.append({**row,"result":adjusted})
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
            request_time = observed()
            if pair_check(first, second, now=request_time):
                return {"status":"analysis_completed_too_late",
                        "reason":"Research exceeded the eligible pre-match window; "
                                 "no new five-double prediction was produced."}
            try:
                result = run_selected(
                    row["fixture"], dated_history[i], row.get("result") or {}, {},
                    remaining, now=request_time, request=request, qwen_url=qwen_url,
                    team_context={**(row.get("research_context") or {}),
                                  "v21_provider_snapshot":provider_packets[i]})
                fresh[i] = result
                errors[i] = result.get("errors") or {}
            except Exception:
                errors[i] = {"research": {"reason": "provider_request_failed"}}
    completed = observed()
    if pair_check(first, second, now=completed):
        return {"status":"analysis_completed_too_late",
                "reason":"Research exceeded the eligible pre-match window; "
                         "no new five-double prediction was produced."}
    evidence = [_as_research(archives[i], fresh[i]) for i in (0, 1)]
    original = merged_five(first, second, now=completed,
                           selection_mode="probability")
    # Until a source-matched AI panel beats statistical-only picks on
    # independent future games, V21 never promotes AI opinions to numerical
    # weighting. The validated votes remain visible for review.
    combined = merged_five(effective_rows[0], effective_rows[1],
                           evidence[0], evidence[1], now=completed,
                           selection_mode=("probability" if
                                           selection_mode in ("goal_form_research",
                                                              "v22_xg_market",
                                                              "v23_market_form")
                                           else selection_mode))
    if combined["status"] != "ready":
        return combined
    if selection_mode == "v23_market_form":
        combined["selection_mode"]="v23_market_form"
        combined["baseline_coverage"]=original.get("selected_model_coverage")
        combined["baseline_diversity"]=original.get("selected_diversity")
        combined["estimated_coverage_is_not_observed_accuracy"]=True
    if selection_mode == "v22_xg_market":
        combined["selection_mode"]="v22_xg_market"
        combined["v22_shadow_coverage"]=combined["selected_model_coverage"]
        combined["baseline_coverage"]=original.get("selected_model_coverage")
        combined["baseline_diversity"]=original.get("selected_diversity")
        combined["estimated_coverage_is_not_observed_accuracy"]=True
    if selection_mode == "goal_form_research":
        # The two values use different distributions; never imply the
        # challenger coverage is an observed or directly comparable win rate.
        combined["selection_mode"] = "goal_form_research"
        combined["research_model_coverage"] = combined["selected_model_coverage"]
        combined["baseline_coverage"] = original.get("selected_model_coverage")
        combined["baseline_diversity"] = original.get("selected_diversity")
        combined["research_coverage_not_comparable_without_validation"] = True
    return {
        "status": "ready", "generated_at_utc": completed.isoformat(),
        "analysis_started_at_utc": now.isoformat(),
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
        "v22_shadow_models":v22_models,"v23_shadow_models":v23_models,
        "v21_provider_packets": provider_packets,
        "research_ai_votes_weighted": False,
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
    v22_mode = calc.get("selection_mode") == "v22_xg_market"
    v23_mode = calc.get("selection_mode") == "v23_market_form"
    method = ("V23 market + goal-form challenger" if v23_mode else
              "V22 verified xG + market challenger" if v22_mode else
              "research goal-form challenger" if form_research else
              "balanced challenger" if challenger else "original baseline")
    st.markdown("### ✅ Final five exact-score doubles · " + method)
    st.caption("Match 1: "+label(first))
    st.caption("Match 2: "+label(second))
    st.caption("Research compiled "+str(report["generated_at_utc"])+
               " UTC · Original forecasts unchanged · session-only comparison.")
    a,b=st.columns(2)
    a.metric("Original coverage estimate", f"{calc['baseline_coverage']:.2%}")
    b.metric("Selected model coverage estimate", f"{calc['selected_model_coverage']:.2%}")
    if v23_mode:
        st.warning("V23 uses time-verified market-winner and goal-total prices "
                   "with the original statistical goal model and any real, "
                   "available home/away form. Genuine shot xG is optional; "
                   "never inferred. No verified increase in accuracy yet.")
        for i,model in enumerate(report.get("v23_shadow_models") or []):
            st.caption("Match "+str(i+1)+" · Goal-total over 2.5: "+
                       f"{model['prior_over_2_5_probability']:.1%} → "+
                       f"{model['adjusted_over_2_5_probability']:.1%}"+
                       " · Sources: "+", ".join(model["applied_signals"]))
    if v22_mode:
        st.warning("V22 uses independent historical API-provided expected goals "
                   "and time-checked market odds. Results are an experimental "
                   "shadow prediction, NOT verified higher accuracy. "
                   "Confirmed lineups are shown without invented player effect.")
    if form_research:
        st.warning("Research form model is a separate, unvalidated probability "
                   "distribution. Original and research coverage estimates are "
                   "not directly comparable evidence of improved accuracy. "
                   "No historic forecast has been replaced.")
    st.caption("Both coverage values are statistical estimates under an "
               "independent-matches assumption, NOT observed accuracy. "
               "External AI votes are shown as annotations and do not "
               "change candidate order or score probabilities.")
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
    if v22_mode:
        st.caption("V22 ignores unproven AI score votes numerically. "
                   "It requires real xG and at least two market types for BOTH "
                   "fixtures; no source means no new prediction.")
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
    try:
        v25_selected=str(st.query_params.get("v25",""))=="1"
    except (AttributeError,TypeError):
        v25_selected=False
    engine=st.radio("Choose your five-double prediction engine",
                    ["🧮 V25 Professional Analyst · all upcoming matches",
                     "🤖 Original AI Five-Double Studio"],
                    index=0 if v25_selected else 1,
                    key="zz_ai_five_engine_choice",
                    help="V25 can model any two eligible future matches with "
                         "complete source score grids on the same Sydney day; "
                         "manual, automatic and all-pairs browsing are included.")
    if engine.startswith("🧮 V25"):
        from v25_upcoming_ui import render as render_v25_upcoming
        render_v25_upcoming(st,rows,histories)
        return
    st.write("Choose **two matches**. ZicZac uses its existing statistical "
             "predictions and any validated saved AI research, with optional "
             "fresh calls to connected ChatGPT/OpenAI, Gemini and other providers.")
    st.caption("The original Five Doubles page and frozen prediction history "
               "remain unchanged.")
    from five_double_accuracy_ui import render as render_diagnosis
    render_diagnosis(st)
    with st.expander("📊 V21 verified research tournament", expanded=False):
        from persist_prematch import load_json as _read_report
        status=_read_report(Path("data/v21_prospective")/"performance.json",{})
        if status.get("methods"):
            st.caption("Real, independently verified five-double pairs. "
                       "Pending sets do NOT count as losses. "
                       "A model needs sufficient future results before promotion.")
            for name,stats in sorted(status["methods"].items()):
                rate=stats.get("observed_hit_rate")
                st.caption(name.replace("_"," ") + ": " +
                           str(stats.get("wins",0)) + "/" +
                           str(stats.get("verified",0)) + " verified wins · " +
                           (f"{rate:.1%}" if rate is not None else "unmeasured") +
                           " · "+str(stats.get("pending",0))+" pending")
        else:
            st.caption("No V21 genuinely frozen five-double match pairs have "
                       "yet been settled. This is NOT evidence of a win or loss.")
        source_status=_read_report(Path("data/v21_sources")/"status.json",{})
        if source_status:
            st.caption("Timestamped external provider packets on latest run: "+
                       str(len(source_status.get("new_packets") or []))+
                       ". Missing data are not converted into scores.")
        st.caption("AI forecasts and new goal models remain research-only "
                   "until independently proven to outperform the original.")
    with st.expander("🧪 V22 source coverage and verified model comparisons", expanded=False):
        from persist_prematch import load_json as _load_v22
        board=_load_v22(Path("data/v22_prospective")/"quality_report.json",{})
        if not board:
            st.caption("V22 has not published a verified performance dashboard yet. "
                       "No retrospective result is being displayed as a live win.")
        else:
            files=board.get("source_completeness") or {}
            st.caption("Source packets: "+str(files.get("packets_checked",0))+
                       " · Genuine past xG: "+str(files.get("with_historical_xg",0))+
                       " · at least 6 per team: "+str(files.get("with_6_per_team",0))+
                       " · 2+ markets: "+str(files.get("with_2_market_types",0))+
                       " · confirmed both starting XIs: "+
                       str(files.get("with_both_starting_XIs",0)))
            for name,stat in (board.get("frozen_v22_vs_v8") or {}).items():
                n=int(stat.get("verified") or 0)
                hits=int(stat.get("hits") or 0)
                st.caption(name+" · "+(f"{hits}/{n} = {hits/n:.1%}" if n else
                                       "no verified pairs")+
                           " · pending "+str(stat.get("pending",0)))
            wide=board.get("v24_broad_market_collection") or {}
            if wide:
                st.caption("V24 worldwide date-market discovery: "+
                           ("enabled" if wide.get("enabled") else "not enabled")+
                           " · API requests this run: "+str(wide.get("queries_last_run",0))+
                           " · requests tracked today: "+str(wide.get("requests_used_today",0))+
                           " / "+str(wide.get("max_requests_per_day") or "not reported")+
                           " · new fixture-market snapshots: "+
                           str(wide.get("new_fixture_markets_last_run",0))+
                           " · matched original V8 packets: "+
                           str(wide.get("new_v8_source_bindings_last_run",0)))
                st.caption("Not every league/provider has bookmaker odds. "
                           "Unmatched market records remain in the source catalog "
                           "until a genuine V8 fixture identity is saved.")
            st.caption("V23 early market collector: "+
                       ("enabled" if board.get("v23_early_market_capture_enabled")
                        else "not confirmed enabled")+
                       " · new source packets on last run: "+
                       str(board.get("v23_early_market_new_packets",0))+
                       " · missing source packets do not count as losses")
            market_method=board.get("v23_market_form_vs_v8") or {}
            if market_method:
                st.markdown("**V23 market + form vs original (same frozen pairs)**")
                for name,stats in sorted(market_method.items()):
                    n=int(stats.get("verified") or 0)
                    wins=int(stats.get("hits") or 0)
                    st.caption(name+": "+(f"{wins}/{n} ({wins/n:.1%})" if n else
                        "no qualifying verified pair yet")+
                        " · pending "+str(stats.get("pending",0)))
            else:
                st.caption("V23 market-form verification is not yet available.")
            ai=board.get("matched_ai_provider_results") or {}
            st.caption("Verified matched AI panels: "+(
                ", ".join(name+"="+str(v.get("source_matched_verified_fixtures",0))
                          for name,v in ai.items()) if ai else "none yet"))
            st.caption("Model promotion disabled. V22 accuracy is unknown until "
                       "qualified pre-match forecasts have verified finals.")
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
        day=utc(row.get("kickoff") or row["fixture"]["utcDate"]).astimezone(SYDNEY).date()
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
    auto_v23=st.checkbox(
        "🧠 V23 Auto Select · time-checked markets and actual team form",
        value=False,key="ziczac_v23_auto_pair",
        help="Compare all same-day pairs with saved bookmaker outcome and "
             "total-goal quotes. No historical shot-xG requirement. "
             "Owner login and saved private evidence are needed.")
    auto_v22=st.checkbox("🔬 V22 Auto Select · genuine xG and markets on BOTH matches",
                         value=False,key="ziczac_v22_auto_pair",
                         help="Ranks every eligible same-day pair with already saved "
                              "verified historical xG and time-stamped bookmaker data. "
                              "Requires private login and a central database. "
                              "No external API calls and no unverified guesses.")
    auto_pick=st.checkbox("🤖 V21 AI Auto Select · evidence-qualified best pair",
                          value=False,key="ziczac_v21_auto_pair",
                          help="Compare ALL eligible same-day pairs using dated team and "
                               "home/away goal form. If none meet the evidence and "
                               "minimum estimated coverage requirements, abstain.")
    if auto_v22 or auto_v23:
        from v22_pair_selector import candidates as _rank_v22_pairs
        from v21_evidence_store import latest as _source_snapshot
        if not private_gate_ready(st):
            st.warning("V22/V23 auto selection needs your private ZicZac login "
                       "to access previously saved source packets.")
            return
        import os
        try:
            _database=st.secrets.get("DATABASE_URL","") or os.environ.get("DATABASE_URL","")
        except (OSError,KeyError,RuntimeError,AttributeError):
            _database=os.environ.get("DATABASE_URL","")
        if not _database:
            st.caption("No private research database is configured. "
                       "Checking the original source-matched, immutable "
                       "GitHub market packets instead; no extra API calls.")
        def _available_packet(fixture):
            saved=repository_provider_packet(fixture,now=clock,
                         priority=("market" if auto_v23 else "research"))
            private_record=(_source_snapshot(fixture,_database,before=clock)
                            if _database else None)
            return (saved or private_record) if auto_v23 else (private_record or saved)
        research_selector=(v23_shadow if auto_v23 else None)
        ranking=_rank_v22_pairs(found,_available_packet,now=clock,
                                 shadow_fn=research_selector)
        if ranking["abstained"]:
            st.warning(("V23" if auto_v23 else "V22")+
                       " Auto Select abstained: no same-day pair has "
                       "sufficient matching bookmaker evidence, goal-model "
                       "history and estimated probability coverage.")
            st.caption("Eligible source-backed fixtures: "+
                       str(ranking["source_qualified_fixtures"])+
                       " · reasons: "+str(ranking["reasons"]))
            return
        best=ranking["pairs"][0]
        first,second=best["first"],best["second"]
        st.caption(("V23" if auto_v23 else "V22")+" selected from "+
                   str(ranking["total_qualifying_pairs"])+
                   " genuinely qualified pairs. Conservative estimated "
                   "five-double coverage: "+
                   f"{best['conservative_coverage_estimate']:.2%}"+
                   ". This is not a verified hit rate.")
    elif auto_pick:
        from v21_pair_selector import candidates as _rank_researched_pairs
        ranking=_rank_researched_pairs(found,histories,now=clock)
        if ranking["abstained"]:
            st.warning("V21 Auto Select abstained: no pair had sufficient "
                       "verified team/venue history and model-estimated coverage. "
                       "This is not a football-data failure or a predicted loss.")
            st.caption("Screened "+str(ranking["screened_fixtures"])+
                       " fixtures; fully researched "+str(ranking["eligible_fixtures"])+
                       ". Reasons: "+str(ranking["reasons"]))
            return
        best=ranking["pairs"][0]
        first,second=best["first"],best["second"]
        st.caption("Compared "+str(ranking["pairs_above_threshold"])+
                   " qualifying pairs. Selected using the lower of baseline "
                   "and research goal-model estimated coverage: "+
                   f"{best['conservative_coverage']:.2%}"+
                   ". This is NOT observed accuracy.")
    else:
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
         "Balanced challenger (experimental)",
         "V22 Verified xG + Markets (research-only)",
         "V23 Market + Goal Form (no real xG required)"],
        # V20 form was slightly worse than original on matched historical
        # score log loss. Never default to an unproven challenger.
        index=1,key="ziczac_ai_five_selection_mode",
        help="Research goal-form recomputes a shadow score grid from verified "
             "last-ten home/away scoring and opposition concessions, only if "
             "both fixtures have complete original score grids and enough "
             "dated results. Original and balanced methods remain available.")
    if not research_ready:
        st.caption("Experimental goal-form mode needs more verified past home/away "
                   "matches for both teams plus complete score grids. "
                   "Original five doubles remain available without inventing research.")
    selection_mode = ("v23_market_form" if auto_v23 or mode_label.startswith("V23") else
                      "v22_xg_market" if auto_v22 or mode_label.startswith("V22") else
                      "goal_form_research" if mode_label.startswith("Research") else
                      "balanced_challenger" if mode_label.startswith("Balanced")
                      else "probability")
    if auto_v23 or auto_v22:
        st.caption(("V23 market-and-form" if auto_v23 else "V22 real-xG-and-market")+
                   " research scoring was selected by the corresponding "
                   "source-qualified automatic match selection.")
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
    source_consent=st.checkbox(
        "📡 Collect verified xG, lineups, injuries and odds for BOTH matches (API quota)",
        value=False,key="ziczac_v21_source_consent",
        help="Optional authenticated, user-approved API-Football calls. "
             "Past-fixture expected goals are sampled only where available; "
             "market quotes must be timestamped and lineups provider-reported. "
             "No API key or missing data is fabricated.")
    xg_depth=st.selectbox(
        "Past xG fixtures to check per team (only if research-source consent is ticked)",
        [2,6,8,12],index=2 if selection_mode=="v22_xg_market" else 0,
        key="ziczac_v22_xg_depth",
        help="Up to twice this many additional historical fixture-statistics "
             "requests PER selected match, plus lineup/injury/odds calls. "
             "API quota or fees may apply. Choose 2 for minimal usage.")
    if selection_mode=="v23_market_form":
        st.caption("V23 needs an authentic market-winner and goal-total snapshot "
                   "for BOTH matches. Without saved sources, tick the paid/quota "
                   "research-source checkbox. Try the lowest xG depth first; "
                   "V23 can work without real shot-based xG.")
    if source_consent and not private:
        st.warning("Verified provider source collection requires your private "
                   "ZicZac login. No extra provider calls will be made without it.")
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
          selection_mode,
          # New pre-match source evidence can arrive between user clicks.
          # Revisit V22/V23 once per 15m, never substitute stale cached
          # outcomes for newer checked market snapshots.
          int(clock.timestamp()//900) if selection_mode in
          ("v22_xg_market","v23_market_form") else 0)
    if clicked:
        if (pair not in cache or cache[pair].get("status")!="ready"
                or bool(source_consent) or bool(consent)):
            permitted=bool(consent and private and selected and configs)
            with st.spinner("Comparing statistical scores and available AI research..."):
                import os
                from v21_provider_evidence import capture as _capture_provider
                from v21_evidence_store import latest as _load_packet, persist as _save_packet
                try:
                    db_url=str(st.secrets.get("DATABASE_URL") or
                               os.environ.get("DATABASE_URL") or "")
                    football_key=str(st.secrets.get("API_FOOTBALL_KEY") or
                                     os.environ.get("API_FOOTBALL_KEY") or "")
                except (OSError,KeyError,RuntimeError,AttributeError):
                    db_url=str(os.environ.get("DATABASE_URL") or "")
                    football_key=str(os.environ.get("API_FOOTBALL_KEY") or "")
                packets=[]
                for item in (first,second):
                    existing_packet=(_load_packet(item["fixture"],db_url,before=clock)
                                     if private and db_url else None)
                    repo_packet=(repository_provider_packet(
                         item["fixture"],now=clock,
                         priority=("market" if selection_mode=="v23_market_form"
                                   else "research"))
                         if private and selection_mode in
                         ("v22_xg_market","v23_market_form") else None)
                    # GitHub-captured odds may be the only source available
                    # even when no private PostgreSQL database was configured.
                    packet=((repo_packet or existing_packet)
                            if selection_mode=="v23_market_form"
                            else (existing_packet or repo_packet))
                    if (selection_mode=="v22_xg_market" and packet is not None and
                            source_consent and private and football_key):
                        from v22_evidence_models import xg_profile
                        insufficient=(not xg_profile(item["fixture"],packet,as_of=clock).get("ready")
                                      or len(packet.get("multi_market") or {})<2)
                        if insufficient:
                            # Only the explicit consent click can trigger a
                            # deeper second packet. Both remain write-once.
                            packet=None
                    if source_consent and private and football_key and packet is None:
                        packet=_capture_provider(item["fixture"],
                                                _history(item,histories or {}),
                                                football_key,now=clock,
                                                max_history_per_team=xg_depth)
                        # Source requests may finish well after the page
                        # clock. Validate their honest completion timestamp
                        # against the time now, then use it for the next leg.
                        clock=utc(datetime.now(timezone.utc))
                        if not valid_snapshot(packet,item["fixture"],now=clock):
                            packet=None
                        elif db_url:
                            stored=_save_packet(packet,item["fixture"],db_url)
                            if not stored.get("durable"):
                                st.warning("Source evidence for "+
                                           item["fixture"]["homeTeam"]["name"]+
                                           " is NOT permanently stored ("+
                                           str(stored.get("state"))+").")
                        else:
                            st.warning("Source evidence is session-only. Configure "
                                       "a private DATABASE_URL to retain the packet.")
                    packets.append(packet)
                if source_consent and private and not football_key:
                    st.warning("API_FOOTBALL_KEY is missing from Streamlit Secrets. "
                               "No xG/lineup/odds calls were made.")
                report=analyze_pair(first,second,histories or {},
                    configs={k:configs[k] for k in selected} if permitted else {},
                    use_live=permitted,qwen_url=server_qwen_url(st),
                    selection_mode=selection_mode,provider_packets=packets)
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
                                    ai_provider=_provider,
                                    evidence_digest=_research.get("source_snapshot_digest") or ""),
                                    _db).get("durable",False))
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
