"""ZicZac v10 shadow ledger: genuine before-kickoff v8/v9 forecasts.

This is a non-destructive *research* pipeline. It writes immutable complete
score grids under data/v10_prospective/forecasts, independently of the
existing v6/v7 collector. No API keys, fake xG, or post-match observations
are stored. Market v9 is emitted ONLY when real timestamped odds exist.
"""
from datetime import datetime, timedelta, timezone
from pathlib import Path
import math
import numpy as np

from forecast_ledger import append_file_record, read_file_records
from market_score_v9 import select_market_response, predict as market_predict
from research_ensemble_v8 import predict as research_predict

ROOT = Path("data/v10_prospective")
POLICY = "v10-shadow-same-evidence-v1"


def utc(value):
    if isinstance(value, datetime):
        result = value
    else:
        result = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if result.tzinfo is None:
        raise ValueError("Pre-match evidence timestamps must be timezone-aware")
    return result.astimezone(timezone.utc)


def _identity(fixture):
    return (str(fixture.get("provider") or "API-Football"),
            str(fixture["id"]))


def _record(fixture, evidence, forecast, model_key, captured_at, market=None):
    start = utc(evidence["as_of_utc"])
    now = utc(captured_at)
    kickoff = utc(fixture["utcDate"])
    if not start <= now < kickoff:
        raise ValueError("Future or after-kickoff forecast cannot be archived")
    grid = np.asarray(forecast["matrix"], dtype=float)
    if grid.ndim != 2 or not np.isfinite(grid).all() or np.any(grid < 0) or abs(grid.sum()-1)>1e-7:
        raise ValueError("Score grid is invalid")
    provider, fixture_id = _identity(fixture)
    home, away = fixture["homeTeam"], fixture["awayTeam"]
    from verified_features_v10 import describe as verified_schedule_context
    research_context = verified_schedule_context(fixture,evidence)
    scores = [[int(h), int(a), float(p)] for h,a,p in forecast["scores"]]
    if not scores:
        raise ValueError("No valid exact-score outcomes")
    row = {
        "mode":"prospective", "score_basis":"regulation", "schema_version":1,
        "model_key":model_key, "model_version":forecast["model_version"],
        "provider":provider, "fixture_id":fixture_id, "fixture_name":
            str(home.get("name") or "")+" vs "+str(away.get("name") or ""),
        "home_team_id":str(home["id"]), "away_team_id":str(away["id"]),
        "home_team":str(home.get("name") or ""), "away_team":str(away.get("name") or ""),
        "competition":str((fixture.get("competition") or {}).get("code") or ""),
        "kickoff_utc":kickoff.isoformat(), "evidence_as_of_utc":start.isoformat(),
        "captured_at_utc":now.isoformat(), "horizon":"research",
        "input_digest":evidence["input_digest"], "policy_version":POLICY,
        "evidence_policy":evidence.get("policy_version"),
        "score_matrix":grid.tolist(), "matrix_shape":list(grid.shape),
        "matrix_mass":float(grid.sum()),
        "top10_scores":scores, "predicted_score":scores[0][:2],
        "predicted_score_probability":scores[0][2],
        "outcome_probabilities":{"home":float(np.tril(grid,-1).sum()),
                                 "draw":float(np.trace(grid)),
                                 "away":float(np.triu(grid,1).sum())},
        "expected_goals":{"home":float(forecast["home_xg"]),
                          "away":float(forecast["away_xg"])},
        "history_matches":len(evidence.get("history") or []),
        "league_history_matches":len(evidence.get("league_history") or []),
        "component_weights":dict(forecast.get("component_weights") or {}),
        "research_context":research_context,
        "market_used":bool(market),
        "market_snapshot":({key:market[key] for key in (
            "bookmaker","bookmaker_count","bookmaker_names","aggregation",
            "snapshot_utc","winner","winner_overround",
            "totals","totals_overround","settlement")
            if key in market} if market else None),
        "unavailable_features":["verified_xg","player_strength",
            "confirmed_starting_xi_adjustment","independent_llm_votes"],
        "release_status":"shadow_validation_only"}
    return row


def capture(fixture, evidence, captured_at, *, api_key=None,
            request=None, root=ROOT, market_calls_remaining=0):
    """Write original v8 distribution and optional v9, returning diagnostics.

    Always use the SAME already-verified historical evidence for both models.
    A missing market has NO invented fallback model key. An invalid/late market
    is not promoted to a verified forecast.
    """
    when=utc(captured_at)
    cutoff=utc(evidence["as_of_utc"])
    kickoff=utc(fixture["utcDate"])
    if when>=kickoff or cutoff>when or not evidence.get("input_digest"):
        raise ValueError("Verified pre-kickoff evidence digest and time are required")
    history=evidence.get("history") or []
    league=evidence.get("league_history") or history
    forecast=research_predict(fixture,history,league_history=league,as_of=cutoff)
    if forecast is None or len(forecast.get("component_weights") or {})<2:
        from v10_source_readiness import history_readiness
        readiness = history_readiness(fixture, history, league, cutoff)
        # V37 is a separate, explicitly NOT V28-eligible research challenger.
        # It can learn from 2-4 verified same-competition games while V8
        # correctly abstains under its unchanged five-match requirement.
        sparse = {"v37_recorded":False,
                  "v37_reason":"sparse_history_research_not_attempted"}
        if readiness["reason"] == "team_history_below_five":
            try:
                from v37_sparse_history_research import freeze as freeze_sparse
                sparse = freeze_sparse(fixture,evidence,when,root=root)
            except (ValueError,TypeError,KeyError,RuntimeError,OverflowError) as exc:
                sparse = {"v37_recorded":False,
                          "v37_reason":"shadow_unavailable_"+type(exc).__name__}
        return {"v8_recorded":False,"v9_recorded":False,"v12_recorded":False,
                "reason":"insufficient_verified_research_evidence",
                "readiness_reason":readiness["reason"],
                "team_history_readiness":readiness,
                **sparse,
                "market_calls":0}
    frozen_at=max(when,utc(datetime.now(timezone.utc)))
    if frozen_at>=kickoff:
        return {"v8_recorded":False,"v9_recorded":False,"v12_recorded":False,
                "reason":"calculation_finished_after_kickoff","market_calls":0}
    base=_record(fixture,evidence,forecast,"v8",frozen_at)
    written_v8=append_file_record(base,root=root,kind="forecasts")
    outcome={"v8_recorded":bool(written_v8),"v9_recorded":False,
             "v10_recorded":False,"v12_recorded":False,
             "v13_recorded":False,"reason":"market_not_queried","market_calls":0}
    # A distinct, write-once v12 calibration forecast. It uses this precise
    # pre-match v8 matrix and the same already-verified league history.
    # It never changes v8, consumes no external API calls, and abstains if
    # 80 qualified league results are unavailable.
    if written_v8:
        try:
            from score_calibration_v12 import predict as calibrated_predict
            calibrated=calibrated_predict(
                fixture,history,league_history=league,as_of=cutoff,
                reference_forecast=forecast)
            completed=max(when,utc(datetime.now(timezone.utc)))
            if calibrated is not None and completed<kickoff:
                row=_record(fixture,evidence,calibrated,"v12",completed)
                row["v12_diagnostics"]=calibrated["v12_diagnostics"]
                row["release_status"]="experimental_shadow_only_not_promoted"
                outcome["v12_recorded"]=bool(append_file_record(
                    row,root=root,kind="forecasts"))
            else:
                outcome["v12_status"]=("insufficient_historical_league_data"
                                      if calibrated is None else
                                      "calculation_finished_after_kickoff")
        except (ValueError,TypeError,KeyError,RuntimeError,OverflowError) as exc:
            # A candidate may fail; do not block the original collector.
            outcome["v12_status"]="calibration_unavailable_"+type(exc).__name__
    # Independent V13 combined-model shadow: same verified history, as-of
    # time, and immutable input digest as the newly written V8 forecast.
    # No external AI/odds calls, no replacement of any prior prediction.
    if written_v8:
        try:
            from hybrid_five_v13 import predict as hybrid_predict
            hybrid=hybrid_predict(fixture,history,league_history=league,
                                  as_of=cutoff,reference_v8=forecast)
            completed=max(when,utc(datetime.now(timezone.utc)))
            if hybrid is not None and completed<kickoff:
                row=_record(fixture,evidence,hybrid,"v13",completed)
                row["hybrid_component_weights"]=dict(hybrid["component_weights"])
                # Store only genuine same-as-of V7/V8 numerical grids on this
                # NEW V13 shadow entry. This permits a future five-double
                # selection to be frozen and independently graded later.
                # Never backfill older predictions once results are known.
                row["v14_component_grids"]={
                    key:np.asarray(value,dtype=float).tolist()
                    for key,value in (hybrid.get("component_score_matrices") or {}).items()
                    if key in ("v7","v8")}
                row["v14_component_versions"]={
                    key:version for key,version in
                    (hybrid.get("component_versions") or {}).items()
                    if key in ("v7","v8")}
                row["v7_fitted_top10_scores"]=(
                    (hybrid.get("component_score_distributions") or {}).get("v7") or [])
                row["release_status"]="experimental_shadow_only_not_promoted"
                outcome["v13_recorded"]=bool(append_file_record(
                    row,root=root,kind="forecasts"))
            else:
                outcome["v13_status"]=("insufficient_distinct_verified_components"
                                       if hybrid is None else
                                       "calculation_finished_after_kickoff")
        except (ValueError,TypeError,KeyError,RuntimeError,OverflowError,
                ArithmeticError) as exc:
            outcome["v13_status"]="hybrid_unavailable_"+type(exc).__name__
    # Do not use quota for distant games, unsupported providers or re-recording
    # the same slot: v8 remains the only stored prediction when odds are absent.
    if (not api_key or request is None or market_calls_remaining<=0
            or not written_v8 or fixture.get("provider")!="API-Football"
            or not timedelta(minutes=20)<kickoff-when<=timedelta(days=2)):
        return outcome
    outcome["market_calls"]=1
    try:
        packet=request(api_key,"/odds",{"fixture":fixture["id"]},with_metadata=True)
        if not isinstance(packet,dict):
            outcome["reason"]="no_timed_provider_metadata"
            return outcome
        meta=packet.get("metadata") or {}
        retrieved=utc(meta["retrieved_at_utc"])
        fetched=utc(meta["fetched_at_utc"])
        # A cached response from BEFORE the as-of date can remain valid, but
        # a response fetched AFTER the captured forecast cannot enter history.
        if (meta.get("stale") or meta.get("freshness_known") is not True
                or fetched>retrieved or fetched>utc(datetime.now(timezone.utc))
                or retrieved>utc(datetime.now(timezone.utc))+timedelta(seconds=10)):
            outcome["reason"]="unverified_or_stale_market_source"
            return outcome
        market_observed=max(when,utc(datetime.now(timezone.utc)))
        if market_observed>=kickoff or fetched>market_observed:
            outcome["reason"]="odds_received_after_kickoff"
            return outcome
        market=select_market_response(packet.get("data") or [],fixture,as_of=market_observed)
        if not market:
            outcome["reason"]="no_verified_prematch_market"
            return outcome
        revised=market_predict(fixture,history,league_history=league,market=market,as_of=market_observed)
        if not revised or not revised.get("market_used"):
            outcome["reason"]="market_model_unavailable"
            return outcome
        completed=max(market_observed,utc(datetime.now(timezone.utc)))
        if completed>=kickoff:
            outcome["reason"]="market_calculation_finished_after_kickoff"
            return outcome
        row=_record(fixture,evidence,revised,"v9",completed,market=market)
        outcome["v9_recorded"]=bool(append_file_record(row,root=root,kind="forecasts"))
        # A separate multi-bookmaker challenger only exists if at least
        # THREE independent vendor quotes are observed for this exact match.
        # Reuse the SAME provider response; never spend more API requests.
        from multi_market_v10 import market_consensus, predict as predict_multi
        consensus=market_consensus(packet.get("data") or [],fixture,as_of=market_observed)
        if consensus:
            combined=predict_multi(fixture,history,league_history=league,
                                   market=consensus,as_of=market_observed)
            complete=utc(datetime.now(timezone.utc))
            if combined and complete<kickoff:
                candidate=_record(fixture,evidence,combined,"v10",
                                  max(complete,market_observed),market=consensus)
                outcome["v10_recorded"]=bool(append_file_record(
                    candidate,root=root,kind="forecasts"))
        outcome["reason"]="verified_market_models_archived" if outcome["v9_recorded"] else "market_already_archived"
        return outcome
    except (ValueError,TypeError,KeyError,RuntimeError,LookupError) as exc:
        outcome["reason"]="market_unavailable_"+type(exc).__name__
        return outcome


def capture_near_research(fixture,evidence,captured_at,*,root=ROOT):
    """Frozen v8 near-kickoff challenger, with new as-of history only.

    Write at most ONE research refresh for a fixture. Do not alter the
    already-saved original v8 snapshot. Make no extra bookmaker requests.
    A newer as-of model is not necessarily a more accurate model.
    """
    now=utc(captured_at)
    kickoff=utc(fixture["utcDate"])
    original=read_latest(root)
    identity=_identity(fixture)
    result={"v8_near_recorded":False,"reason":"not_eligible"}
    if (*identity,"v8") not in original:
        result["reason"]="no_existing_original_v8"
        return result
    if (*identity,"v8_near") in original:
        result["reason"]="near_research_already_frozen"
        return result
    if not timedelta(minutes=45)<kickoff-now<=timedelta(hours=48):
        result["reason"]="outside_near_kickoff_window"
        return result
    cutoff=utc(evidence["as_of_utc"])
    if cutoff>now or not evidence.get("input_digest"):
        raise ValueError("New research requires verified before-kickoff evidence")
    history=evidence.get("history") or []
    league=evidence.get("league_history") or history
    candidate=research_predict(fixture,history,league_history=league,as_of=cutoff)
    if candidate is None or len(candidate.get("component_weights") or {})<2:
        result["reason"]="insufficient_current_history"
        return result
    completed=max(now,utc(datetime.now(timezone.utc)))
    if completed>=kickoff:
        result["reason"]="research_finished_after_kickoff"
        return result
    challenger=dict(candidate)
    challenger["model_version"]="ziczac-v8-near-kickoff-research-v1"
    record=_record(fixture,evidence,challenger,"v8_near",completed)
    record["numerical_model_version"]=candidate["model_version"]
    record["horizon"]="near_kickoff"
    record["early_original_forecast_id"]=original[(*identity,"v8")].get("forecast_id")
    result["v8_near_recorded"]=bool(append_file_record(
        record,root=root,kind="forecasts"))
    result["reason"]="research_refreshed" if result["v8_near_recorded"] else "already_recorded"
    return result


def capture_market_followup(fixture,evidence,captured_at,*,api_key=None,
                            request=None,root=ROOT,market_calls_remaining=0):
    """Near-kickoff fresh market check for a fixture ALREADY frozen in v8.

    Saves a separately named SAME-EVIDENCE v8 comparison control alongside
    verified v9/v10. The original early v8 score is not changed. One market
    request per fixture, only if v9 is not yet present; no invented markets.
    """
    before=utc(captured_at)
    kickoff=utc(fixture["utcDate"])
    identity=_identity(fixture)
    diagnostics={"v8_control_recorded":False,"v9_recorded":False,
                 "v10_recorded":False,"market_calls":0,
                 "reason":"not_eligible"}
    if (not api_key or request is None or market_calls_remaining<=0
        or fixture.get("provider")!="API-Football"
        or not timedelta(minutes=30)<kickoff-before<=timedelta(hours=48)):
        return diagnostics
    saved=read_latest(root)
    if (*identity,"v8") not in saved:
        diagnostics["reason"]="no_original_frozen_v8"
        return diagnostics
    if (*identity,"v9") in saved:
        diagnostics["reason"]="v9_already_frozen"
        return diagnostics
    cutoff=utc(evidence["as_of_utc"])
    if cutoff>before or not evidence.get("input_digest"):
        raise ValueError("New market comparison requires verified as-of evidence")
    history=evidence.get("history") or []
    league=evidence.get("league_history") or history
    base=research_predict(fixture,history,league_history=league,as_of=cutoff)
    if base is None or len(base.get("component_weights") or {})<2:
        diagnostics["reason"]="insufficient_current_history"
        return diagnostics
    diagnostics["market_calls"]=1
    try:
        packet=request(api_key,"/odds",{"fixture":fixture["id"]},
                       with_metadata=True)
        meta=packet.get("metadata") or {}
        fetched=utc(meta["fetched_at_utc"])
        retrieved=utc(meta["retrieved_at_utc"])
        observed=max(before,utc(datetime.now(timezone.utc)))
        if (meta.get("freshness_known") is not True or meta.get("stale")
                or fetched>retrieved or fetched>observed or observed>=kickoff):
            diagnostics["reason"]="market_metadata_not_prematch_fresh"
            return diagnostics
        market=select_market_response(packet.get("data") or [],fixture,as_of=observed)
        if not market:
            diagnostics["reason"]="no_matching_timestamped_market"
            return diagnostics
        updated=market_predict(fixture,history,league_history=league,
                               market=market,as_of=observed)
        if updated is None or not updated.get("market_used"):
            diagnostics["reason"]="v9_market_unavailable"
            return diagnostics
        frozen=max(observed,utc(datetime.now(timezone.utc)))
        if frozen>=kickoff:
            diagnostics["reason"]="market_calculation_finished_after_kickoff"
            return diagnostics
        control=dict(base)
        control["model_version"]="ziczac-v8-market-time-matched-control-v1"
        original=_record(fixture,evidence,control,"v8_control",frozen)
        original["numerical_model_version"]=base["model_version"]
        original["horizon"]="market_control"
        comparison=_record(fixture,evidence,updated,"v9",frozen,market=market)
        # Neither data source nor original early pre-match forecast is altered.
        diagnostics["v8_control_recorded"]=bool(append_file_record(
            original,root=root,kind="forecasts"))
        diagnostics["v9_recorded"]=bool(append_file_record(
            comparison,root=root,kind="forecasts"))
        from multi_market_v10 import market_consensus, predict as predict_multi
        consensus=market_consensus(packet.get("data") or [],fixture,as_of=observed)
        if consensus:
            candidate=predict_multi(fixture,history,league_history=league,
                                    market=consensus,as_of=observed)
            complete=utc(datetime.now(timezone.utc))
            if candidate is not None and complete<kickoff:
                row=_record(fixture,evidence,candidate,"v10",complete,market=consensus)
                diagnostics["v10_recorded"]=bool(append_file_record(
                    row,root=root,kind="forecasts"))
        diagnostics["reason"]="verified_market_comparison_saved"
        return diagnostics
    except (KeyError,ValueError,TypeError,RuntimeError,LookupError) as exc:
        diagnostics["reason"]="market_source_"+type(exc).__name__
        return diagnostics


def read_latest(root=ROOT):
    """Return earliest independently saved forecast for each model/fixture."""
    found={}
    for row in read_file_records(root=root,kind="forecasts"):
        if row.get("model_key") not in ("v8","v8_near","v8_control","v9","v10","v12","v13"):
            continue
        try:
            key=(row["provider"],str(row["fixture_id"]),row["model_key"])
            if utc(row["captured_at_utc"])>=utc(row["kickoff_utc"]):
                continue
            prior=found.get(key)
            if prior is None or utc(row["captured_at_utc"])<utc(prior["captured_at_utc"]):
                found[key]=row
        except (KeyError,ValueError,TypeError):
            continue
    return found
