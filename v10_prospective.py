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
        "market_used":bool(market),
        "market_snapshot":({key:market[key] for key in (
            "bookmaker","snapshot_utc","winner","winner_overround",
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
        return {"v8_recorded":False,"v9_recorded":False,
                "reason":"insufficient_verified_research_evidence","market_calls":0}
    base=_record(fixture,evidence,forecast,"v8",when)
    written_v8=append_file_record(base,root=root,kind="forecasts")
    outcome={"v8_recorded":bool(written_v8),"v9_recorded":False,
             "reason":"market_not_queried","market_calls":0}
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
        market=select_market_response(packet.get("data") or [],fixture,as_of=when)
        if not market:
            outcome["reason"]="no_verified_prematch_market"
            return outcome
        revised=market_predict(fixture,history,league_history=league,market=market,as_of=when)
        if not revised or not revised.get("market_used"):
            outcome["reason"]="market_model_unavailable"
            return outcome
        row=_record(fixture,evidence,revised,"v9",when,market=market)
        outcome["v9_recorded"]=bool(append_file_record(row,root=root,kind="forecasts"))
        outcome["reason"]="market_verified_and_archived" if outcome["v9_recorded"] else "market_already_archived"
        return outcome
    except (ValueError,TypeError,KeyError,RuntimeError,LookupError) as exc:
        outcome["reason"]="market_unavailable_"+type(exc).__name__
        return outcome


def read_latest(root=ROOT):
    """Return earliest independently saved forecast for each model/fixture."""
    found={}
    for row in read_file_records(root=root,kind="forecasts"):
        if row.get("model_key") not in ("v8","v9"):
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
