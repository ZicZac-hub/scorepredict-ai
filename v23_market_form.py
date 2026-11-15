"""V23 market-and-form exact-score SHADOW: works without fabricated shot xG.

V22 correctly abstains if verified historical shot xG is absent; V23 adds
a separate path when genuine pre-match 1X2 AND goal-total odds exist and
the independent original full score grid has sufficient model evidence.
It does NOT invent xG, injuries, unobserved lineups, player impacts or odds.
"""
import numpy as np
from datetime import timedelta

from v10_prospective import utc
from v21_provider_evidence import valid_snapshot
from v21_goal_model_comparison import top_scores
from v22_evidence_models import calibrate_markets,xg_profile,combine_goal_evidence,lineup_status
from goal_research_v20 import goal_form_challenger


def shadow(row, packet, *, as_of, history=None, min_league_history=45):
    fixture=row.get("fixture") or {}
    if as_of is None:
        return {"ready":False,"reason":"as_of_missing"}
    clock=utc(as_of)
    if not valid_snapshot(packet,fixture,now=clock):
        return {"ready":False,"reason":"missing_or_mismatched_provider_snapshot"}
    original=row.get("result") or {}
    try:
        grid=np.asarray(original["matrix"],dtype=float)
        if (grid.ndim!=2 or min(grid.shape)<5 or
            np.any(~np.isfinite(grid)) or np.any(grid<0) or
            abs(float(grid.sum())-1)>1e-5):
            raise ValueError("invalid prior")
    except (TypeError,KeyError,ValueError):
        return {"ready":False,"reason":"full_original_probability_grid_required"}
    league_count=(row.get("league_history_matches") or
                  row.get("history_matches_reported") or
                  original.get("league_history_matches"))
    if league_count is not None:
        try:
            if int(league_count)<min_league_history:
                return {"ready":False,"reason":"insufficient_league_model_history"}
        except (ValueError,TypeError):
            return {"ready":False,"reason":"unverified_league_history_count"}

    markets=packet.get("multi_market") or {}
    checked={}
    for kind,market in markets.items():
        try:
            updated=utc(market["last_source_update_utc"])
            if updated<=utc(fixture["utcDate"]) and updated<=clock and clock-updated<=timedelta(hours=12):
                checked[kind]=market
        except (TypeError,ValueError,KeyError):
            continue
    if "1x2" not in checked or not ({"over_2_5","over_3_5"} & set(checked)):
        return {"ready":False,"reason":"timestamped_1x2_and_goal_total_markets_required",
                "market_types_available":sorted(checked)}

    # Independent genuine xG is optional, not a necessary condition. Avoid
    # treating raw goals and Elo-calculated goal means as true shot-based xG.
    genuine=xg_profile(fixture,packet,as_of=clock,minimum=6)
    has_xg=bool(genuine.get("ready"))
    applied=[]
    context={}
    if has_xg:
        r=combine_goal_evidence(grid,genuine,weight=.20)
        if r.get("ready"):
            grid=r["matrix"]
            applied.append("genuine_historical_shot_xg")
            context["genuine_xg_history_counts"]={
                "home":genuine["home_count"],"away":genuine["away_count"]}
    elif history:
        form=goal_form_challenger(row,history,clock)
        if form.get("available"):
            # Research-only small blend, V20 is not demonstrated to beat V8.
            grid=.8*grid+.2*np.asarray(form["matrix"],dtype=float)
            grid/=grid.sum()
            applied.append("venue_goal_form_conservative_shadow")
            context["goal_form_observations"]={
                key:form["profile"][key]["matches"] for key in
                ("home_last10","away_last10",
                 "home_at_home_last10","away_away_last10")}
    else:
        context["goal_form_observations"]="not_supplied; original V8 prior only"
    if not applied:
        applied.append("unchanged_original_goal_prior")

    grid_before=grid.copy()
    tuned=calibrate_markets(grid,checked,strength=.35,max_move=.15)
    if not tuned.get("ready"):
        return {"ready":False,"reason":"invalid_market_probability_vector"}
    grid=tuned["matrix"]
    h=np.arange(grid.shape[0])[:,None]
    a=np.arange(grid.shape[1])[None,:]
    baseline=np.asarray(original["matrix"],dtype=float)
    source_quality={
        "bookmakers":{kind:checked[kind].get("n_bookmakers",0) for kind in sorted(checked)},
        "quote_timestamps":{kind:checked[kind]["last_source_update_utc"] for kind in sorted(checked)},
        "historical_xg_available":has_xg,
        "lineups":lineup_status(fixture,packet,as_of=clock).get("ready",False),
        "unverified_player_impact_weight":0.0
    }
    return {
        "ready":True,"model":"v23_market_form_shadow_not_validated",
        "matrix":grid,"scores":top_scores(grid),
        "home_xg":float((grid*h).sum()),"away_xg":float((grid*a).sum()),
        "goal_estimate_kind":"statistical_expectation_not_observed_shot_xg",
        "market_types":tuned["market_types"],
        "applied_signals":applied+["timestamped_1x2_and_goal_totals"],
        "snapshot_sha256":packet["snapshot_sha256"],
        "source_snapshot_digest":packet["snapshot_sha256"],
        "prediction_method":"v23_market_form_shadow_not_validated",
        "prior_over_2_5_probability":float(baseline[h+a>=3].sum()),
        "adjusted_over_2_5_probability":float(grid[h+a>=3].sum()),
        "prior_four_plus_goals":float(baseline[h+a>=4].sum()),
        "adjusted_four_plus_goals":float(grid[h+a>=4].sum()),
        "market_total_variation":float(np.abs(grid-grid_before).sum()/2),
        "source_quality":source_quality,"context":context,
        "proven_accuracy_gain":False,"production_promotion":False,
        "no_unvalidated_AI_numeric_weight":True
    }
