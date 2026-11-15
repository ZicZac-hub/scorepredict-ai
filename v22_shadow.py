"""V22 research goal probabilities, never a production override.

Genuine past fixture xG plus pre-match 1X2/total-goals/BTTS market evidence
must BOTH be present. Source-verified lineups/injuries are described only:
no invented player-adjustment coefficients. All probability transforms preserve
full score grids; unsupported fixtures explicitly abstain.
"""
import numpy as np

from v21_provider_evidence import valid_snapshot
from v22_evidence_models import (
    xg_profile, lineup_status, combine_goal_evidence, calibrate_markets)
from v21_goal_model_comparison import top_scores
from v10_prospective import utc


def shadow(row, packet, *, as_of=None, min_xg=6):
    fixture=row.get("fixture") or {}
    when=utc(as_of) if as_of is not None else None
    if when is None:
        return {"ready":False,"reason":"as_of_time_required"}
    if not valid_snapshot(packet,fixture,now=when):
        return {"ready":False,"reason":"missing_identity_matched_pre_match_packet"}
    result=row.get("result") or {}
    original=result.get("matrix")
    if original is None:
        return {"ready":False,"reason":"full_original_score_grid_missing"}
    try:
        base=np.asarray(original,dtype=float)
        if base.ndim!=2 or min(base.shape)<5 or not np.all(np.isfinite(base)) or np.any(base<0) or abs(base.sum()-1)>1e-5:
            raise ValueError("bad baseline")
    except (TypeError,ValueError):
        return {"ready":False,"reason":"baseline_matrix_invalid"}
    profile=xg_profile(fixture,packet,as_of=when,minimum=min_xg)
    if not profile.get("ready"):
        return {"ready":False,"reason":"verified_xg_"+str(profile.get("reason")),
                "xg_diagnostics":profile}
    with_xg=combine_goal_evidence(base,profile)
    if not with_xg.get("ready"):
        return {"ready":False,"reason":"xg_combination_unavailable"}
    prices=packet.get("multi_market") or {}
    # Quotes must remain fresh at the ACTUAL prediction time, not merely
    # when an earlier source packet was collected.
    from datetime import timedelta
    valid_prices={}
    for kind,market in prices.items():
        try:
            source_time=utc(market["last_source_update_utc"])
            if source_time<=when and when-source_time<=timedelta(hours=12):
                valid_prices[kind]=market
        except (KeyError,ValueError,TypeError):
            continue
    prices=valid_prices
    # One bookmaker with two independent market types is an allowed
    # research signal; number of independent bookmakers disclosed separately.
    tuned=calibrate_markets(with_xg["matrix"],prices)
    if not tuned.get("ready"):
        return {"ready":False,"reason":"market_"+str(tuned.get("reason")),
                "available_market_types":sorted(prices)}
    grid=tuned["matrix"]
    xi=lineup_status(fixture,packet,as_of=when)
    # Do not imply "lineups ready" when players named but no fitted
    # historical player impact; always zero numeric lineup adjustment.
    goal_home=float((grid*np.arange(grid.shape[0])[:,None]).sum())
    goal_away=float((grid*np.arange(grid.shape[1])[None,:]).sum())
    return {"ready":True,"matrix":grid,
            "scores":top_scores(grid),
            "home_xg":goal_home,"away_xg":goal_away,
            "expected_goals_source":"research_blend_not_observed_shot_xg",
            "historical_xg_counts":[profile["home_count"],profile["away_count"]],
            "venue_xg_counts":[profile["home_venue_count"],profile["away_venue_count"]],
            "source_snapshot_digest":packet["snapshot_sha256"],
            "source_captured_at_utc":packet["captured_at_utc"],
            "markets":tuned["market_types"],
            "books_by_type":{k:int((v or {}).get("n_bookmakers",0))
                             for k,v in prices.items() if k in tuned["market_types"]},
            "lineup_context":xi,"numerical_lineup_adjustment":0.0,
            "player_injury_impact":"unvalidated_no_numeric_weight",
            "market_total_variation":tuned["total_variation"],
            "prediction_method":"v22_xg_market_shadow_not_promoted",
            "observed_accuracy_unproven":True}
