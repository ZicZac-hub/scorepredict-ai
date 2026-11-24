"""Apply verified archived bookmaker evidence to a new score forecast."""

from datetime import datetime, timezone
from itertools import chain
from pathlib import Path

from market_score_v9 import _summarize
from persist_prematch import load_json
from v10_prospective import ROOT as FORECAST_ROOT, utc
from v16_pair_research import original_model
from v21_capture_sources import qualified_packet, source_packet_path
from v21_provider_evidence import _identity
from v22_evidence_models import calibrate_markets
from v24_bulk_market_capture import ROOT as CATALOG_ROOT, SOURCE_ROOT, catalog_rejection_reason
from v41_unified_trial import _markets


VERSION = "v8-archived-market-v1"
REQUIRED_MARKETS = ("1x2", "over_2_5")
STRENGTH = .35
MAX_MOVE = .15


def _catalog_packets(fixture, identity, root, cutoff):
    fixture_id, _, _, kickoff = identity
    competition = fixture.get("competition") or {}
    source = {"fixture_id": fixture_id, "kickoff_utc": kickoff,
              "competition": competition.get("code")}
    folder = Path(root) / "fixtures" / fixture_id
    for path in folder.glob("*.json"):
        try:
            row = load_json(path, {})
            if (row.get("schema") != 1 or row.get("snapshot_sha256") != path.stem
                    or utc(row["source_fixture_kickoff_utc"]) != kickoff
                    or catalog_rejection_reason(row, source, cutoff) is not None):
                continue
            league, fixture_league = row.get("source_league_id"), competition.get("id")
            if league not in (None, "") and fixture_league not in (None, "") and str(league) != str(fixture_league):
                continue
            yield {"source": "V24 market catalog", "captured_at_utc": row["captured_at_utc"],
                   "snapshot_sha256": row["snapshot_sha256"], "multi_market": row["market_types"]}
        except (OSError, ValueError, TypeError, KeyError, AttributeError, ArithmeticError):
            continue


def _source_packets(identity, source_root, forecast_root, cutoff):
    fixture_id, home, away, kickoff = identity
    root = Path(source_root)
    pattern = f"API-Football_{fixture_id}_*.json"
    paths = [*root.glob(pattern), *(root / "versions").glob(pattern)]
    for path in paths:
        try:
            row = load_json(path, {})
            reference = row["source_forecast"]
            if (reference["provider"] != "API-Football" or str(reference["fixture_id"]) != fixture_id
                    or str(reference["home_team_id"]) != home or str(reference["away_team_id"]) != away
                    or utc(reference["kickoff_utc"]) != kickoff):
                continue
            original = original_model(forecast_root, reference)
            if (original is None or utc(original["captured_at_utc"]) > cutoff
                    or not qualified_packet(row, original, now=cutoff)):
                continue
            packet = row["provider_snapshot"]
            base = source_packet_path(root, original)
            if path not in (base, root / "versions" / f"{base.stem}_{packet['snapshot_sha256'][:16]}.json"):
                continue
            limits = packet.get("source_limits") or {}
            if limits.get("bulk_market_catalog_sha256") and (
                utc(limits["source_fixture_kickoff_utc"]) != kickoff
                or utc(limits["provider_fetched_at_utc"]) > utc(packet["captured_at_utc"])
            ):
                continue
            yield {**packet, "source": "V21 fixture source"}
        except (OSError, ValueError, TypeError, KeyError, AttributeError, ArithmeticError):
            continue


def archived_market(fixture, *, now=None, catalog_root=CATALOG_ROOT,
                    source_root=SOURCE_ROOT, forecast_root=FORECAST_ROOT):
    """Return the freshest complete eligible market packet using only repository data."""
    try:
        cutoff = utc(now if now is not None else datetime.now(timezone.utc))
        identity = _identity(fixture)
    except (ValueError, TypeError, AttributeError, ArithmeticError):
        return None
    if identity is None or cutoff >= identity[-1]:
        return None

    best = None
    candidates = chain(_catalog_packets(fixture, identity, catalog_root, cutoff),
                       _source_packets(identity, source_root, forecast_root, cutoff))
    for packet in candidates:
        try:
            complete = _markets(packet, cutoff)
            if any(kind not in complete for kind in REQUIRED_MARKETS):
                continue
            markets = {kind: complete[kind] for kind in REQUIRED_MARKETS}
            oldest_update = min(utc(market["last_source_update_utc"]) for market in markets.values())
            captured = utc(packet["captured_at_utc"])
            rank = (oldest_update, captured, packet["snapshot_sha256"])
            if best is None or rank > best[0]:
                receipt = {key: packet[key] for key in ("source", "captured_at_utc", "snapshot_sha256")}
                best = rank, {"markets": markets, "receipt": receipt}
        except (TypeError, KeyError, ValueError, AttributeError, ArithmeticError):
            continue
    return best[1] if best is not None else None


def adjust_forecast(fixture, result, *, now=None, catalog_root=CATALOG_ROOT,
                    source_root=SOURCE_ROOT, forecast_root=FORECAST_ROOT):
    """Return a separate market research forecast or the original result unchanged."""
    if result is None:
        return None
    market = archived_market(fixture, now=now, catalog_root=catalog_root,
                             source_root=source_root, forecast_root=forecast_root)
    if market is None:
        return result
    try:
        update = calibrate_markets(result["matrix"], market["markets"],
                                   strength=STRENGTH, max_move=MAX_MOVE)
        if not update["ready"]:
            return result
        adjusted = _summarize(update["matrix"], result, market["receipt"])
        leading_score = adjusted["scores"][0][:2]
        agreement = sum(tuple(component["predicted_score"]) == leading_score
                        for component in (result.get("components") or {}).values())
        return {**adjusted, "model_version": VERSION, "release_status": "historical_replay_supported",
                "component_top_score_agreement": agreement,
                "limitations": "Historical replay supports the market adjustment. Live source coverage varies. "
                               "Verified Five Doubles accuracy is measured separately.",
                "market_types": update["market_types"],
                "market_total_variation": update["total_variation"]}
    except (ValueError, TypeError, KeyError, AttributeError, ArithmeticError):
        return result
