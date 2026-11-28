"""Capture bookmaker snapshots for the two legs of a frozen five-double pair."""
from collections import Counter
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import re

from api_football import LEAGUES, ProviderAccessError, ProviderQuotaError, _request
from five_double_accuracy import SYDNEY
from persist_prematch import load_json, write_json
from source_capture_clock import capture_clock
from v10_prospective import ROOT as SOURCE_ROOT, utc
from v21_capture_sources import qualified_packet, source_packet_path
from v22_evidence_models import parse_markets
from v24_bulk_market_capture import (
    DEFAULT_DAILY_REQUESTS, MAX_DAILY_REQUESTS, ROOT as BROAD_ROOT,
)
from v42_research_trial import PAIR_ROOT, _markets, _pair
from v44_early_history import _once, digest

ROOT = Path("data/v45_market_capture")
VERSION = "v45-paired-api-football-markets-v1"
MARKER_PREFIX = "v45_target_"
TARGETED_DAILY_REQUESTS = 4
MAX_CACHE_SECONDS = 300
MAX_QUOTE_AGE = timedelta(hours=3)
MAX_LEAD = timedelta(minutes=35)
LATE_LEAD = timedelta(minutes=20)
MIN_LEAD = timedelta(minutes=5)
MARKETS = {"1x2", "over_2_5", "over_3_5"}
MODEL_IDENTITY = ("provider", "fixture_id", "forecast_id", "input_digest", "home_team_id",
                  "away_team_id", "kickoff_utc")
BAD_RECORD = (OSError, AttributeError, ArithmeticError, KeyError, TypeError, ValueError)


class CaptureRejected(ValueError):
    """A local, sanitized validation reason for an attempted capture."""


def phase_at(kickoff, now):
    lead = utc(kickoff) - utc(now)
    if MIN_LEAD < lead <= LATE_LEAD:
        return "late"
    if LATE_LEAD < lead <= MAX_LEAD:
        return "early"
    return None


def _enabled(environ, enabled):
    return (str(environ.get("V45_MARKET_ENABLED", "")).lower() != "false"
            if enabled is None else bool(enabled))


def _daily_limit(environ):
    try:
        value = int(environ.get("V24_DAILY_ODDS_REQUESTS") or DEFAULT_DAILY_REQUESTS)
    except (TypeError, ValueError):
        value = DEFAULT_DAILY_REQUESTS
    return max(0, min(MAX_DAILY_REQUESTS, value))


def _request_folder(broad_root, day):
    return Path(broad_root) / "requests" / day.isoformat()


def _usage(broad_root, now):
    entries = list(_request_folder(broad_root, now.date()).glob("*.json"))
    return len(entries), sum(path.name.startswith(MARKER_PREFIX) for path in entries)


def broad_daily_limit(*, broad_root=BROAD_ROOT, now=None, environ=None, enabled=None):
    """Preserve the targeted reserve inside the existing shared UTC-day budget."""
    env = os.environ if environ is None else environ
    clock = utc(now or datetime.now(timezone.utc))
    limit = _daily_limit(env)
    if not _enabled(env, enabled):
        return limit
    _, targeted = _usage(broad_root, clock)
    reserved = min(limit, TARGETED_DAILY_REQUESTS)
    return limit - max(0, reserved - targeted)


def _positive_id(value):
    text = str(value)
    return text if re.fullmatch(r"[1-9][0-9]*", text) else None


def _source_path(root, model, sha):
    base = source_packet_path(root, model)
    return Path(root) / "versions" / f"{base.stem}_{sha}.json"


def _market_coverage(snapshot, cutoff):
    markets = _markets(snapshot, cutoff)
    verified = {}
    for kind, market in markets.items():
        quotes = market.get("quotes")
        if (not isinstance(quotes, list) or not quotes or market.get("n_bookmakers") != len(quotes)
                or len({quote["bookmaker"].casefold() for quote in quotes}) != len(quotes)):
            continue
        if all(quote["timestamp_basis"] == "bookmaker_quote_updated_at"
               and timedelta(0) <= utc(cutoff) - utc(quote["updated_at_utc"]) <= MAX_QUOTE_AGE
               for quote in quotes):
            verified[kind] = market
    return verified if "1x2" in verified and ({"over_2_5", "over_3_5"} & set(verified)) else {}


def _read_market(path, model, *, provider_root, cutoff, pair_sha256, earliest_kickoff):
    record = json.loads(path.read_text(encoding="utf-8"))
    if record.get("source_scope") != VERSION or not qualified_packet(record, model, now=cutoff):
        raise ValueError("invalid_targeted_market_source_identity")
    snapshot = record["provider_snapshot"]
    sha = snapshot["snapshot_sha256"]
    if path != _source_path(provider_root, model, sha):
        raise ValueError("targeted_market_path_mismatch")
    limits = snapshot["source_limits"]
    captured, fetched, deadline = map(utc, (snapshot["captured_at_utc"],
                                            limits["provider_fetched_at_utc"], earliest_kickoff))
    phase = phase_at(deadline, captured)
    if (not phase or phase != limits["phase"]
            or phase_at(deadline, fetched) != phase
            or not timedelta(0) <= captured - fetched <= timedelta(seconds=MAX_CACHE_SECONDS)
            or captured > utc(cutoff)
            or utc(limits["earliest_kickoff_utc"]) != deadline
            or limits["pair_sha256"] != pair_sha256):
        raise ValueError("invalid_targeted_market_cutoff_or_pair")
    markets = _market_coverage(snapshot, cutoff)
    if not markets:
        raise ValueError("incomplete_or_stale_targeted_markets")
    reference = {"path": path.relative_to(provider_root).as_posix(), "snapshot_sha256": sha,
                 "captured_at_utc": snapshot["captured_at_utc"]}
    return {"reference": reference, "value": markets}


def load_market_reference(model, reference, *, provider_root=ROOT, cutoff, pair_sha256,
                          earliest_kickoff):
    """Revalidate the exact archived receipt named by a frozen trial."""
    provider_root = Path(provider_root)
    try:
        sha = reference["snapshot_sha256"]
        if not isinstance(sha, str) or not re.fullmatch(r"[a-f0-9]{64}", sha):
            raise ValueError("invalid_market_snapshot_digest")
        component = _read_market(_source_path(provider_root, model, sha), model,
                                 provider_root=provider_root, cutoff=cutoff, pair_sha256=pair_sha256,
                                 earliest_kickoff=earliest_kickoff)
        if component["reference"] != reference:
            raise ValueError("targeted_market_reference_mismatch")
        return component
    except BAD_RECORD as exc:
        raise ValueError("unverifiable_targeted_market_reference") from exc


def best_market(model, *, provider_root=ROOT, cutoff, pair_sha256, earliest_kickoff):
    """Select the newest eligible T−30 or T−15 receipt available at the cutoff."""
    provider_root = Path(provider_root)
    base = source_packet_path(provider_root, model)
    candidates = []
    for path in (provider_root / "versions").glob(f"{base.stem}_*.json"):
        try:
            candidates.append(_read_market(path, model, provider_root=provider_root, cutoff=cutoff,
                                           pair_sha256=pair_sha256, earliest_kickoff=earliest_kickoff))
        except BAD_RECORD:
            continue
    return max(candidates, key=lambda item: (utc(item["reference"]["captured_at_utc"]),
                                             item["reference"]["snapshot_sha256"]), default=None)


def _fixture_rows(raw, model):
    if not isinstance(raw, list):
        raise CaptureRejected("malformed_odds_response")
    wanted = str(model["fixture_id"])
    matched = []
    for row in raw:
        if not isinstance(row, dict) or not isinstance(row.get("fixture"), dict):
            continue
        fixture = row["fixture"]
        if _positive_id(fixture.get("id")) != wanted:
            continue
        if utc(fixture["date"]) != utc(model["kickoff_utc"]):
            raise CaptureRejected("odds_fixture_kickoff_mismatch")
        league = row.get("league") or {}
        if league.get("id") is not None and str(league["id"]) != str(LEAGUES[model["competition"]]):
            raise CaptureRejected("odds_fixture_league_mismatch")
        teams = row.get("teams") or {}
        for side in ("home", "away"):
            team = teams.get(side) or {}
            if team.get("id") is not None and str(team["id"]) != str(model[f"{side}_team_id"]):
                raise CaptureRejected("odds_fixture_team_mismatch")
        matched.append(row)
    if len(matched) != 1:
        raise CaptureRejected("missing_or_ambiguous_odds_fixture")
    return matched


def _capture_packet(response, model, *, observed, kickoff, phase, pair_sha256):
    meta = response["metadata"]
    fetched = utc(meta["fetched_at_utc"])
    if (meta.get("provider") != "API-Football" or meta.get("stale") is not False
            or not timedelta(0) <= observed - fetched <= timedelta(seconds=MAX_CACHE_SECONDS)):
        raise CaptureRejected("stale_or_unverified_odds_response")
    if phase_at(kickoff, fetched) != phase or phase_at(kickoff, observed) != phase:
        raise CaptureRejected("odds_response_outside_collection_phase")
    markets = parse_markets(_fixture_rows(response["data"], model), observed_at=observed,
                            max_age_hours=MAX_QUOTE_AGE.total_seconds() / 3600)
    snapshot = {"schema": 1, "status": "captured_pre_kickoff", "provider": model["provider"],
                "fixture_id": str(model["fixture_id"]), "home_team_id": str(model["home_team_id"]),
                "away_team_id": str(model["away_team_id"]), "kickoff_utc": model["kickoff_utc"],
                "captured_at_utc": observed.isoformat(),
                "multi_market": {kind: item for kind, item in markets.items() if kind in MARKETS},
                "source_limits": {"phase": phase, "pair_sha256": pair_sha256,
                                  "earliest_kickoff_utc": kickoff.isoformat(),
                                  "provider_fetched_at_utc": fetched.isoformat()}}
    if not _market_coverage(snapshot, observed):
        raise CaptureRejected("fresh_timestamped_winner_and_totals_odds_unavailable")
    snapshot["snapshot_sha256"] = digest(snapshot)
    record = {"schema": 1, "source_forecast": {field: model[field] for field in MODEL_IDENTITY},
              "provider_snapshot": snapshot, "source_scope": VERSION}
    if not qualified_packet(record, model, now=observed):
        raise CaptureRejected("odds_packet_identity_validation_failed")
    return record


def _supported(model):
    return (model["provider"] == "API-Football" and _positive_id(model["fixture_id"])
            and model["competition"] in LEAGUES)


def _candidate_pairs(pair_root, source_root, now, rejections):
    candidates = []
    days = {moment.astimezone(SYDNEY).date().isoformat() for moment in (now, now + MAX_LEAD)}
    for day in sorted(days):
        path = Path(pair_root) / "sets" / (day + ".json")
        if not path.is_file():
            continue
        try:
            pair, models = _pair(path, source_root=source_root, decision=now)
            kickoff = min(utc(model["kickoff_utc"]) for model in models)
            phase = phase_at(kickoff, now)
            if not phase:
                continue
            if not all(map(_supported, models)):
                rejections["unsupported_market_provider_or_competition"] += 1
                continue
            candidates.append((kickoff, pair["sydney_date"], phase, pair["record_sha256"], models))
        except BAD_RECORD:
            rejections["unverifiable_original_pair"] += 1
    return sorted(candidates, key=lambda candidate: candidate[:2])


def capture(*, now=None, clock=None, enabled=None, api_key=None, environ=None, source_root=SOURCE_ROOT,
            pair_root=PAIR_ROOT, root=ROOT, broad_root=BROAD_ROOT, request=None):
    """Attempt at most one pair per run without exceeding the shared odds quota."""
    start, observed = capture_clock(now, clock)
    env = os.environ if environ is None else environ
    active = _enabled(env, enabled)
    key = api_key if api_key is not None else env.get("API_FOOTBALL_KEY")
    daily_limit = _daily_limit(env)
    root = Path(root)
    rejections = Counter()
    report = {"schema": VERSION, "checked_at_utc": start.isoformat(), "enabled": active,
              "shared_daily_limit": daily_limit, "requests": 0, "receipts": [],
              "bookmaker_counts": {}, "eligible_pairs": 0}

    def finish(reason):
        used, targeted = _usage(broad_root, observed())
        report.update(reason=reason, shared_requests_used=used, targeted_requests_used=targeted,
                      rejections=dict(rejections))
        status_path = root / "status.json"
        previous = load_json(status_path, {})
        if isinstance(previous, dict):
            previous.pop("checked_at_utc", None)
        if previous != {key: value for key, value in report.items() if key != "checked_at_utc"}:
            write_json(status_path, report)
        return report

    if not active:
        return finish("collection_disabled")
    if not key:
        return finish("provider_credentials_missing")
    candidates = _candidate_pairs(pair_root, source_root, start, rejections)
    report["eligible_pairs"] = len(candidates)
    for kickoff, day, phase, pair_sha256, models in candidates:
        pending = []
        for model in models:
            name = f"{MARKER_PREFIX}{pair_sha256}_{phase}_{model['fixture_id']}.json"
            if any((_request_folder(broad_root, date) / name).exists()
                   for date in (start.date(), (start - MAX_LEAD).date())):
                continue
            pending.append((model, name))
        if not pending:
            continue
        report.update(sydney_date=day, phase=phase, earliest_kickoff_utc=kickoff.isoformat())
        used, targeted = _usage(broad_root, start)
        if min(daily_limit - used, TARGETED_DAILY_REQUESTS - targeted) < len(pending):
            return finish("daily_request_budget_exhausted")
        for model, name in pending:
            stamp = observed()
            if phase_at(kickoff, stamp) != phase:
                return finish("collection_window_closed")
            used, targeted = _usage(broad_root, stamp)
            if used >= daily_limit or targeted >= TARGETED_DAILY_REQUESTS:
                return finish("daily_request_budget_exhausted")
            marker = _request_folder(broad_root, stamp.date()) / name
            if not _once(marker, {"phase": phase, "pair_sha256": pair_sha256,
                                  "fixture_id": str(model["fixture_id"]),
                                  "reserved_at_utc": stamp.isoformat()}):
                continue
            report["requests"] += 1
            try:
                phase_open = kickoff - (LATE_LEAD if phase == "late" else MAX_LEAD)
                cache_seconds = min(MAX_CACHE_SECONDS, (stamp - phase_open).total_seconds())
                response = (request or _request)(key, "/odds", {"fixture": int(model["fixture_id"])},
                                              with_metadata=True, max_cache_age_seconds=cache_seconds)
                record = _capture_packet(response, model, observed=observed(), kickoff=kickoff,
                                         phase=phase, pair_sha256=pair_sha256)
                sha = record["provider_snapshot"]["snapshot_sha256"]
                path = _source_path(root, model, sha)
                _once(path, record)
                report["receipts"].append(path.relative_to(root).as_posix())
                report["bookmaker_counts"][str(model["fixture_id"])] = {
                    kind: data["n_bookmakers"] for kind, data in record["provider_snapshot"]["multi_market"].items()}
            except (ProviderAccessError, ProviderQuotaError) as exc:
                rejections[type(exc).__name__] += 1
                return finish("provider_access_or_quota_limited")
            except CaptureRejected as exc:
                rejections[str(exc)] += 1
            except (RuntimeError, ValueError, KeyError, TypeError, AttributeError, ArithmeticError) as exc:
                rejections[type(exc).__name__] += 1
        return finish("market_capture_completed")
    return finish("already_attempted" if candidates else "no_pair_in_collection_window")


if __name__ == "__main__":
    print(json.dumps(capture(), indent=2, sort_keys=True))
