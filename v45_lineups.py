"""Bounded pre-kickoff starting-XI collection and cross-provider checks."""
from datetime import timedelta
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory

import requests

from api_football import ProviderAccessError, ProviderQuotaError, _request
from source_capture_clock import capture_clock
from sportmonks_probability_research import API_ROOT, _get, _kickoff, _name
from sportmonks_xg_research import (
    RequestBudgetExhausted, _fixture_candidates, _query_pages, _safe_message_class,
)
from v10_prospective import ROOT as FORECAST_ROOT, read_latest, utc
from v44_early_history import _once, digest


ROOT = Path("data/v45_intelligence")
SIDES = ("home", "away")
WINDOW = timedelta(minutes=60)
MIN_LEAD = timedelta(minutes=5)
MAX_API_AGE = timedelta(minutes=20)
MAX_REQUESTS_PER_RUN = 6
MAX_DAILY_REQUESTS = 24
DEFAULT_DAILY_REQUESTS = 12
VERSION = "v45-lineups-v1"


class CaptureWindowClosed(RuntimeError):
    pass


def _id(value):
    raw = str(value)
    if raw.isascii() and raw.isdecimal() and int(raw) > 0:
        return str(int(raw))
    return None


def _event(model):
    if model["provider"] != "API-Football":
        raise ValueError("unsupported_forecast_provider")
    ids = [_id(model[key]) for key in ("fixture_id", "home_team_id", "away_team_id")]
    if not all(ids) or ids[1] == ids[2]:
        raise ValueError("invalid_fixture_identity")
    return [model["provider"], *ids, utc(model["kickoff_utc"]).isoformat()]


def _player(player_id, name):
    if not isinstance(name, str) or not _id(player_id) or not _name(name):
        return None
    return {"id": _id(player_id), "name": name.strip()}


def _complete(teams):
    players = [player for side in SIDES for player in teams[side]]
    return (all(len(teams[side]) == 11 for side in SIDES) and
            len({p["id"] for p in players}) == 22 and
            all(len({_name(p["name"]) for p in teams[side]}) == 11 for side in SIDES))


def _summary(event, fixture_id, team_ids, fetched, kickoff, teams, confirmed):
    return {"fixture_id": str(fixture_id), "kickoff_utc": kickoff.isoformat(),
            "observed_at_utc": fetched.isoformat(), "provider_confirmed": confirmed,
            "teams": {side: {"team_id": str(team_ids[side]), "starter_count": 11,
                      "starting_xi_sha256": digest([event, side, sorted(
                          _name(player["name"]) for player in teams[side])])}
                      for side in SIDES}}


def _comparison(feeds):
    sportmonks, api = feeds.get("Sportmonks"), feeds.get("API-Football")
    if sportmonks and api:
        matches = {side: sportmonks["teams"][side]["starting_xi_sha256"] ==
                   api["teams"][side]["starting_xi_sha256"] for side in SIDES}
        return ("verified" if all(matches.values()) else "conflict"), matches
    if sportmonks:
        return "confirmed", None
    return ("unconfirmed" if api else "missing"), None


def _sportmonks(forecast, event, lookup, call, observed):
    match, reason, _ = lookup(forecast, call)
    if match is None:
        return None, None, reason
    fid = str(match["fixture_id"])
    response = call(f"{API_ROOT}/fixtures/{fid}",
                    {"include": "participants;lineups;metadata"})
    fetched = observed()
    item = response.get("data")
    if not isinstance(item, dict):
        category = _safe_message_class(response.get("message"))
        return None, None, "fixture_response_missing" if category == "absent" else category
    candidates, conflicted = _fixture_candidates(forecast, [item])
    kickoff = _kickoff(item)
    if conflicted or candidates != [match] or kickoff is None or str(item.get("state_id")) != "1":
        return None, None, "fixture_identity_or_state_mismatch"
    if fetched >= min(kickoff, utc(event[-1])):
        return None, None, "capture_finished_after_kickoff"
    rows = item.get("metadata")
    if not isinstance(rows, list):
        return None, None, "confirmation_metadata_missing"
    confirmation = [row.get("values") for row in rows if isinstance(row, dict) and
                    str(row.get("type_id")) == "572"]
    if not confirmation or any(not isinstance(values, dict) or values.get("confirmed") is not True
                               for values in confirmation):
        return None, None, "lineups_not_confirmed"
    team_ids = {side: str(match[side+"_team_id"]) for side in SIDES}
    side_for_id = {tid: side for side, tid in team_ids.items()}
    teams = {side: [] for side in SIDES}
    lineups = item.get("lineups")
    if not isinstance(lineups, list):
        return None, None, "starting_xis_missing"
    for row in lineups:
        if not isinstance(row, dict) or str(row.get("type_id")) != "11":
            continue
        side = side_for_id.get(_id(row.get("team_id")))
        player = _player(row.get("player_id"), row.get("player_name"))
        if not side or not player or _id(row.get("fixture_id")) != fid:
            return None, None, "invalid_starting_xi_row"
        teams[side].append(player)
    if not _complete(teams):
        return None, None, "incomplete_or_duplicate_starting_xis"
    return _summary(event, fid, team_ids, fetched, kickoff, teams, True), teams, "confirmed"


def _api_response(envelope, now):
    if not isinstance(envelope, dict):
        return None, None
    meta = envelope.get("metadata") or {}
    if not isinstance(meta, dict):
        return None, None
    try:
        fetched = utc(meta["fetched_at_utc"])
    except (KeyError, TypeError, ValueError):
        return None, None
    if (meta.get("provider") != "API-Football" or meta.get("stale") is not False
            or not timedelta(0) <= now-fetched <= MAX_API_AGE):
        return None, None
    return envelope.get("data"), fetched


def _api_football(event, call, observed):
    fid, home, away, raw_kickoff = event[1:]
    kickoff = utc(raw_kickoff)
    fixtures, fetched = _api_response(call("/fixtures", {"id": fid}), observed())
    if not isinstance(fixtures, list) or len(fixtures) != 1 or not isinstance(fixtures[0], dict):
        return None, None, "fixture_response_missing_or_stale"
    item = fixtures[0]
    fixture, sides = item.get("fixture") or {}, item.get("teams") or {}
    try:
        if (_id(fixture.get("id")) != fid or utc(fixture.get("date")) != kickoff
                or (fixture.get("status") or {}).get("short") != "NS"
                or [_id((sides.get(side) or {}).get("id")) for side in SIDES] != [home, away]):
            return None, None, "fixture_identity_or_state_mismatch"
    except (AttributeError, TypeError, ValueError):
        return None, None, "fixture_identity_or_state_mismatch"
    # Single-ID fixture responses can contain the XI alongside fixture and
    # team identity. Reuse that reply and call the lineup endpoint only when
    # the licensed response omits the embedded detail field entirely.
    if "lineups" in item:
        rows = item["lineups"]
    else:
        rows, fetched = _api_response(call("/fixtures/lineups", {"fixture": fid}), observed())
    if not isinstance(rows, list) or fetched is None:
        return None, None, "lineup_response_missing_or_stale"
    if not kickoff-WINDOW <= fetched < kickoff:
        return None, None, "lineup_response_outside_window"
    teams = {side: [] for side in SIDES}
    side_for_id = {home: "home", away: "away"}
    for row in rows:
        if not isinstance(row, dict):
            return None, None, "invalid_starting_xi_row"
        team = row.get("team")
        side = side_for_id.get(_id(team.get("id"))) if isinstance(team, dict) else None
        entries = row.get("startXI")
        if not side or teams[side] or not isinstance(entries, list) or len(entries) != 11:
            return None, None, "incomplete_or_duplicate_starting_xis"
        for entry in entries:
            player = (entry.get("player") or {}) if isinstance(entry, dict) else {}
            clean = _player(player.get("id"), player.get("name")) if isinstance(player, dict) else None
            if not clean:
                return None, None, "invalid_starting_xi_row"
            teams[side].append(clean)
    if not _complete(teams):
        return None, None, "incomplete_or_duplicate_starting_xis"
    return _summary(event, fid, dict(zip(SIDES, (home, away))), fetched, kickoff, teams, False), teams, "reported"


def _is_sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def load(path, model, *, cutoff):
    """Verify event, timing, source IDs, shape and derived agreement on read."""
    path = Path(path)
    row = json.loads(path.read_text(encoding="utf-8"))
    event = _event(model)
    observed, kickoff, cutoff = map(utc, (row["observed_at_utc"], event[-1], cutoff))
    if (row["version"] != VERSION or row["event"] != event
            or path.stem != row["receipt_sha256"]
            or row["receipt_sha256"] != digest({k: v for k, v in row.items() if k != "receipt_sha256"})
            or not kickoff-WINDOW <= observed < kickoff or observed > cutoff):
        raise ValueError("invalid_lineup_receipt")
    feeds = row["sources"]
    if not isinstance(feeds, dict) or not feeds or set(feeds)-{"Sportmonks", "API-Football"}:
        raise ValueError("invalid_lineup_sources")
    for provider, feed in feeds.items():
        source_kickoff, fetched = map(utc, (feed["kickoff_utc"], feed["observed_at_utc"]))
        teams = feed["teams"]
        if (not kickoff-WINDOW <= fetched <= observed < source_kickoff
                or abs(source_kickoff-kickoff) > timedelta(minutes=5)
                or not _id(feed["fixture_id"]) or set(teams) != set(SIDES)
                or feed["provider_confirmed"] is not (provider == "Sportmonks")):
            raise ValueError("invalid_lineup_source_identity_or_timestamp")
        team_ids = [_id(teams[side]["team_id"]) for side in SIDES]
        if (not all(team_ids) or team_ids[0] == team_ids[1] or
                any(teams[side]["starter_count"] != 11 or
                    not _is_sha(teams[side]["starting_xi_sha256"]) for side in SIDES)):
            raise ValueError("invalid_lineup_starting_xi")
        if provider == "API-Football" and (source_kickoff != kickoff or
                [feed["fixture_id"], *team_ids] != event[1:4]):
            raise ValueError("invalid_api_football_fixture_identity")
    if _comparison(feeds) != (row["status"], row["agreement_by_side"]):
        raise ValueError("invalid_lineup_source_comparison")
    return row


def status_for(model, *, root=ROOT, cutoff):
    """Newest observed evidence, preserving a corroboration or conflict on ties."""
    try:
        folder = Path(root)/"lineups"/"receipts"/digest(_event(model))
    except (KeyError, TypeError, ValueError):
        return {"status": "missing", "reason": "unsupported_or_invalid_fixture"}
    valid = []
    for path in folder.glob("*.json"):
        try:
            valid.append(load(path, model, cutoff=cutoff))
        except (OSError, KeyError, TypeError, AttributeError, ValueError):
            continue
    if not valid:
        return {"status": "missing", "reason": "no_qualified_starting_xi_receipt"}
    valid.sort(key=lambda r: (utc(r["observed_at_utc"]), len(r["sources"]), r["receipt_sha256"]), reverse=True)
    row = valid[0]
    # A later single-feed repeat of the same XI does not erase a completed
    # cross-check. Stop at the first changed XI or another two-feed result.
    observed_sources = dict(row["sources"])
    if len(observed_sources) == 1:
        for prior in valid[1:]:
            if any(prior["sources"][provider]["teams"] != observed_sources[provider]["teams"]
                   for provider in prior["sources"].keys() & observed_sources.keys()):
                break
            if len(prior["sources"]) == 2:
                row = prior
                break
            observed_sources.update(prior["sources"])
    return {"status": row["status"], "observed_at_utc": row["observed_at_utc"],
            "minutes_before_kickoff": round((utc(row["event"][-1])-
                                              utc(row["observed_at_utc"])).total_seconds()/60, 2),
            "agreement_by_side": row["agreement_by_side"], "sources": row["sources"],
            "receipt_sha256": row["receipt_sha256"]}


def _private_archive(database_url, receipt, teams_by_source):
    """Keep the named observations in the existing authenticated database."""
    if not database_url:
        return "database_not_configured"
    payload = json.dumps({"receipt": receipt, "starting_xis": teams_by_source},
                         sort_keys=True, separators=(",", ":"), allow_nan=False)
    try:
        import psycopg
        with psycopg.connect(database_url, connect_timeout=7) as conn:
            with conn.cursor() as cursor:
                cursor.execute("""CREATE TABLE IF NOT EXISTS ziczac_v45_starting_xis (
                    receipt_sha256 TEXT PRIMARY KEY,
                    fixture_id TEXT NOT NULL,
                    observed_at_utc TIMESTAMPTZ NOT NULL,
                    payload_json JSONB NOT NULL
                )""")
                cursor.execute("""INSERT INTO ziczac_v45_starting_xis
                    (receipt_sha256, fixture_id, observed_at_utc, payload_json)
                    VALUES (%s, %s, %s, %s::jsonb) ON CONFLICT DO NOTHING""",
                    (receipt["receipt_sha256"], receipt["event"][1],
                     receipt["observed_at_utc"], payload))
                state = "saved" if cursor.rowcount else "original_preserved"
        return state
    except Exception:
        return "database_unavailable"


def _atomic_status(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as stream:
        temp = Path(stream.name)
        try:
            json.dump(report, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
            stream.close()
            os.replace(temp, path)
        finally:
            temp.unlink(missing_ok=True)


def _eligible(originals, now):
    events = {}
    for row in originals.values() if isinstance(originals, dict) else originals:
        try:
            if (row["model_key"] not in ("v8", "v8_near") or not row["forecast_id"] or
                    not row["input_digest"] or not all(_name(row[side+"_team"]) for side in SIDES)):
                continue
            event = _event(row)
            if not utc(row["captured_at_utc"]) <= now or not MIN_LEAD <= utc(event[-1])-now <= WINDOW:
                continue
            key = tuple(event)
            prior = events.get(key)
            if prior is None or utc(row["captured_at_utc"]) < utc(prior["captured_at_utc"]):
                events[key] = row
        except (KeyError, TypeError, ValueError):
            continue
    return sorted(events.items(), key=lambda item: (item[0][-1], int(item[0][1])))


def _capture_phase(lead):
    if lead <= timedelta(minutes=20):
        return "late"
    if lead <= timedelta(minutes=40):
        return "announcement"
    return "early"


def _provider_check(func):
    try:
        return func()
    except RequestBudgetExhausted:
        return None, None, "request_budget_exhausted"
    except CaptureWindowClosed:
        return None, None, "capture_window_closed"
    except ProviderQuotaError:
        return None, None, "rate_limited"
    except ProviderAccessError:
        return None, None, "authentication_or_plan_access"
    except requests.HTTPError as exc:
        status = getattr(exc.response, "status_code", None)
        return None, None, {401: "authentication_failed", 403: "subscription_or_permission_denied",
                            404: "fixture_or_endpoint_unavailable", 429: "rate_limited"}.get(
                                status, "provider_http_error")
    except requests.RequestException:
        return None, None, "provider_network_error"
    except (AttributeError, KeyError, OverflowError, ValueError, TypeError, RuntimeError):
        return None, None, "provider_response_unavailable"


def collect(*, root=ROOT, source=FORECAST_ROOT, now=None, clock=None, environ=None,
            originals=None, sportmonks_request=None, api_football_request=None):
    """Collect one eligible frozen fixture with bounded requests and retries.

    Public receipts contain only verified structural metadata and one-way XI
    digests. Named observations are saved privately when DATABASE_URL is set.
    """
    started, observed = capture_clock(now, clock)
    env = os.environ if environ is None else environ
    root = Path(root)
    lineups = root/"lineups"
    enabled = str(env.get("V45_LINEUPS_ENABLED", "")).lower() == "true"
    sportmonks_key = env.get("THIRD_FOOTBALL_API_KEY") or env.get("SPORTMONKS_API_TOKEN")
    api_key = env.get("API_FOOTBALL_KEY")
    try:
        daily_limit = int(env.get("V45_LINEUP_DAILY_REQUESTS") or DEFAULT_DAILY_REQUESTS)
    except (TypeError, ValueError):
        daily_limit = DEFAULT_DAILY_REQUESTS
    daily_limit = max(0, min(MAX_DAILY_REQUESTS, daily_limit))
    report = {"schema": VERSION, "checked_at_utc": started.isoformat(), "enabled": enabled,
              "daily_request_limit": daily_limit, "requests_this_run": 0, "eligible_fixtures": 0,
              "providers": {"Sportmonks": "not_attempted" if sportmonks_key else "token_missing",
                            "API-Football": "not_attempted" if api_key else "key_missing"},
              "saved_receipt": False, "private_archive": "not_attempted"}

    def finish(reason):
        report["reason"] = reason
        report["daily_requests_used"] = len(list((lineups/"requests"/started.date().isoformat()).glob("*.json")))
        status = lineups/"status.json"
        if report["requests_this_run"]:
            report["last_provider_observation"] = {key: report[key] for key in (
                "checked_at_utc", "providers", "reason", "fixture_id", "private_archive")}
        else:
            try:
                report["last_provider_observation"] = json.loads(
                    status.read_text(encoding="utf-8")).get("last_provider_observation")
            except (OSError, ValueError, AttributeError):
                report["last_provider_observation"] = None
        _atomic_status(status, report)
        return report

    if not enabled:
        return finish("disabled")
    if not (sportmonks_key or api_key):
        return finish("provider_credentials_missing")
    if len(list((lineups/"requests"/started.date().isoformat()).glob("*.json"))) >= daily_limit:
        return finish("daily_request_budget_exhausted")
    candidates = _eligible(originals if originals is not None else read_latest(source), started)
    report["eligible_fixtures"] = len(candidates)
    if not candidates:
        return finish("no_frozen_fixtures_in_lineup_window")

    for identity, forecast in candidates:
        event = list(identity)
        kickoff = utc(event[-1])
        key = digest(event)
        phase = _capture_phase(kickoff-started)
        if phase != "late" and status_for(forecast, root=root, cutoff=started)["status"] == "verified":
            continue
        if _once(lineups/"attempts"/key/(phase+".json"),
                 {"event": event, "attempted_at_utc": started.isoformat()}):
            break
    else:
        return finish("eligible_fixtures_already_checked_in_phase")

    report.update(fixture_id=event[1], kickoff_utc=event[-1], capture_phase=phase)

    def reserve(provider, endpoint):
        stamp = observed()
        if stamp >= kickoff:
            raise CaptureWindowClosed
        if report["requests_this_run"] >= MAX_REQUESTS_PER_RUN:
            raise RequestBudgetExhausted
        folder = lineups/"requests"/stamp.date().isoformat()
        for slot in range(daily_limit):
            if _once(folder/(f"{slot:02}.json"), {"reserved_at_utc": stamp.isoformat(),
                    "event_sha256": key, "provider": provider, "endpoint": endpoint}):
                report["requests_this_run"] += 1
                return
        raise RequestBudgetExhausted

    def call_api(endpoint, params):
        reserve("API-Football", endpoint)
        return (api_football_request or _request)(api_key, endpoint, params, with_metadata=True)

    def call_sportmonks(url, params):
        category = "fixture_discovery" if "/date/" in url else "fixture_lineups"
        reserve("Sportmonks", category)
        return _get(sportmonks_key, url, params=params, request=sportmonks_request)

    sources, private = {}, {}
    # API-Football uses at most two requests. Retain its reported XI even if
    # Sportmonks discovery spends the remaining run budget on pagination.
    if api_key:
        result, teams, reason = _provider_check(lambda: _api_football(event, call_api, observed))
        report["providers"]["API-Football"] = reason
        if result:
            sources["API-Football"], private["API-Football"] = result, teams
    if sportmonks_key:
        with TemporaryDirectory(prefix="ziczac-v45-discovery-") as work:
            def lookup(model, call):
                return _query_pages(model, started, Path(work), call)

            result, teams, reason = _provider_check(
                lambda: _sportmonks(forecast, event, lookup, call_sportmonks, observed))
        report["providers"]["Sportmonks"] = reason
        if result:
            sources["Sportmonks"], private["Sportmonks"] = result, teams
    finished = observed()
    qualified = {provider: result for provider, result in sources.items()
                 if finished < utc(result["kickoff_utc"])}
    if finished >= kickoff:
        return finish("collection_finished_after_kickoff")
    if not qualified:
        return finish("no_qualified_starting_xis")

    state, agreement = _comparison(qualified)
    receipt = {"version": VERSION, "event": event, "observed_at_utc": finished.isoformat(),
               "status": state, "agreement_by_side": agreement, "sources": qualified}
    receipt["receipt_sha256"] = digest(receipt)
    report["saved_receipt"] = _once(
        lineups/"receipts"/key/(receipt["receipt_sha256"]+".json"), receipt)
    report["private_archive"] = _private_archive(
        env.get("DATABASE_URL"), receipt, {p: private[p] for p in qualified})
    report["receipt_sha256"] = receipt["receipt_sha256"]
    return finish(state)


if __name__ == "__main__":
    print(json.dumps(collect(), indent=2, sort_keys=True))
