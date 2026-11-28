"""Probe Sportmonks fixture access and xG coverage with redacted diagnostics."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re

import requests

from sportmonks_probability_research import API_ROOT
from sportmonks_xg_research import EXPECTED_GOALS_TYPE, _safe_message_class, _xg_pair
from sportmonks_xg_validation import _xg_number
from v44_collection_health import read, utc


ROOT = Path("data")
FIXTURE_ID_ENV = "V45_SPORTMONKS_FIXTURE_ID"
HTTP_FAILURES = {
    401: "authentication_failed",
    403: "subscription_denied",
    404: "resource_not_found",
    429: "rate_limited",
}
STOP_REASONS = {
    "authentication_failed", "rate_limited", "provider_server_error", "network_error",
}
ARCHIVED_REASONS = {
    "disabled", "sportmonks_token_missing", "no_frozen_forecasts_within_48_hours",
    "eligible_forecasts_already_checked", "daily_request_budget_exhausted",
    "past_kickoff_not_valid_for_research", "invalid_historical_response",
    "collection_finished_after_kickoff", "genuine_xg_unavailable_subscription_or_coverage",
    "genuine_sportmonks_xg_captured", "request_budget_exhausted_mid_research",
    "provider_access_subscription_or_network_error", "source_error",
    "provider_access_or_subscription", "provider_quota_or_rate_limit",
    "provider_invalid_request", "unclassified_provider_message", "provider_reported_error",
    "discovery_request_budget_exhausted", "invalid_discovery_response",
    "invalid_cached_discovery", "discovery_pagination_incomplete",
    "ambiguous_cross_provider_fixture", "no_unique_cross_provider_fixture_match",
}


def _id(value):
    if type(value) is int:
        return value if value > 0 else None
    if isinstance(value, str) and re.fullmatch(r"[1-9][0-9]{0,19}", value):
        return int(value)
    return None


def _step(status, http_status=None):
    return {"status": status, "http_status": http_status}


def _read_response(get, token, path, params=None):
    try:
        response = get(
            f"{API_ROOT}{path}",
            headers={"Authorization": token, "Accept": "application/json"},
            params=params, timeout=20, allow_redirects=False,
        )
    except requests.RequestException:
        return _step("network_error"), None
    code = response.status_code
    if type(code) is not int or not 100 <= code <= 599:
        return _step("invalid_response"), None
    if code != 200:
        default = "provider_server_error" if code >= 500 else "unexpected_http_status"
        return _step(HTTP_FAILURES.get(code, default), code), None
    try:
        body = response.json()
    except ValueError:
        return _step("invalid_response", code), None
    if not isinstance(body, dict):
        return _step("invalid_response", code), None
    message_class = _safe_message_class(body.get("message"))
    if message_class == "provider_quota_or_rate_limit":
        return _step("rate_limited", code), None
    if message_class in {"provider_access_or_subscription", "provider_invalid_request"}:
        return _step(message_class, code), None
    if body.get("error") or body.get("errors"):
        return _step("provider_reported_error", code), None
    return _step("available", code), body.get("data")


def _fixture_xg_status(fixture):
    fields = (fixture.get("xgfixture"), fixture.get("expected"))
    if any(entries is not None and not isinstance(entries, list) for entries in fields):
        return "invalid_response"
    if not any(fields):
        return "xg_missing"
    try:
        if _xg_pair(fixture) is not None:
            return "available"
    except (AttributeError, TypeError, ValueError, OverflowError):
        return "invalid_response"
    return "xg_pair_unverified"


def _is_team_xg(row):
    if not isinstance(row, dict) or _id(row.get("type_id")) != EXPECTED_GOALS_TYPE:
        return False
    value = row.get("data")
    return bool(_id(row.get("fixture_id")) and _id(row.get("participant_id"))
                and row.get("location") in ("home", "away") and isinstance(value, dict)
                and _xg_number(value.get("value")))


def probe_sportmonks(*, fixture_id=None, environ=None, request=None, now=None):
    """Check one known fixture with up to four requests and no automatic retries.

    A recent completed fixture in a target league is the best coverage sample.
    Positive xG observations establish access to the observed data. Empty data
    leaves entitlement unresolved unless an independent route reports denial.
    """
    environ = os.environ if environ is None else environ
    requested = fixture_id if fixture_id is not None else environ.get(FIXTURE_ID_ENV)
    fixture_id = _id(requested)
    token = environ.get("THIRD_FOOTBALL_API_KEY") or environ.get("SPORTMONKS_API_TOKEN")
    report = {
        "report_type": "v45_sportmonks_provider_probe_v1",
        "checked_at_utc": utc(now or datetime.now(timezone.utc)).isoformat(),
        "status": "fixture_not_configured" if requested in (None, "") else "invalid_fixture_id",
        "fixture_id": fixture_id,
        "token_present": bool(token),
        "requests_this_run": 0,
        "league_access": _step("not_checked"),
        "fixture_access": _step("not_checked"),
        "fixture_xg": _step("not_checked"),
        "team_xg_feed": _step("not_checked"),
        "xg_access": "unconfirmed",
    }
    if fixture_id is None:
        return report
    if not token:
        report["status"] = "sportmonks_token_missing"
        return report

    def check(field, path, params=None):
        report["requests_this_run"] += 1
        step, data = _read_response(request or requests.get, token, path, params)
        report[field] = step
        return step, data

    leagues, data = check("league_access", "/leagues", {"per_page": 1})
    if leagues["status"] != "available":
        report["status"] = leagues["status"]
        return report
    if not isinstance(data, list) or any(not isinstance(league, dict) or not _id(league.get("id"))
                                         for league in data):
        leagues["status"] = report["status"] = "invalid_response"
        return report
    if not data:
        leagues["status"] = "no_leagues_returned"

    fixture_path = f"/fixtures/{fixture_id}"
    fixture, data = check("fixture_access", fixture_path)
    if fixture["status"] == "available":
        if isinstance(data, dict) and _id(data.get("id")) == fixture_id:
            fixture.update({field: _id(data.get(field))
                            for field in ("league_id", "season_id", "state_id")})
            xg, xg_data = check("fixture_xg", fixture_path, {"include": "participants;xGFixture"})
            if xg["status"] == "available":
                xg["status"] = (_fixture_xg_status(xg_data)
                                if isinstance(xg_data, dict) and _id(xg_data.get("id")) == fixture_id
                                else "invalid_response")
        else:
            fixture["status"] = "invalid_response"

    fixture_xg_status = report["fixture_xg"]["status"]
    if fixture_xg_status == "available":
        report["xg_access"] = "observed"
    elif not STOP_REASONS.intersection({fixture["status"], fixture_xg_status}):
        feed, data = check("team_xg_feed", "/expected/fixtures", {"per_page": 10})
        if feed["status"] == "available":
            if not isinstance(data, list):
                feed["status"] = "invalid_response"
            elif not any(_is_team_xg(row) for row in data):
                feed["status"] = "xg_not_observed"
        if feed["status"] == "available":
            report["xg_access"] = "observed"
        elif "subscription_denied" in (fixture_xg_status, feed["status"]):
            report["xg_access"] = "restricted"
    report["status"] = "checked"
    return report


def _archive_entry(item, counters):
    if not item:
        return None
    try:
        at = utc(item.get("at_utc")).isoformat()
    except (ValueError, TypeError, OverflowError):
        return None
    reason = item.get("reason")
    entry = {"at_utc": at, "reason": reason if isinstance(reason, str)
             and reason in ARCHIVED_REASONS else "unclassified"}
    for field in counters:
        value = item.get(field)
        entry[field] = value if type(value) is int and value >= 0 else None
    return entry


def archived_health(root=ROOT):
    """Read the previous collector outcome and last actual provider observation."""
    root = Path(root) / "sportmonks_xg"
    latest = _archive_entry(read(root / "status.json"),
                            ("requests_this_run", "daily_requests_used", "daily_request_limit"))
    observation = _archive_entry(read(root / "last_provider_observation.json"),
                                 ("requests_this_run", "matched_future_fixtures",
                                  "matched_historical_xg_fixtures", "captured"))
    return {"report_type": "v45_archived_sportmonks_health_v1", "latest_run": latest,
            "last_provider_observation": observation, "requests_this_run": 0}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture-id", help="Sportmonks ID of a recent completed fixture")
    arguments = parser.parse_args()
    print(json.dumps(probe_sportmonks(fixture_id=arguments.fixture_id), indent=2))
