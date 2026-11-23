"""Opt-in, two-request source-checked market availability for V25 on-demand.

This is not an odds-weighted model and never changes any frozen score grid.
The API-Football fixture ID and provider kickoff must match the modelled leg.
Unverifiable/missing/stale odds are UNKNOWN, never treated as verified.
"""
from datetime import datetime, timezone, timedelta
from v10_prospective import utc
from v22_evidence_models import parse_markets

MAX_MARKET_AGE = timedelta(hours=12)
MAX_KICKOFF_DIFFERENCE = timedelta(minutes=5)


def inspect_pair(api_key, report, *, now=None, request=None):
    from api_football import _request
    clock = utc(now or datetime.now(timezone.utc))
    fixtures = [report.get("first_fixture"), report.get("second_fixture")]
    output = {"checked_at_utc":clock.isoformat(), "legs": [],
              "both_verified": False, "max_provider_requests":2,
              "does_not_change_v25_scores":True,
              "does_not_establish_actual_accuracy":True,
              "provider_requests_attempted":0}
    if not api_key or report.get("status") != "ready":
        output["reason"] = "missing_api_key_or_invalid_report"
        return output

    fetch = request or _request
    for fixture in fixtures:
        leg = {"fixture_id":str((fixture or {}).get("id") or ""),
               "verified_pre_match_market":False,
               "reason":"not_checked"}
        output["legs"].append(leg)
        try:
            if not isinstance(fixture,dict) or fixture.get("provider")!="API-Football":
                leg["reason"]="unsupported_provider"
                continue
            kickoff = utc(fixture["utcDate"])
            if clock >= kickoff - timedelta(minutes=5):
                leg["reason"]="too_close_to_kickoff"
                continue
            fid = int(fixture["id"])
            if fid <= 0 or any(
                not str((fixture.get(side+"Team") or {}).get("id") or "").strip()
                for side in ("home","away")
            ):
                leg["reason"]="invalid_fixture_identity"
                continue

            output["provider_requests_attempted"] += 1
            envelope = fetch(api_key,"/odds",{"fixture":fid},with_metadata=True)
            if not isinstance(envelope,dict):
                leg["reason"]="missing_fetch_metadata"
                continue
            metadata = envelope.get("metadata") or {}
            if metadata.get("stale") or not metadata.get("fetched_at_utc"):
                leg["reason"]="stale_or_unknown_response_time"
                continue
            fetched = utc(metadata["fetched_at_utc"])
            if fetched > clock or fetched >= kickoff or clock-fetched > MAX_MARKET_AGE:
                leg["reason"]="unverified_response_freshness"
                continue
            raw = envelope.get("data") or []
            matching=[]
            for item in raw if isinstance(raw,list) else []:
                if not isinstance(item,dict):
                    continue
                quote_fixture=item.get("fixture") or {}
                if str(quote_fixture.get("id"))!=str(fid):
                    continue
                # When the quote response lacks a kickoff, its identity is
                # incomplete; do not bind it to an unrelated scheduled match.
                if not quote_fixture.get("date"):
                    continue
                source_kickoff=utc(quote_fixture["date"])
                if abs(source_kickoff-kickoff)>MAX_KICKOFF_DIFFERENCE:
                    continue
                matching.append(item)
            if not matching:
                leg["reason"]="no_id_and_kickoff_matched_market"
                continue
            markets=parse_markets(matching,observed_at=clock,
                                  verified_provider_fetched_at=fetched)
            if "1x2" not in markets or not (
                "over_2_5" in markets or "over_3_5" in markets
            ):
                leg["reason"]="missing_fresh_1x2_and_totals_quotes"
                continue
            leg.update({
                "verified_pre_match_market":True,
                "reason":"verified_provider_fixture_id_kickoff_1x2_and_totals",
                "source_response_fetched_at_utc":fetched.isoformat(),
                "bookmaker_quote_types":sorted(markets),
                "source":"API-Football pre-match odds (on-demand spot check)",
                "used_for_v25_probability_calibration":False,
            })
        except (ValueError,TypeError,KeyError) as exc:
            leg["reason"]="invalid_provider_data_"+type(exc).__name__
        except Exception as exc:
            # Never print provider response bodies, keys or credentials.
            leg["reason"]="provider_request_failed_"+type(exc).__name__
            if type(exc).__name__ in ("ProviderQuotaError","ProviderAccessError"):
                output["provider_limited"]=True
                break
    output["both_verified"] = (
        len(output["legs"])==2
        and all(x["verified_pre_match_market"] for x in output["legs"])
    )
    return output
