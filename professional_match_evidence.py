"""One source-qualified pre-match dossier for the UI and every AI reviewer.

Pure, no-network descriptions only. Unknown fields stay unknown; model goal
means are never relabelled observed xG or player-impact coefficients.
"""
from collections import Counter
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math

from v21_provider_evidence import valid_snapshot

VERSION = "professional-match-evidence-v1"
WINDOWS = (5, 10, 20)


def _utc(value):
    try:
        at = value if isinstance(value, datetime) else datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return at.astimezone(timezone.utc) if at.tzinfo else None
    except (ValueError, TypeError, OverflowError):
        return None


def _id(value):
    return str(value) if value is not None and type(value) is not bool else ""


def _num(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def _feature(value=None, *, source=None, observed=None, reason=None, partial=False):
    available = value is not None
    return {"status": "partial" if available and partial else "available" if available else "missing",
            "available": available, "source": source,
            "source_observed_at_utc": observed,
            "value": value, "reason": reason,
            "numerical_weight_applied": False}


def validated_history(fixture, history, *, as_of):
    """Provider/team/time-qualified regulation results, conflicts excluded.

    An estimated three-hour availability lag is explicit, and is NOT called
    the provider observation time. Unknown fetch timestamps remain unknown.
    """
    clock, kickoff = _utc(as_of), _utc(fixture.get("utcDate"))
    if not clock or not kickoff or clock >= kickoff:
        raise ValueError("A timezone-aware pre-kickoff evidence cutoff is required")
    provider = str(fixture.get("provider") or "football-data.org")
    if not _id(fixture.get("id")):
        raise ValueError("A source fixture identity is required")
    comp = fixture.get("competition") or {}
    target = {_id((fixture.get(side + "Team") or {}).get("id")) for side in ("home", "away")}
    if "" in target or len(target) != 2:
        raise ValueError("Two distinct source team identities are required")
    groups = {}; rejected = Counter()
    for m in history or []:
        try:
            if not isinstance(m, dict):
                raise ValueError("invalid_row")
            if str(m.get("provider") or "football-data.org") != provider:
                raise ValueError("different_provider")
            other = m.get("competition") or {}
            if comp.get("id") is not None and str(other.get("id")) != str(comp["id"]):
                raise ValueError("different_competition")
            if comp.get("id") is None and comp.get("code") and other.get("code") != comp["code"]:
                raise ValueError("different_competition")
            if m.get("status") != "FINISHED":
                raise ValueError("unfinished")
            at = _utc(m.get("utcDate"))
            h, a = _id((m.get("homeTeam") or {}).get("id")), _id((m.get("awayTeam") or {}).get("id"))
            if not h or not a or h == a or not target.intersection((h, a)):
                raise ValueError("unrelated_or_invalid_teams")
            if not at or at >= clock or at < clock-timedelta(days=730):
                raise ValueError("outside_history_cutoff")
            if m.get("id") is not None and _id(m["id"]) == _id(fixture.get("id")):
                raise ValueError("target_fixture")
            score = m.get("score") or {}
            duration = score.get("duration")
            if duration not in (None, "", "REGULAR", "EXTRA_TIME", "PENALTY_SHOOTOUT"):
                raise ValueError("unknown_score_duration")
            goals = (score.get("regularTime") if duration in ("EXTRA_TIME", "PENALTY_SHOOTOUT") else score.get("fullTime")) or {}
            gh, ga = goals.get("home"), goals.get("away")
            if any(type(v) is not int or v < 0 for v in (gh, ga)) or m.get("score_basis", "regulation") != "regulation":
                raise ValueError("invalid_regulation_score")
            explicit_end = _utc(m.get("finishedAt"))
            if m.get("finishedAt") and explicit_end is None:
                raise ValueError("invalid_result_timestamp")
            available = explicit_end or at+timedelta(hours=3)
            if available < at or available > clock:
                raise ValueError("result_not_available_as_of")
            raw_observed = m.get("observed_at_utc") or m.get("fetched_at_utc")
            observed = _utc(raw_observed)
            if raw_observed and observed is None:
                raise ValueError("invalid_source_observation_timestamp")
            if observed and observed > clock:
                raise ValueError("source_observed_after_cutoff")
            row = {"provider": provider, "fixture_id": _id(m.get("id")) or None,
                   "kickoff_utc": at.isoformat(), "home_team_id": h, "away_team_id": a,
                   "home": str((m.get("homeTeam") or {}).get("name") or "")[:100],
                   "away": str((m.get("awayTeam") or {}).get("name") or "")[:100],
                   "regulation_score": [gh, ga],
                   "result_available_at_utc": available.isoformat(),
                   "result_availability_basis": "reported_finishedAt" if explicit_end else "conservative_kickoff_plus_3h",
                   "source_observed_at_utc": observed.isoformat() if observed else None}
            key = row["fixture_id"] or (at.isoformat(), h, a)
            groups.setdefault(str(key), []).append(row)
        except (ValueError, KeyError, TypeError, OverflowError) as exc:
            rejected[str(exc)[:80]] += 1
    rows = []
    for values in groups.values():
        signatures = {(v["kickoff_utc"], v["home_team_id"], v["away_team_id"], tuple(v["regulation_score"])) for v in values}
        if len(signatures) != 1:
            rejected["conflicting_duplicate"] += len(values)
            continue
        rows.append(min(values, key=lambda v: json.dumps(v, sort_keys=True)))
        rejected["duplicate"] += len(values)-1
    return sorted(rows, key=lambda r: (r["kickoff_utc"], r["fixture_id"] or ""), reverse=True), dict(rejected)


def _summary(rows, tid, window):
    samples = rows[:window]
    if not samples:
        return {"requested_matches": window, "matches": 0, "complete_window": False}
    scores = [(r["regulation_score"] if r["home_team_id"] == tid else r["regulation_score"][::-1]) for r in samples]
    n = len(samples)
    mean = lambda fn: round(sum(fn(gf, ga) for gf, ga in scores)/n, 4)
    return {"requested_matches": window, "matches": n, "complete_window": n == window,
            "goals_for_per_match": mean(lambda gf, ga: gf),
            "goals_against_per_match": mean(lambda gf, ga: ga),
            "scored_2plus_rate": mean(lambda gf, ga: gf >= 2),
            "conceded_2plus_rate": mean(lambda gf, ga: ga >= 2),
            "clean_sheet_rate": mean(lambda gf, ga: ga == 0),
            "btts_rate": mean(lambda gf, ga: gf > 0 and ga > 0),
            "over_2_5_rate": mean(lambda gf, ga: gf+ga >= 3),
            "last_scores_for_against": scores,
            "source_fixture_ids": [r["fixture_id"] for r in samples]}


def _windows(rows, tid):
    return {"last"+str(n): _summary(rows, tid, n) for n in WINDOWS}


def _xg(packet, rows, ids, clock):
    """Only observed xG on identity-matched, already available past fixtures."""
    lookup = {r["fixture_id"]: r for r in rows if r["fixture_id"]}
    found = {}; conflicts = set()
    for raw in (packet or {}).get("verified_historical_fixture_xg") or []:
        if not isinstance(raw, dict):
            continue
        fid = _id(raw.get("fixture_id")); match = lookup.get(fid)
        at = _utc(raw.get("kickoff_utc"))
        # A valid first-party packet can contain genuine past-statistics
        # observations even when the caller has no archived final scores.
        if match is None and fid and at and at+timedelta(hours=3) <= clock:
            home, away = _id(raw.get("home_team_id")), _id(raw.get("away_team_id"))
            if home and away and home != away and set(ids.values()).intersection((home, away)):
                match = {"fixture_id": fid, "kickoff_utc": at.isoformat(),
                         "home_team_id": home, "away_team_id": away}
        if not match or at != _utc(match["kickoff_utc"]):
            continue
        if any(_id(raw.get(k)) != match[k] for k in ("home_team_id", "away_team_id")):
            continue
        if not all(_num(raw.get(k)) for k in ("home_xg", "away_xg")):
            continue
        signature = (raw["home_xg"], raw["away_xg"])
        if fid in found and found[fid][1] != signature:
            conflicts.add(fid)
        found[fid] = (match, signature)
    entries = [v for fid, v in found.items() if fid not in conflicts]
    if not entries:
        return None
    entries.sort(key=lambda v: v[0]["kickoff_utc"], reverse=True)
    output = {}
    for side, tid in ids.items():
        relevant = [(r, x if r["home_team_id"] == tid else x[::-1]) for r, x in entries if tid in (r["home_team_id"], r["away_team_id"])]
        output[side] = {}
        for n in WINDOWS:
            sample = relevant[:n]
            output[side]["last"+str(n)] = {"matches": len(sample), "requested_matches": n,
                "complete_window": len(sample) == n,
                "xg_for_per_match": round(sum(x[0] for _, x in sample)/len(sample), 4) if sample else None,
                "xg_against_per_match": round(sum(x[1] for _, x in sample)/len(sample), 4) if sample else None,
                "source_fixture_ids": [r["fixture_id"] for r, _ in sample]}
    return output


def _shots(packet, ids, clock, *, on_target=False):
    rows = {}; conflicts = set()
    prefix = "shots_on_target" if on_target else "shots"
    observations=list((packet or {}).get("verified_historical_fixture_shots") or [])
    # Preserve compatibility with the already archived, source-verified
    # statistics packets; do not request the same fixtures again for shots.
    for raw in (packet or {}).get("verified_historical_match_statistics") or []:
        if not isinstance(raw,dict):
            continue
        observations.append({**raw,
            "home_shots":(raw.get("home") or {}).get("total_shots"),
            "away_shots":(raw.get("away") or {}).get("total_shots"),
            "home_shots_on_target":(raw.get("home") or {}).get("shots_on_target"),
            "away_shots_on_target":(raw.get("away") or {}).get("shots_on_target")})
    for raw in observations:
        if not isinstance(raw, dict):
            continue
        at = _utc(raw.get("kickoff_utc"))
        fid = _id(raw.get("fixture_id"))
        home, away = _id(raw.get("home_team_id")), _id(raw.get("away_team_id"))
        if (not at or at+timedelta(hours=3) > clock or not fid or
                not home or not away or home == away or
                not set(ids.values()).intersection((home, away))):
            continue
        totals = [raw.get(side+"_shots") for side in ("home", "away")]
        values = [raw.get(side+"_"+prefix) for side in ("home", "away")]
        if any(type(v) is not int or v < 0 for v in totals+values):
            continue
        if on_target and any(v > total for v, total in zip(values, totals)):
            continue
        signature = (at.isoformat(), home, away, *values)
        if fid in rows and rows[fid] != signature:
            conflicts.add(fid)
        rows[fid] = signature
    entries = [v for fid, v in rows.items() if fid not in conflicts]
    entries.sort(reverse=True)
    if not entries:
        return None
    output = {}
    for side, tid in ids.items():
        relevant = [(v[3], v[4]) if v[1] == tid else (v[4], v[3])
                    for v in entries if tid in (v[1], v[2])]
        output[side] = {}
        for n in WINDOWS:
            sample = relevant[:n]
            output[side]["last"+str(n)] = {
                "requested_matches": n, "matches": len(sample),
                "complete_window": len(sample) == n,
                prefix+"_for_per_match": round(sum(v[0] for v in sample)/len(sample), 4) if sample else None,
                prefix+"_against_per_match": round(sum(v[1] for v in sample)/len(sample), 4) if sample else None}
    return output


def build_dossier(fixture, history, *, as_of, provider_packet=None, statistical=None, context=None):
    """Common facts for professional research. Makes zero external requests."""
    clock, kickoff = _utc(as_of), _utc(fixture.get("utcDate"))
    rows, exclusions = validated_history(fixture, history, as_of=as_of)
    provider = str(fixture.get("provider") or "football-data.org")
    ids = {s: _id((fixture.get(s+"Team") or {}).get("id")) for s in ("home", "away")}
    observed = [r["source_observed_at_utc"] for r in rows if r["source_observed_at_utc"]]
    history_observed = max(observed) if observed else None
    source = provider+" archived regulation results"
    features = {}; form = {}; venue = {}; rest = {}
    for side, tid in ids.items():
        games = [r for r in rows if tid in (r["home_team_id"], r["away_team_id"])]
        venue_games = [r for r in games if r[side+"_team_id"] == tid]
        form[side] = _windows(games, tid); venue[side] = _windows(venue_games, tid)
        last = games[0] if games else None
        rest[side] = {"latest_known_fixture_id": last["fixture_id"] if last else None,
                      "latest_known_kickoff_utc": last["kickoff_utc"] if last else None,
                      "days_since_latest_known_kickoff": round((kickoff-_utc(last["kickoff_utc"])).total_seconds()/86400, 3) if last else None,
                      "schedule_complete": False, "scope": "supplied provider/competition history only"}
    any_history = bool(rows)
    features["goal_form"] = _feature(form if any_history else None, source=source, observed=history_observed,
        partial=any(form[s]["last20"]["matches"] < 20 for s in ids), reason="Windows report actual counts; short samples are not padded.")
    features["venue_form"] = _feature(venue if any_history else None, source=source, observed=history_observed,
        partial=any(venue[s]["last20"]["matches"] < 20 for s in ids), reason="Home team at home; away team away. Available competition history only.")
    meetings = [r for r in rows if {r["home_team_id"], r["away_team_id"]} == set(ids.values())][:10]
    features["head_to_head"] = _feature(meetings or None, source=source, observed=history_observed,
        reason="Descriptive context; no fitted H2H adjustment.")
    features["rest_schedule"] = _feature(rest if any_history else None, source=source, observed=history_observed, partial=True,
        reason="Gap from latest known kickoff; incomplete competition history cannot prove actual team rest.")
    packet = provider_packet if valid_snapshot(provider_packet, fixture, now=clock) else None
    stamp = packet["captured_at_utc"] if packet else None
    packet_source = "API-Football verified pre-match packet" if packet else None
    xg = _xg(packet, rows, ids, clock)
    features["shot_xg_xga"] = _feature(xg, source=packet_source, observed=stamp,
        partial=bool(xg and any(xg[s]["last20"]["matches"] < 20 for s in ids)),
        reason="Genuine provider-reported historical Expected Goals only; not a future score or observed future-match xG.")
    lineups = []; injuries = []
    if packet:
        for r in packet.get("confirmed_starting_xi") or []:
            if isinstance(r, dict) and _id(r.get("team_id")) in ids.values() and len(r.get("starting_xi") or []) == 11:
                lineups.append({"team_id": _id(r["team_id"]), "starting_xi": [str(n)[:100] for n in r["starting_xi"]],
                                "formation": str(r.get("formation") or "")[:40] or None})
        for r in packet.get("reported_injuries") or []:
            if isinstance(r, dict) and _id(r.get("team_id")) in ids.values() and r.get("player"):
                injuries.append({"team_id": _id(r["team_id"]), "player": str(r["player"])[:100], "reason": str(r.get("reason") or "unreported")[:120]})
    features["starting_lineups"] = _feature(lineups or None, source=packet_source, observed=stamp,
        partial=len({r["team_id"] for r in lineups}) < 2,
        reason="Provider-reported XI; absence of a response does not establish the expected lineup.")
    formations = [{"team_id": r["team_id"], "formation": r["formation"]} for r in lineups if r["formation"]]
    features["formations"] = _feature(formations or None, source=packet_source, observed=stamp,
        partial=len({r["team_id"] for r in formations}) < 2, reason="Source-reported formation only; no tactical-strength coefficient.")
    features["injuries_absences"] = _feature(injuries or None, source=packet_source, observed=stamp,
        reason="Reported absences only; no reports does not confirm a fully fit squad.")
    suspensions = [r for r in injuries if "suspend" in r["reason"].lower() or "suspension" in r["reason"].lower()]
    features["suspensions"] = _feature(suspensions or None, source=packet_source, observed=stamp,
        reason="Only explicitly reported suspensions; absence is unknown.")
    markets = {}
    for key, r in ((packet or {}).get("multi_market") or {}).items():
        updated = _utc(r.get("last_source_update_utc")) if isinstance(r, dict) else None
        if updated and updated <= clock and updated < kickoff and clock-updated <= timedelta(hours=12):
            markets[key] = r
    features["markets_1x2"] = _feature(markets.get("1x2"), source=packet_source, observed=stamp,
        reason="Dated complete bookmaker market; estimate includes quoted source timestamps.")
    other_markets = {k: v for k, v in markets.items() if k in ("over_2_5", "over_3_5", "btts")}
    features["markets_totals_btts"] = _feature(other_markets or None, source=packet_source, observed=stamp,
        reason="Only actually captured timestamped totals/BTTS prices.")
    for key in ("shots", "shots_on_target"):
        shot_data = _shots(packet, ids, clock, on_target=key == "shots_on_target")
        features[key] = _feature(shot_data, source=packet_source, observed=stamp,
            partial=bool(shot_data and any(shot_data[s]["last20"]["matches"] < 20 for s in ids)),
            reason="Actual historical source statistics from existing fixture-statistics responses; not estimated from xG.")
    for key, reason in {
        "goalkeeper": "Starting XI names do not establish goalkeeper identity, fitness, availability or shot-stopping quality.",
        "weather": "No venue-matched forecast observed before this evidence cutoff.",
        "travel": "No verified origin, destination or trip schedule; team names cannot establish travel distance.",
        "player_impact": "No independently validated player-absence or formation coefficients; no numeric adjustment applied."}.items():
        features[key] = _feature(reason=reason)
    model = statistical or {}
    model_means = {s: model.get(s+"_xg") for s in ids}
    result = {"version": VERSION, "as_of_utc": clock.isoformat(),
              "fixture": {"provider": provider, "fixture_id": _id(fixture.get("id")),
                          "kickoff_utc": kickoff.isoformat(), **{s+"_team_id": tid for s, tid in ids.items()}},
              "features": features, "historical_results": rows[:40], "history_exclusions": exclusions,
              "model_expected_goals_not_observed_shot_xg": {s: v if _num(v) else None for s, v in model_means.items()},
              "provider_snapshot_sha256": packet["snapshot_sha256"] if packet else None,
              "unvalidated_context_not_used": bool(context),
              "external_calls_made": 0, "ai_numeric_weight": 0.0,
              "original_forecasts_changed": False,
              "limitations": ["Missing features are unknown, never zero.",
                 "Source observation timestamps remain null where not archived.",
                 "Historical availability estimates are distinct from actual fetch times.",
                 "Descriptive features do not prove a prediction accuracy gain."]}
    body = json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False)
    result["dossier_sha256"] = hashlib.sha256(body.encode()).hexdigest()
    return result


def feature_rows(dossier):
    """Compact records for the common UI availability table."""
    return [{"Feature": key.replace("_", " ").title(), "Status": item["status"],
             "Source": item.get("source") or "No verified source",
             "Observed UTC": item.get("source_observed_at_utc") or "Unknown",
             "Details": item.get("reason") or ""}
            for key, item in (dossier.get("features") or {}).items()]
