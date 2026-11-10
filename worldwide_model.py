"""Verified API-Football Pro history for worldwide fixture forecasting.

Only completed fixtures are modelled. Uses team-specific recent form and H2H,
with date-range fallbacks for providers that do not support 'last'.
"""
from datetime import datetime, timedelta, timezone
from api_football import _request, _convert
from engine import predict, finished


def model_worldwide(key, event):
    fixture = event.get("fixture")
    if not fixture:
        return None, "Fixture has no compatible API-Football identifiers."
    if not key or fixture.get("provider") != "API-Football":
        return None, "API-Football key or compatible fixture identifiers are missing."
    home = (fixture.get("homeTeam") or {}).get("id")
    away = (fixture.get("awayTeam") or {}).get("id")
    if home is None or away is None or home == away:
        return None, "Both API-Football team IDs are required."
    try:
        kickoff = datetime.fromisoformat(fixture["utcDate"].replace("Z", "+00:00"))
    except (KeyError, ValueError, TypeError):
        return None, "Fixture kickoff date is missing or invalid."
    cutoff = min(kickoff, datetime.now(timezone.utc))
    if cutoff <= datetime(2020, 1, 1, tzinfo=timezone.utc):
        return None, "Fixture date is outside the supported research window."
    season = event.get("season")
    league = event.get("league_id")
    past = {}
    errors = []
    sources = []

    def collect(label, params):
        try:
            raw_matches = _request(key, "/fixtures", params)
            added = 0
            for raw in raw_matches:
                m = _convert(raw, "WORLD")
                if not finished(m):
                    continue
                try:
                    date = datetime.fromisoformat(m["utcDate"].replace("Z", "+00:00"))
                except (ValueError, TypeError, KeyError):
                    continue
                if date >= kickoff or date > datetime.now(timezone.utc):
                    continue
                ids = ((m.get("homeTeam") or {}).get("id"), (m.get("awayTeam") or {}).get("id"))
                if home not in ids and away not in ids:
                    continue
                identifier = m.get("id") or (m.get("utcDate"), *ids)
                if identifier not in past:
                    past[identifier] = m
                    added += 1
            sources.append(f"{label} ({added} additional results)")
        except Exception as exc:
            errors.append(f"{label}: {str(exc)[:125]}")

    # Pro plans may support 'last' and head-to-head; the free tier often does not.
    # Fetch BOTH teams before modelling, so the result is not biased by an
    # incomplete first successful response.
    for label, tid in (("Home last 30", home), ("Away last 30", away)):
        collect(label, {"team": tid, "last": 30})

    # Include actual H2H fixtures (up to 10), which may be older than form data.
    collect("Head-to-head last 10", {"h2h": f"{home}-{away}", "last": 10})

    # Fallback across two football seasons when last/H2H endpoints are restricted.
    def count(tid):
        return sum(tid in ((m.get("homeTeam") or {}).get("id"),
                           (m.get("awayTeam") or {}).get("id")) for m in past.values())

    if count(home) < 10 or count(away) < 10:
        end = (cutoff - timedelta(days=1)).date()
        start = end - timedelta(days=365)
        for label, tid in (("Home date-range", home), ("Away date-range", away)):
            if count(tid) >= 10:
                continue
            collect(label, {"team": tid, "from": start.isoformat(), "to": end.isoformat()})

    # Last fallback: league-wide season data can fill missing home/away splits.
    if (count(home) < 5 or count(away) < 5) and league and season:
        collect("League season", {"league": league, "season": season,
                                  "from": (cutoff - timedelta(days=365)).date().isoformat(),
                                  "to": (cutoff - timedelta(days=1)).date().isoformat()})

    history = sorted(past.values(), key=lambda m: m.get("utcDate", ""), reverse=True)
    result = predict(fixture, history)
    if result is not None:
        # Research extra independent evidence. Missing provider data is reported,
        # never converted into made-up injuries, formations or probabilities.
        context = {}
        context_errors = []
        def optional(label, endpoint, params):
            try:
                return _request(key, endpoint, params)
            except Exception as exc:
                context_errors.append(f"{label}: {str(exc)[:120]}")
                return []
        if league and season:
            for label, tid in (("home", home), ("away", away)):
                stats = optional(f"{label} season statistics", "/teams/statistics",
                                 {"league": league, "season": season, "team": tid})
                # API-Sports returns a dict for this endpoint, unlike fixture lists.
                if isinstance(stats, dict) and stats:
                    context[label+"_season"] = {
                        "form": stats.get("form"),
                        "fixtures": stats.get("fixtures"),
                        "goals": stats.get("goals"),
                        "lineups": (stats.get("lineups") or [])[:3],
                        "clean_sheet": stats.get("clean_sheet"),
                        "failed_to_score": stats.get("failed_to_score")}
            standings = optional("league standings", "/standings",
                                 {"league": league, "season": season})
            try:
                groups = standings[0]["league"]["standings"]
                table = [entry for group in groups for entry in group]
                context["standings"] = [
                    {"team": row.get("team",{}).get("name"),
                     "rank": row.get("rank"),
                     "points": row.get("points"),
                     "goal_difference": row.get("goalsDiff"),
                     "form": row.get("form")}
                    for row in table if row.get("team",{}).get("id") in (home,away)]
            except (IndexError,KeyError,TypeError):
                pass
        # Fixture injuries and pre-match lineups may be absent, especially for
        # lower divisions and fixtures many days ahead.
        injuries = optional("fixture injuries", "/injuries", {"fixture": fixture.get("id")})
        if isinstance(injuries, list):
            context["injuries"] = [
                {"team": (i.get("team") or {}).get("name"),
                 "player": (i.get("player") or {}).get("name"),
                 "reason": (i.get("player") or {}).get("reason") or
                           (i.get("fixture") or {}).get("status",{}).get("long")}
                for i in injuries[:30]]
        lineups = optional("fixture lineups", "/fixtures/lineups",
                           {"fixture": fixture.get("id")})
        if isinstance(lineups, list) and lineups:
            context["confirmed_lineups"] = [
                {"team": (i.get("team") or {}).get("name"),
                 "formation": i.get("formation"),
                 "coach": (i.get("coach") or {}).get("name")}
                for i in lineups]
        context["limitations"] = (
            "Team statistics, standings, injury reports and formations are provider "
            "context only; they are not yet fitted as numerical adjustments to the "
            "Poisson score model. Missing records do not mean no injuries.")
        sources.extend(["Season statistics, standings, injuries and lineups checked where available"])
        errors.extend(context_errors)
    if result is not None:
        return {"league": event.get("league", "Unknown"),
                "fixture": fixture, "result": result,
                "kickoff": fixture["utcDate"], "past": history,
                "research_sources": ["API-Football (verified completed results)"],
                "research_details": sources,
                "research_context": context,
                "research_warnings": errors,
                "history_counts": {"home": count(home), "away": count(away)}}, None

    reason = (f"Verified completed matches: home team {count(home)}, "
              f"away team {count(away)}. At least 5 each are required.")
    if errors:
        reason += " Data access details: " + " | ".join(errors[:3])
    return None, reason
