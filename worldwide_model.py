"""Verified API-Football Pro history for worldwide fixture forecasting.

Only completed fixtures are modelled. League scoring and opponent ratings use
the complete target competition, separately from team-history fallbacks.
"""
from datetime import datetime, timedelta, timezone
from api_football import _request, _convert
from engine import finished
from forecast_model import predict


def model_worldwide(key, event, model_variant='baseline'):
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
        if kickoff.tzinfo is None:
            kickoff = kickoff.replace(tzinfo=timezone.utc)
    except (KeyError, ValueError, TypeError):
        return None, "Fixture kickoff date is missing or invalid."
    observed_at = datetime.now(timezone.utc)
    cutoff = min(kickoff, observed_at)
    if cutoff <= datetime(2020, 1, 1, tzinfo=timezone.utc):
        return None, "Fixture date is outside the supported research window."
    competition = dict(fixture.get("competition") or {})
    season = event.get("season") or competition.get("season") or fixture.get("season")
    league = event.get("league_id") or competition.get("id")
    # Keep the fixture and its converted histories in the same provider-local
    # competition namespace. Never equate team or league IDs across providers.
    fixture = dict(fixture)
    if league is not None:
        competition["id"] = league
    if season is not None:
        competition["season"] = season
    fixture["competition"] = competition
    past = {}
    league_past = {}
    errors = []
    sources = []

    def same_competition(match):
        comp = match.get("competition") or {}
        if league is not None:
            return comp.get("id") is not None and str(comp["id"]) == str(league)
        code = competition.get("code")
        return bool(code and comp.get("code") == code)

    def collect(label, params, league_population=False):
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
                if date.tzinfo is None:
                    date = date.replace(tzinfo=timezone.utc)
                if date >= cutoff:
                    continue
                ids = ((m.get("homeTeam") or {}).get("id"), (m.get("awayTeam") or {}).get("id"))
                if ids[0] is None or ids[1] is None or ids[0] == ids[1]:
                    continue
                if league_population:
                    comp = m.get("competition") or {}
                    if not same_competition(m) or str(comp.get("season")) != str(params["season"]):
                        continue
                elif home not in ids and away not in ids:
                    continue
                identifier = m.get("id") or (m.get("utcDate"), *ids)
                if league_population:
                    league_past[identifier] = m
                if identifier not in past:
                    past[identifier] = m
                    added += 1
            sources.append(f"{label} ({added} additional results)")
        except Exception as exc:
            errors.append(f"{label}: {str(exc)[:125]}")

    def count(tid):
        return sum(same_competition(m) and tid in ((m.get("homeTeam") or {}).get("id"),
                           (m.get("awayTeam") or {}).get("id")) for m in past.values())

    # One cached request per league/season is shared by every fixture in that
    # competition. Do not filter this population down to the two target teams:
    # other clubs supply the league scoring environment and opponent ratings.
    if league is not None and season is not None:
        collect("Complete league season", {"league": league, "season": season},
                league_population=True)
        try:
            previous_season = int(season) - 1
        except (TypeError, ValueError):
            previous_season = 0
        if previous_season >= 2020 and (len(league_past) < 12 or min(count(home), count(away)) < 5):
            collect("Complete previous league season", {"league": league, "season": previous_season},
                    league_population=True)

    # Fall back to team endpoints only for sparse histories. Their rows retain
    # their real competition IDs; cup results cannot masquerade as league games.
    if min(count(home), count(away)) < 10:
        for label, tid in (("Home last 30", home), ("Away last 30", away)):
            collect(label, {"team": tid, "last": 30})
        collect("Head-to-head last 10", {"h2h": f"{home}-{away}", "last": 10})

    if count(home) < 10 or count(away) < 10:
        end = (cutoff - timedelta(days=1)).date()
        start = end - timedelta(days=365)
        for label, tid in (("Home date-range", home), ("Away date-range", away)):
            if count(tid) >= 10:
                continue
            collect(label, {"team": tid, "from": start.isoformat(), "to": end.isoformat()})

    history = sorted(past.values(), key=lambda m: m.get("utcDate", ""), reverse=True)
    league_history = sorted(league_past.values(), key=lambda m: m.get("utcDate", ""), reverse=True)
    if len(league_history) < 12:
        errors.append("Fewer than 12 verified league-wide results are available; league scoring rates use generic priors. Team-only results are not a league-wide baseline.")
    result = predict(fixture, history, league_history=league_history, variant=model_variant)
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
                 "reason": (i.get("player") or {}).get("reason") or "Reason not provided"}
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
                "league_history": league_history,
                "research_sources": ["API-Football (verified completed results)"],
                "research_details": sources,
                "research_context": context,
                "research_warnings": errors,
                "history_counts": {"home": count(home), "away": count(away),
                                   "league": len(league_history)}}, None

    reason = (f"Verified completed matches: home team {count(home)}, "
              f"away team {count(away)}. At least 5 each are required.")
    if errors:
        reason += " Data access details: " + " | ".join(errors[:3])
    return None, reason

