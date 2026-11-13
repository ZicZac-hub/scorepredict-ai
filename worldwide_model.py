"""Verified API-Football Pro history for worldwide fixture forecasting.

All numerical forecasts use the shared two-season, result-availability policy.
Optional team context remains separate from the model evidence.
"""
from datetime import datetime, timezone
from api_football import _request
from forecast_model import predict
from history_pipeline import load_evidence


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
    evidence=load_evidence(fixture, as_of=observed_at, api_key=key, api_request=_request)
    fixture=evidence['fixture']
    history=evidence['history']
    league_history=evidence['league_history']
    errors=list(evidence['warnings'])
    sources=[f"Shared {evidence['policy_version']} policy ({len(history)} available league results)"]

    def count(tid):
        return sum(tid in ((match.get('homeTeam') or {}).get('id'),
                          (match.get('awayTeam') or {}).get('id')) for match in history)

    if min(count(home),count(away)) < 5:
        reason=(f"Verified completed matches: home team {count(home)}, away team {count(away)}. "
                "At least 5 each are required under the shared league-history policy.")
        if errors:
            reason += " Data access details: " + " | ".join(errors[:3])
        return None, reason
    if len(league_history)<12:
        errors.append('Fewer than 12 verified league-wide results are available; league scoring rates use generic priors.')
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
                context_errors.append(f"{label} unavailable ({type(exc).__name__})")
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
                "league_history": league_history, "evidence": evidence,
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

