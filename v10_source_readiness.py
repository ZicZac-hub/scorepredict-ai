"""Explain why authenticated pre-kickoff V8 evidence cannot issue a forecast.

This module is an evidence audit, not a relaxation of model minimums.
Use only the same already-collected, as-of filtered league results that V8
received; never query external providers or fill missing history.
"""
import engine
from research_ensemble_v8 import _asof_history


def team_history_band(qualified_matches):
    """A bounded, aggregate-friendly count, not permission to forecast."""
    if type(qualified_matches) is not int or qualified_matches < 0:
        return None
    if qualified_matches == 0:
        return "0"
    if qualified_matches <= 2:
        return "1-2"
    if qualified_matches <= 4:
        return "3-4"
    return "5+"


def history_readiness(fixture, history, league_history, as_of):
    """Summarize VERIFIED, as-of source coverage without new API requests.

    Team counts are computed from exactly the same same-competition evidence
    accepted by the original V8 code. Counts are observability only; no
    insufficient match is converted into a forecast.
    """
    report = {
        "reason": "unusable_authenticated_history_structure",
        "home_qualified_matches": None,
        "away_qualified_matches": None,
        "minimum_team_qualified_matches": None,
        "league_qualified_matches": None,
        "unrelated_qualified_matches": None,
        "minimum_missing_to_five": None,
    }
    try:
        home = (fixture.get("homeTeam") or {}).get("id")
        away = (fixture.get("awayTeam") or {}).get("id")
        kickoff = engine._date(fixture.get("utcDate"))
        cutoff = engine._date(as_of)
        if home is None or away is None or home == away or kickoff is None:
            report["reason"] = "invalid_source_fixture_identity"
            return report
        if cutoff is None or cutoff >= kickoff:
            report["reason"] = "not_before_original_kickoff"
            return report
        prior = _asof_history(fixture, history or [], cutoff)
        population = _asof_history(fixture, league_history or [], cutoff)
        home_count = len(engine._games(prior, home, limit=10))
        away_count = len(engine._games(prior, away, limit=10))
        minimum = min(home_count, away_count)
        report.update({
            "home_qualified_matches": home_count,
            "away_qualified_matches": away_count,
            "minimum_team_qualified_matches": minimum,
            "league_qualified_matches": len(population),
            "minimum_missing_to_five": max(0, 5 - minimum),
        })
        if minimum < 5:
            report["reason"] = "team_history_below_five"
            return report
        if len(population) < 35:
            report["reason"] = "league_history_below_thirty_five"
            return report
        unrelated = sum(
            home not in (m["homeTeam"]["id"], m["awayTeam"]["id"]) and
            away not in (m["homeTeam"]["id"], m["awayTeam"]["id"])
            for m in population
        )
        report["unrelated_qualified_matches"] = unrelated
        if unrelated < 12:
            report["reason"] = "unrelated_league_history_below_twelve"
            return report
        report["reason"] = "independent_strength_component_unqualified"
        return report
    except (KeyError, ValueError, TypeError, OverflowError, AttributeError):
        return report


def abstention_reason(fixture, history, league_history, as_of):
    """Preserve the historical allowlisted abstention API."""
    return history_readiness(fixture, history, league_history, as_of)["reason"]
