"""Explain why authenticated pre-kickoff V8 evidence cannot issue a forecast.

This module is an evidence audit, not a relaxation of model minimums.
Use only the same already-collected, as-of filtered league results that V8
received; never query external providers or fill missing history.
"""
import engine
from research_ensemble_v8 import _asof_history


def abstention_reason(fixture, history, league_history, as_of):
    """Bounded allowlisted reason, with no fixture identity or secret values.

    Checks are ordered like the existing V8 model: enough team evidence for
    the baseline, then enough league and unrelated matches for independent
    strength estimates. A qualifying count alone does not imply fit success.
    """
    try:
        home = (fixture.get("homeTeam") or {}).get("id")
        away = (fixture.get("awayTeam") or {}).get("id")
        kickoff = engine._date(fixture.get("utcDate"))
        cutoff = engine._date(as_of)
        if home is None or away is None or home == away or kickoff is None:
            return "invalid_source_fixture_identity"
        if cutoff is None or cutoff >= kickoff:
            return "not_before_original_kickoff"
        prior = _asof_history(fixture, history or [], cutoff)
        population = _asof_history(fixture, league_history or [], cutoff)
        home_count = len(engine._games(prior, home, limit=10))
        away_count = len(engine._games(prior, away, limit=10))
        if min(home_count, away_count) < 5:
            return "team_history_below_five"
        if len(population) < 35:
            return "league_history_below_thirty_five"
        unrelated = sum(
            home not in (m["homeTeam"]["id"], m["awayTeam"]["id"]) and
            away not in (m["homeTeam"]["id"], m["awayTeam"]["id"])
            for m in population
        )
        if unrelated < 12:
            return "unrelated_league_history_below_twelve"
        return "independent_strength_component_unqualified"
    except (KeyError, ValueError, TypeError, OverflowError, AttributeError):
        return "unusable_authenticated_history_structure"
