"""Read-only diagnostics of UEFA history already delivered to the capture run.

Never request a provider, change a forecast, expand a history window or
assume that an empty response means a team did not participate.
"""
from collections import Counter
from history_pipeline import season_year
from research_ensemble_v8 import _asof_history


def audit_history(fixture, evidence, history_cache):
    if fixture.get("provider") != "API-Football":
        return None
    comp = fixture.get("competition") or {}
    if comp.get("code") not in ("CL", "EL"):
        return None
    home = (fixture.get("homeTeam") or {}).get("id")
    away = (fixture.get("awayTeam") or {}).get("id")
    if home is None or away is None or home == away:
        return None
    season = season_year(fixture)
    key = str(comp.get("id") or comp.get("code"))
    qualified = _asof_history(fixture, evidence.get("history") or [],
                              evidence["as_of_utc"])
    def counts(rows):
        result = Counter()
        for row in rows:
            teams = {str((row.get(side) or {}).get("id"))
                     for side in ("homeTeam", "awayTeam")}
            result["home"] += int(str(home) in teams)
            result["away"] += int(str(away) in teams)
        return {"home": result["home"], "away": result["away"]}
    seasons = []
    for year in (season, season - 1):
        subset = [row for row in qualified if season_year(row) == year]
        cached = history_cache.get(("API-Football", key, year))
        item = {"season": year, "qualified_population": len(subset),
                "qualified_team_matches": counts(subset),
                "source_state": "not_observed", "returned_season_rows": None,
                "returned_team_rows": None, "source_season_mismatches": None}
        if cached is not None:
            rows, metadata, error = cached
            item["source_state"] = ("request_unavailable" if error else
                                    "stale_response" if metadata.get("stale") else
                                    "empty_response" if not rows else "rows_returned")
            item["returned_season_rows"] = len(rows)
            item["returned_team_rows"] = counts(rows)
            item["source_season_mismatches"] = metadata.get("source_season_mismatches")
            item["freshness_known"] = bool(metadata.get("freshness_known"))
        seasons.append(item)
    team_counts = counts(qualified)
    return {"provider": "API-Football", "fixture_id": str(fixture["id"]),
            "competition": comp["code"], "as_of_utc": evidence["as_of_utc"],
            "kickoff_utc": fixture["utcDate"], "seasons": seasons,
            "qualified_team_matches": team_counts,
            "minimum_missing_to_five": max(0, 5-min(team_counts.values())),
            "policy_exclusions": dict(evidence.get("exclusions") or {}),
            "absence_not_proof_of_nonparticipation": True,
            "no_additional_provider_calls": True,
            "forecast_policy_unchanged": True}
