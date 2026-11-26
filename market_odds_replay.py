"""Reproduce the fixed V8 versus opening-Pinnacle score-grid replay."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import csv
from datetime import datetime, timedelta, timezone
import hashlib
import io
from itertools import combinations, groupby
import json
import math
from pathlib import Path

import numpy as np
import requests

from five_double_accuracy import wilson
from research_ensemble_v8 import predict
from target70_data import DEFAULT_CACHE, SOURCE_MANIFEST, load_normalized
from v22_evidence_models import calibrate_markets
from v28_independent_pair_trial import _discordance_p

SEASONS = ("2023-24", "2024-25")
COMPETITIONS = {"de.1": "D1", "en.1": "E0", "es.1": "SP1", "fr.1": "F1", "it.1": "I1"}
ODDS_URL = "https://www.football-data.co.uk/mmz4281"
ODDS_SHA256 = {
    "2324-D1": "1de8bc12a133dbbf71bc6ffda62082642f087f61b1b7c15ae9f2d320876fb07e",
    "2324-E0": "b2e057b0ed959f198b0f63d2391c01239f3608e6de5db68edab3f88e04d07ff3",
    "2324-F1": "ca0a4bf51d5b1ccfd0e1558ea41de6bb829df6c96c759888a87248d296d5adb7",
    "2324-I1": "343a89db026fe4204ebeabd663cdfee8d02a45457828fc1cb63fa630ca170723",
    "2324-SP1": "5323f596c7ed318e4bce95557a369d809d98501fa499ea5f41350cd45cf3c7d9",
    "2425-D1": "a295c58033cfe6393cbbc28f7db85671839b960c5f1153fd3678e20ad2adb92e",
    "2425-E0": "d0c8ce4a96d886cf60cf101f570f4a3893844226f91c7bd769eb568c49edbfa4",
    "2425-F1": "fe6ae6a0c30d05f0ac88e0fc4a31452c7203e2b444703dfa941e22f38f20ab63",
    "2425-I1": "79d87a5b18771e066ac7419b7876bfd9771a68ca117a9712375d6bc9685abee3",
    "2425-SP1": "c086e96d6a53f6e6c0fce4e35db0d1728c773b4362323da0161a34769e315a36",
}
CLUB_ALIASES = {
    'AFC Bournemouth': 'Bournemouth',
    'Arsenal FC': 'Arsenal',
    'Aston Villa FC': 'Aston Villa',
    'Brentford FC': 'Brentford',
    'Brighton & Hove Albion FC': 'Brighton',
    'Burnley FC': 'Burnley',
    'Chelsea FC': 'Chelsea',
    'Crystal Palace FC': 'Crystal Palace',
    'Everton FC': 'Everton',
    'Fulham FC': 'Fulham',
    'Ipswich Town FC': 'Ipswich',
    'Leicester City FC': 'Leicester',
    'Liverpool FC': 'Liverpool',
    'Luton Town FC': 'Luton',
    'Manchester City FC': 'Man City',
    'Manchester United FC': 'Man United',
    'Newcastle United FC': 'Newcastle',
    'Nottingham Forest FC': "Nott'm Forest",
    'Sheffield United FC': 'Sheffield United',
    'Southampton FC': 'Southampton',
    'Tottenham Hotspur FC': 'Tottenham',
    'West Ham United FC': 'West Ham',
    'Wolverhampton Wanderers FC': 'Wolves',
    'Athletic Club': 'Ath Bilbao',
    'CA Osasuna': 'Osasuna',
    'CD Leganés': 'Leganes',
    'Club Atlético de Madrid': 'Ath Madrid',
    'Cádiz CF': 'Cadiz',
    'Deportivo Alavés': 'Alaves',
    'FC Barcelona': 'Barcelona',
    'Getafe CF': 'Getafe',
    'Girona FC': 'Girona',
    'Granada CF': 'Granada',
    'RC Celta de Vigo': 'Celta',
    'RCD Espanyol de Barcelona': 'Espanol',
    'RCD Mallorca': 'Mallorca',
    'Rayo Vallecano de Madrid': 'Vallecano',
    'Real Betis Balompié': 'Betis',
    'Real Madrid CF': 'Real Madrid',
    'Real Sociedad de Fútbol': 'Sociedad',
    'Real Valladolid CF': 'Valladolid',
    'Sevilla FC': 'Sevilla',
    'UD Almería': 'Almeria',
    'UD Las Palmas': 'Las Palmas',
    'Valencia CF': 'Valencia',
    'Villarreal CF': 'Villarreal',
    '1. FC Heidenheim 1846': 'Heidenheim',
    '1. FC Köln': 'FC Koln',
    '1. FC Union Berlin': 'Union Berlin',
    '1. FSV Mainz 05': 'Mainz',
    'Bayer 04 Leverkusen': 'Leverkusen',
    'Borussia Dortmund': 'Dortmund',
    'Borussia Mönchengladbach': "M'gladbach",
    'Eintracht Frankfurt': 'Ein Frankfurt',
    'FC Augsburg': 'Augsburg',
    'FC Bayern München': 'Bayern Munich',
    'FC St. Pauli 1910': 'St Pauli',
    'Holstein Kiel': 'Holstein Kiel',
    'RB Leipzig': 'RB Leipzig',
    'SC Freiburg': 'Freiburg',
    'SV Darmstadt 98': 'Darmstadt',
    'SV Werder Bremen': 'Werder Bremen',
    'TSG 1899 Hoffenheim': 'Hoffenheim',
    'VfB Stuttgart': 'Stuttgart',
    'VfL Bochum 1848': 'Bochum',
    'VfL Wolfsburg': 'Wolfsburg',
    'AJ Auxerre': 'Auxerre',
    'AS Monaco FC': 'Monaco',
    'AS Saint-Étienne': 'St Etienne',
    'Angers SCO': 'Angers',
    'Clermont Foot 63': 'Clermont',
    'FC Lorient': 'Lorient',
    'FC Metz': 'Metz',
    'FC Nantes': 'Nantes',
    'Le Havre AC': 'Le Havre',
    'Lille OSC': 'Lille',
    'Montpellier HSC': 'Montpellier',
    'OGC Nice': 'Nice',
    'Olympique Lyonnais': 'Lyon',
    'Olympique de Marseille': 'Marseille',
    'Paris Saint-Germain FC': 'Paris SG',
    'RC Strasbourg Alsace': 'Strasbourg',
    'Racing Club de Lens': 'Lens',
    'Stade Brestois 29': 'Brest',
    'Stade Rennais FC 1901': 'Rennes',
    'Stade de Reims': 'Reims',
    'Toulouse FC': 'Toulouse',
    'AC Milan': 'Milan',
    'AC Monza': 'Monza',
    'ACF Fiorentina': 'Fiorentina',
    'AS Roma': 'Roma',
    'Atalanta BC': 'Atalanta',
    'Bologna FC 1909': 'Bologna',
    'Cagliari Calcio': 'Cagliari',
    'Como 1907': 'Como',
    'Empoli FC': 'Empoli',
    'FC Internazionale Milano': 'Inter',
    'Frosinone Calcio': 'Frosinone',
    'Genoa CFC': 'Genoa',
    'Hellas Verona FC': 'Verona',
    'Juventus FC': 'Juventus',
    'Parma Calcio 1913': 'Parma',
    'SS Lazio': 'Lazio',
    'SSC Napoli': 'Napoli',
    'Torino FC': 'Torino',
    'US Lecce': 'Lecce',
    'US Salernitana 1919': 'Salernitana',
    'US Sassuolo Calcio': 'Sassuolo',
    'Udinese Calcio': 'Udinese',
    'Venezia FC': 'Venezia',
}


def read_quotes(directory, *, download=False):
    """Read hash-matching public CSVs, downloading only when asked explicitly."""
    output = {}
    for name, digest in ODDS_SHA256.items():
        path = directory / (name + ".csv")
        if path.exists():
            raw = path.read_bytes()
        elif download:
            period, competition = name.split("-")
            response = requests.get(f"{ODDS_URL}/{period}/{competition}.csv", timeout=30)
            response.raise_for_status()
            raw = response.content
        else:
            raise FileNotFoundError(path)
        if hashlib.sha256(raw).hexdigest() != digest:
            raise ValueError(f"Pinned odds checksum differs for {name}")
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
        rows = {}
        for row in csv.DictReader(io.StringIO(raw.decode("utf-8-sig"))):
            if not all(row[key] for key in ("Date", "HomeTeam", "AwayTeam")):
                continue
            date = datetime.strptime(row["Date"], "%d/%m/%Y").date().isoformat()
            key = date, row["HomeTeam"], row["AwayTeam"]
            if key in rows:
                raise ValueError(f"Ambiguous dated fixture in {name}: {key}")
            rows[key] = row
        output[name] = rows
    return output


def probabilities(row, columns):
    prices = [float(row[column]) for column in columns]
    if any(not math.isfinite(price) or price <= 1 for price in prices):
        raise ValueError("Invalid decimal price")
    implied = [1 / price for price in prices]
    overround = math.fsum(implied)
    return [value / overround for value in implied]


def replay_competition(job):
    league, rows = job
    past, forecasts = [], []
    for day, group in groupby(rows, key=lambda row: row["source_date"]):
        # Midnight is a date-ordering surrogate. Train exclusively on earlier dates.
        cutoff = datetime.fromisoformat(day).replace(tzinfo=timezone.utc) - timedelta(hours=1)
        horizon = (cutoff - timedelta(days=540)).date().isoformat()
        prior = [match for match in past if match["utcDate"][:10] >= horizon]
        finished = []
        for index, row in enumerate(group):
            home, away = row["home_team"], row["away_team"]
            fixture = {
                "id": f"{league}/{day}/{index}", "utcDate": day + "T00:00:00Z",
                "provider": "openfootball-calendar-replay", "competition": {"id": league},
                "homeTeam": {"id": home}, "awayTeam": {"id": away}, "status": "SCHEDULED",
                "score": {"fullTime": {"home": None, "away": None}},
            }
            actual = row["home_goals"], row["away_goals"]
            if row["season"] in SEASONS:
                teams = {home, away}
                history = [match for match in prior if teams.intersection(
                    (match["homeTeam"]["id"], match["awayTeam"]["id"]))]
                result = predict(fixture, history, league_history=prior, as_of=cutoff)
                if result and result["component_count"] >= 2:
                    forecasts.append((row["season"], day, fixture["id"], home, away, actual, result["matrix"]))
            finished.append({**fixture, "status": "FINISHED", "score": {
                "fullTime": {"home": actual[0], "away": actual[1]},
            }})
        past.extend(finished)
    return league, forecasts


def best_scores(grid):
    leading = np.argsort(-grid.ravel(), kind="stable")[:5]
    return [(float(grid[home, away]), (int(home), int(away)))
            for home, away in zip(*np.unravel_index(leading, grid.shape))]


def best_tickets(left, right):
    joint = [(p * q, first, second) for p, first in left for q, second in right]
    return sorted(joint, key=lambda row: (-row[0], row[1:]))[:5]


def score_match(entry, arm):
    grid, leading = entry["models"][arm]
    actual = entry["actual"]
    probability = float(grid[actual]) if all(i < extent for i, extent in zip(actual, grid.shape)) else 0.
    return (actual == leading[0][1], any(actual == score for _, score in leading),
            -math.log(max(1e-15, probability)))


def compare_day(entries):
    """Exhaust both pair policies and rerank five tickets on the V8 control pair."""
    best = {"v8": None, "market": None}
    pairs = 0
    for left, right in combinations(sorted(entries, key=lambda entry: entry["id"]), 2):
        if left["teams"] & right["teams"]:
            continue
        pairs += 1
        for arm in best:
            tickets = best_tickets(left["models"][arm][1], right["models"][arm][1])
            estimate = math.fsum(p for p, _, _ in tickets)
            if best[arm] is None or estimate > best[arm][0]:
                best[arm] = estimate, left, right, tickets
    if not pairs:
        return None
    _, left, right, _ = best["v8"]
    matched = best_tickets(left["models"]["market"][1], right["models"]["market"][1])
    best["market_on_v8_pair"] = math.fsum(p for p, _, _ in matched), left, right, matched
    return {
        "pairs": pairs, **{
            ("market_own_pair" if arm == "market" else arm): (
                estimate, any(row[1:] == (left["actual"], right["actual"]) for row in tickets))
            for arm, (estimate, left, right, tickets) in best.items()
        },
    }


def paired_log_loss(days):
    """Use whole source dates as paired blocks for the mean loss difference."""
    if not days:
        return None
    differences = np.array([
        math.fsum(v8[2] - market[2] for v8, market in rows) for rows in days
    ])
    sizes = np.array([len(rows) for rows in days])
    samples = np.random.default_rng(20261124).integers(len(days), size=(5000, len(days)))
    bootstrap = differences[samples].sum(axis=1) / sizes[samples].sum(axis=1)
    return {
        "mean_v8_minus_market": float(differences.sum() / sizes.sum()),
        "source_dates": len(days),
        "source_date_bootstrap_95": np.quantile(bootstrap, (.025, .975)).tolist(),
    }


def compare_season(season, forecasts, quotes):
    group = defaultdict(list)
    opportunity_days, excluded, competitions = set(), Counter(), Counter()
    for league, rows in forecasts.items():
        name = season[2:4] + season[5:] + "-" + COMPETITIONS[league]
        for year, date, fixture_id, home, away, actual, baseline in rows:
            if year != season:
                continue
            opportunity_days.add(date)
            quote = quotes[name].get((date, CLUB_ALIASES[home], CLUB_ALIASES[away]))
            if quote is None:
                excluded["identity_not_found"] += 1
                continue
            if actual != (int(quote["FTHG"]), int(quote["FTAG"])):
                excluded["full_time_score_mismatch"] += 1
                continue
            try:
                winner = probabilities(quote, ("PSH", "PSD", "PSA"))
                totals = probabilities(quote, ("P>2.5", "P<2.5"))
            except (KeyError, TypeError, ValueError):
                excluded["missing_opening_pinnacle_markets"] += 1
                continue
            markets = {
                "1x2": {"probabilities": dict(zip(("home", "draw", "away"), winner))},
                "over_2_5": {"probabilities": dict(zip(("yes", "no"), totals))},
            }
            adjusted = calibrate_markets(baseline, markets, strength=.35, max_move=.15)
            if not adjusted["ready"]:
                raise ValueError(f"Qualified markets did not produce a grid for {fixture_id}")
            market = adjusted["matrix"]
            group[date].append({
                "id": fixture_id, "teams": {(league, home), (league, away)}, "actual": actual,
                "models": {"v8": (baseline, best_scores(baseline)), "market": (market, best_scores(market))},
            })
            competitions[league] += 1
    matched = sum(competitions.values())
    by_day = [[(score_match(entry, "v8"), score_match(entry, "market")) for entry in group[day]]
              for day in sorted(group)]
    single_match = {}
    for index, arm in enumerate(("v8", "market")):
        scores = [row[index] for rows in by_day for row in rows]
        top1, top5 = sum(hit for hit, _, _ in scores), sum(hit for _, hit, _ in scores)
        single_match[arm] = {
            "top1_hits": top1, "top1_rate": top1 / matched if matched else None,
            "top5_hits": top5, "top5_rate": top5 / matched if matched else None,
            "exact_score_log_loss": math.fsum(loss for _, _, loss in scores) / matched if matched else None,
        }
    comparisons = [result for entries in group.values() if (result := compare_day(entries))]
    issued = len(comparisons)
    doubles = {}
    for arm in ("v8", "market_own_pair", "market_on_v8_pair"):
        wins = sum(row[arm][1] for row in comparisons)
        v8_only = sum(row["v8"][1] and not row[arm][1] for row in comparisons)
        market_only = sum(row[arm][1] and not row["v8"][1] for row in comparisons)
        doubles[arm] = {
            "hits": wins, "wilson_95": wilson(wins, issued),
            "mean_model_mass": math.fsum(row[arm][0] for row in comparisons) / issued if issued else None,
            "v8_only_hits": v8_only, "arm_only_hits": market_only,
            "paired_exact_test_two_sided_p": _discordance_p(v8_only, market_only),
        }
    return {
        "matched_fixtures": matched, "matched_fixtures_by_league": dict(competitions),
        "matching_exclusions": dict(excluded), "single_match": single_match,
        "paired_exact_score_log_loss": paired_log_loss(by_day),
        "source_dates_with_v8_grids": len(opportunity_days), "dates_issuing_both_arms": issued,
        "four_team_candidate_pairs_compared": sum(row["pairs"] for row in comparisons), "five_doubles": doubles,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--openfootball-cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--odds-directory", type=Path, required=True)
    parser.add_argument("--download-odds", action="store_true")
    args = parser.parse_args()
    quotes = read_quotes(args.odds_directory, download=args.download_odds)
    leagues = defaultdict(list)
    for row in load_normalized(cache_dir=args.openfootball_cache):
        leagues[row["competition"]].append(row)
    with ProcessPoolExecutor(max_workers=len(COMPETITIONS)) as pool:
        forecasts = dict(pool.map(replay_competition, sorted(leagues.items())))
    print(json.dumps({
        "protocol": "v8-pinnacle-opening-v22-fixed-035-015-v1", "historical_replay": True,
        "openfootball_manifest": SOURCE_MANIFEST.as_posix(), "odds_csv_sha256": ODDS_SHA256,
        "seasons": {season: compare_season(season, forecasts, quotes) for season in SEASONS},
    }, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
