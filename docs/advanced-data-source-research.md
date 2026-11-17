# Advanced football source research — 17 November 2026

This document distinguishes genuinely observed pre-match research from model forecasts. **No provider can supply every feature for every fixture**, and missing observations are not assigned zero values or imputed as verified facts.

## Provider evidence and limitations

Official API-Football references:
- https://www.api-football.com/documentation-v3
- https://www.api-football.com/news/post/how-to-get-started-with-api-football-the-complete-beginners-guide
- https://www.api-football.com/pricing/

1. **League/season coverage**: The `/leagues` endpoint provides coverage flags such as fixture statistics, starting lineups, injuries and odds. Even a `true` flag does not guarantee records for every match.
2. **Historical event data**: `/fixtures/statistics?fixture=ID` may provide shots on/off target, shots in/out of box, blocks, possession, corners, fouls, offsides, cards, goalkeeper saves and passes. Collect this only for independently finished past matches and preserve the fixture identity and capture time. **These counts are not themselves xG**. Genuine xG is stored separately, *only* if both teams have explicitly reported `Expected Goals` in the returned statistics.
3. **Lineups**: `/fixtures/lineups` may publish first elevens about 20–40 minutes before kickoff. A pre-match XI requires exactly eleven identified starters per side. Some competitions release lineups only after the match.
4. **Injuries/suspensions**: `/injuries` distinguishes unavailable or questionable players where covered. An empty result is **not** proof every player is healthy.
5. **Pre-match odds**: `/odds` is generally available 1–14 days before kickoff, with limited history; the source changes about every three hours. Store price snapshots and source timestamps, do not confuse fair probabilities with actual bookmaker odds or assume independence of bookmakers.
6. **Independent API-Football opinion**: `/predictions?fixture=ID` publishes a model-based view on some fixtures, without bookmakers' odds. Save it only under source-matched fixture and teams. This is not an exact-score ground truth and has **zero production model weight** until prospective testing demonstrates value.

## Collection stages

- `48h → 90m`: bounded early source research for *already frozen* fixtures: history, matched injuries, market evidence, independent provider opinion. This allows collection for an upcoming match well before its final lineup.
- `90m → 35m`: no repeated expensive early polls; wait for the near-kickoff stage.
- `35m → 8m`: one additional bounded snapshot for qualifying fixtures to capture announced starting lineups without rewriting the early evidence.
- Every provider request must finish before actual kickoff to be eligible as **pre-match** evidence; later responses must be rejected.
- Maximum **two attempted fixture captures per workflow run** and **six per UTC day** by default (configurable via `V21_DAILY_CAPTURE_ATTEMPTS`, capped at 24). Reserve up to two daily attempts for near-kickoff research; historical fixture lookups may consume multiple underlying HTTP requests. Quota exhaustion is reported rather than hidden.
- Original predictions and earlier research packets remain write-once. New revisions are timestamped and saved as separate source versions.

## Next measured improvement priorities

1. Check coverage per league and season, compare recorded source availability with provider coverage flags and avoid unsupported endpoints where feasible.
2. Extend *source-independent* scheduled research to model-ready fixtures that have not yet received an original V8 freeze, while preserving separate forecast identities; this is not implemented in V21's original-only collector.
3. Build a fixture-source reliability dashboard: coverage, freshness, failed endpoints, and data latency by competition.
4. Use historical shot features as **separate candidate predictors** with proper cross-season walk-forward validation, not as invented xG.
5. Evaluate calibration (log loss/Brier/score-grid likelihood) and archived paired five-double hit rates on prospective, immutable, timestamped matches. No method gets promoted because it increases a retrospectively tuned hit rate.
6. Distinguish subscription quota and league/season data availability. Adding AI providers does not solve missing first-party evidence.

**Scope caveat**: The current scheduled implementation is limited to originally frozen V8 legs and the source budget; it does not ingest all global fixtures or guarantee a 70% exact-score double success rate.

## Newly integrated Sportmonks independent genuine-xG collector

The Sportmonks `xGFixture` fixture include can return `EXPECTED_GOALS`
(type ID **5304**). Official references:
- https://docs.sportmonks.com/v3/tutorials-and-guides/tutorials/expected/includes
- https://docs.sportmonks.com/v3/endpoints-and-entities/endpoints/expected-xg
- https://docs.sportmonks.com/v3/definitions/states

`sportmonks_xg_research.py` is a strictly source-separated, research-only
collector. It uses an **existing** `THIRD_FOOTBALL_API_KEY` or
`SPORTMONKS_API_TOKEN` GitHub secret, if present. It does not purchase or
upgrade any provider package. A normal Sportmonks fixture subscription does
not imply that the optional xG add-on is available.

For eligible original V8 forecasts within 48 hours of kickoff, it uses the
Sportmonks **fixture-by-date** endpoint to uniquely match exact normalised
home and away names and kickoff within five minutes. It then uses only the
matched Sportmonks home/away participant IDs to look up their independently
completed historical fixtures within the last 60 days (one page of 50 per
team), using the documented `xGFixture` include. Records are accepted only
if the historical fixture has state ID `5` (regulation full time), was
available at least three hours before collection and occurred before the
target fixture, and contains **both** participant-matched type-5304 xG values.

The collector has a default maximum of **four Sportmonks requests per UTC
day**, reserved before network access, with an absolute maximum of twelve if
the owner adjusts `SPORTMONKS_XG_DAILY_REQUESTS`. An attempted source match
is checkpointed once per original forecast to avoid repeated requests every
15 minutes. No requests occur on GitHub source-code pushes.

It writes evidence under `data/sportmonks_xg/forecasts/`, audit reasons to
`data/sportmonks_xg/status.json`, and shows separate coverage on the
Advanced Quality Report. These Sportmonks xG records **are not currently fed
into V22 forecasts**, not merged with API-Football fixture IDs, and cannot
support a claimed accuracy gain until tested prospectively on matched data.
Missing add-on access, incomplete xG, restricted league coverage and ambiguous
matching are reported, not treated as xG of zero.

### Fixture-matching upgrade (17 November 2026)

Sportmonks date discovery now tries an official `participantSearch` filter for
the forecast home team, with **full pagination** before claiming a unique
match. If that name search returns no match, the collector falls back to
paginated date discovery. It will **not** trust a match while `has_more=true`
and pages remain unexamined. Partial scans and cached page responses remain
diagnosable, and previously fetched discovery pages can be reused for up to
36 hours to avoid repetitive provider requests.

Team aliases are a small reviewed whole-name dictionary (e.g. Coritiba /
Coritiba Foot Ball Club, Santos / Santos FC, Manchester City / Man City).
Case and accents are normalised, but no fuzzy edit-distance matching or
blanket deletion of `U20`, women's, reserve or location qualifiers occurs.
Both teams, home/away order and kickoff within 5 minutes must agree; multiple
eligible Sportmonks fixture IDs mean **no match**, not a guess.

Unmatched forecast attempts are no longer permanently blacklisted after one
try. A previous attempt marker is migrated to a count with **45-minute
minimum retry interval**, at most three attempts for each original forecast,
and a slower 24-hour cooldown after provider access/network failures.
Previously successful forecasts remain immutable. The existing four-request
daily Sportmonks limit (absolute configurable maximum twelve) and four-call
per-run budget remain enforced. Retry checkpoints and source status never
save API tokens. Exceeded pagination/request budgets are reported explicitly
instead of reporting an unverified fixture as confirmed.

A correctly configured token proves authentication availability only; it does
**not** prove the subscription includes the Sportmonks xG add-on or that the
target competition has historical xG. The real production run must return
type-5304 xG values for both verified historical participants before those
records are reported as collected.

References:
- https://docs.sportmonks.com/v3/endpoints-and-entities/endpoints/fixtures/get-fixtures-by-date
- https://docs.sportmonks.com/v3/endpoints-and-entities/endpoints/fixtures/get-fixtures-by-date-range-for-team
- https://docs.sportmonks.com/v3/tutorials-and-guides/tutorials/expected/includes
