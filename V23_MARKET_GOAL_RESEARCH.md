# V23 — Market-adjusted goal research (shadow mode)

## Why this upgrade exists

The V21 historical benchmark compared unchanged V8 against V20 goal-form
on the **same 393 finished matches**. V8 had 43/393 top exact-score hits and
V20 had 41/393. V20 placed the correct score in its top five for 185/393,
versus V8 for 183/393, but V20's score log loss was worse: 3.01486 vs
3.01183. This is retrospective and **does not prove any future model is
superior**. V23 therefore gives V20's home/away observations **zero
numerical weight** unless independent future tests support their inclusion.

API-Football's documented fixture-statistics fields don't consistently
include shot-based xG. V22 correctly requires real xG and can abstain for
all matches when that field is missing. V23 adds a **separately tracked
market + original-score forecast** that does **not** invent xG.

## How V23 scores

1. Require the original, full, frozen V8 probability matrix.
2. Require an identity-matched provider packet with **1X2 AND an Over/Under
   2.5 or 3.5 goal market**, with time-proven price observations no later
   than the fixture kickoff and no older than twelve hours at analysis.
3. Accept bookmaker-supplied quote timestamps when available; otherwise
   accept the verified fetched-at time of a first-party API-Football response.
   These timestamps are **labelled separately** and are never guessed from
   the current clock or stale cache.
4. Use true historical shot-xG to improve the goal prior only when six
   samples per team and appropriate home/away samples genuinely exist.
   Otherwise retain the unchanged original statistical goal prior.
5. Adjust the full 0–10 goal-probability matrix toward de-vigged 1X2 and
   totals/BTTS market probabilities, with bounded 15% total-variation move.
   It does **not** force a 3–1 or 3–2 result for famous teams.
6. Rank five two-match exact-score combinations by estimated joint
   probability, subject to same-Sydney-day/four-distinct-team rules and
   a conservative minimum estimated coverage. Automatically abstain on
   missing evidence.
7. Save write-once original and V23 predictions for the *same two fixtures*
   and independently grade verified hit/miss/pending results.

The resulting probabilities are **experimental and not calibrated to real
V23 performance**. Estimated coverage is not an observed win rate.

## How to use

In the **AI Five-Double Studio**, choose either:
- **V23 Market + Goal Form (no real xG required)** for a manually selected
  pair, or
- **V23 Auto Select** to compare all saved-source-qualified pairs.

For new odds research, sign into your private website and explicitly tick
the API-Football source-consent checkbox; select minimal historical xG
requests if your plan has limited quota. Paid research and AI calls are never
enabled simply by deploying the repository.

For optional scheduled near-kickoff source collection, after checking API
quotas, set the GitHub Actions variable `V21_EVIDENCE_ENABLED=true` and
ensure `API_FOOTBALL_KEY` exists as a repository secret. The separate
`V22_DEEP_XG_ENABLED` variable is **not needed** for V23; leave it unset
to conserve provider requests. The scheduled research requires a previously
frozen original two-match selection.

For **earlier market collection** (the most useful V23 option if odds vanish
near kickoff), optionally set another GitHub Actions variable,
`V23_EARLY_ODDS_ENABLED=true`. It captures at most two original frozen
fixture legs in **18–36 hours before kickoff** and, if necessary,
**6–12 hours before kickoff**. Each eligible stage allows at most one odds
request per fixture and the workflow caps source requests at two per run.
A complete, still-fresh earlier packet avoids another request. A failed
early lookup may be retried once later instead of permanently disabling the
match. Each successful packet remains write-once. This is
**disabled by default** and can consume API quota. It never claims the
quote was updated at the moment it was retrieved: provider-fetched time
and bookmaker update time are labelled separately.

For prospective V23 score prediction, the freshest valid 1X2 + goal-total
market snapshot is preferred over unrelated historical statistical fields.
Older original snapshots remain immutable.

Manual source capture is durable only if private `DATABASE_URL` is
configured in Streamlit Secrets. The app warns when a new packet is
session-only. GitHub-scheduled forecasts are saved to
`data/v23_prospective/` separately and automatically verified.

## Testing, evidence and limitations

- Regression suite: `tests/test_v23_market_form.py` and existing V21/V22
  source, identity, lineup and same-pair tests.
- Historical replay source: `v21_historical_benchmark.py` and its
  `paired_v8_v20_same_fixtures` report.
- Real V23 win rate is **unknown until future pairs settle**. V22 most
  recently had zero qualifying source packets and zero verified sets.
- League-specific bookmaker coverage and actual quote availability vary;
  reliable odds can be missing even when normal match results exist.
- Original frozen forecasts are immutable and **never silently replaced**.
- External AI opinions retain zero numerical weight until validated
  independently on matched, pre-match evidence.
