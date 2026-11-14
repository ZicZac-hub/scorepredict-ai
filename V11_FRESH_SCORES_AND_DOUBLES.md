# ZicZac v11 — Fresher exact-score evidence and automatic five doubles

**Implemented 14 November 2026.** All original archived forecasts and double
combinations remain untouched; this is an evidence-quality improvement, NOT
proof of a new win rate.

## Main changes

1. **One-time near-kickoff research.** Previously, the v8 record often
   existed days before a match and never received a new verified historical
   forecast. `v10_prospective.capture_near_research` stores a *separate* v8
   challenger (`model_key=v8_near`) when kickoff is within 45 minutes to 48
   hours and verified team/league evidence exists. It is strictly captured
   before kickoff and never overwrites the original.

2. **Automatic schedule.** The existing 15-minute GitHub capture continues.
   A heavy run (about every six hours) scans at most eight near-kickoff
   candidates and archives qualifying refreshes. Bookmaker requests are
   still capped at four per heavy run, and fresh historical research does
   not depend on having bookmaker odds.

3. **Five-double selection uses the newer saved information where
   available.** On future matchdays, the scheduled selector deduplicates
   fixtures, prefers valid separately archived `v8_near` records, and
   otherwise retains original v8 predictions. It still requires two matches
   on the same Sydney date, four distinct teams, adequate research, a
   minimum 45-minute lead and the unvalidated 4% modelled-coverage
   operating guardrail. It abstains if no candidate qualifies. Earlier
   frozen selections are never revised retroactively.

4. **Fair market and historical comparisons.** Independently saved v9/v10
   results are only included as *same-information* comparisons when each
   match has a before-kickoff v8/v8_near/v8_control record with an identical
   historical evidence digest. Older original-versus-fresh v8 are graded
   separately as a *different-information* comparison.

5. **Automatic website display.** On the **Five Doubles** page,
   `automatic_doubles_ui.py` displays the next genuinely frozen match
   pair, five exact-score combinations, Sydney kickoff times, model source,
   estimated coverage and the eventual verified status. No user button or
   extra odds request is needed. Pending matches are not losses.

6. **Improve the bounded manual AI selector.** Instead of simply taking
   the first 18 worldwide kickoffs, the new `select_auto_research_pool`
   samples up to 18 researched matches from up to three upcoming Sydney
   calendar days, reserving at least two fixtures per date, then
   prioritises grounded existing home/away match history.

7. **Independent evaluation.** `data/v10_prospective/performance.json`
   will report v8_near exact-score accuracy separately, plus paired
   observed older-vs-fresher hits on the same matches. Do not interpret
   estimated joint-double coverage as observed accuracy.

## Important accuracy limitations

- There is no established "world's most accurate" exact-score formula.
- Newer verified evidence may help but can also worsen predictions.
  The gain must be demonstrated on future completed matches.
- No genuine licensed xG/xGA, measured player-availability numerical
  coefficients, or independently tested external LLM votes were added.
  Team news, scorelines or bookmaker prices are never fabricated.
- The Streamlit live HTTP monitor encounters an authentication 303
  redirect; this can be inconclusive. Passed GitHub tests confirm code
  health, NOT a rendered production session or deployment commit.
- Historic five-double files frozen before the 24-hour selection policy
  remain unchanged by design; the policy applies to NEW selections only.

## Verification

GitHub Actions -> Prediction integrity tests; Capture verified pre-match
evidence; Streamlit hosting monitor (inconclusive if redirect-gated).

Key files:
- `v10_prospective.py`, `v10_evaluate.py`, `capture_prematch.py`
- `smart_double_v8.py`, `v8_doubles_ui.py`
- `automatic_doubles_ui.py`, `v10_status_ui.py`, `app.py`
- `tests/test_v10_prospective.py`, `tests/test_automatic_doubles_ui.py`
