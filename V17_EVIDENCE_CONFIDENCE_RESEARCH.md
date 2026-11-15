# ZicZac V17 — evidence-gated high-confidence outcomes (research only)

**Date:** 15 November 2026. The goal is to create a *testable path* toward the user's 70% aspiration, not claim it was reached. **The original V8 exact-score five doubles, saved forecasts, verified historical scores, and V15/V16 research are unchanged.**

## What is new

- **Separate prediction category:** V17 selects single-fixture double chance (1X, X2, or 12) and, only when qualified, two-match double chance. This is NOT five exact-score doubles. It does not substitute broad-outcome accuracy for exact-score accuracy.
- **Full score probabilities:** Derived from genuine normalized, existing original V8 pre-kickoff exact-score matrices, checked against original 1X2 metadata. Scores, source IDs, digest and team IDs must match the saved source; no synthetic LLM probabilities.
- **Strict evidence and lead time:** Require 30+ verified historical source and league fixtures, at least two V8 model contributors, recorded source before kickoff, and kickoff 45 minutes–24 hours away. Four distinct teams on the same Sydney date for a two-match selection. A separate source file with the original forecast ID and input digest is mandatory for freezing. Unqualified selections abstain.
- **Transparent selection:** A 70% **uncalibrated model-estimated probability threshold** is required. For a pair, the product of each leg's estimated double-chance outcome probability must clear 70%. If no qualifying pair exists, **NO paired selection is issued**. This estimated probability is NOT an empirical success rate and requires independent model calibration.
- **Prospective write-once history:** Saves new `data/v17_confidence/forecasts/YYYY-MM-DD.json` before kickoff only. No revised result can replace one or be called prospective.
- **Automatic, read-only settlement:** `v17_confidence.py` runs from the existing 15-minute collector with no extra network/API calls and uses independently validated original-source, provider-ID, both-team, kickoff, final 90-minute results. Pending outcomes never count as losses. Publishes `data/v17_confidence/performance.json` and `evidence_status.json`.
- **Independent verification gate:** Separate observed single and paired win/loss/pending rates with 95% Wilson interval. The 70% paired win-rate gate requires at least **100 truly future frozen and verified pairs AND a Wilson 95% lower confidence bound of >=70%**. The sample size is a conservative *reporting rule*, not an estimate of guaranteed future success or profitability. If the system rarely issues picks, results must also report selection frequency.
- **Website:** Five Doubles → **V17 evidence & high-confidence outcomes**; Performance → V17 full panel. It identifies missing evidence instead of hiding a failed provider connection.

## Source readiness audit

Current real data gaps found on 15 November 2026:
- Configured AI services are not equivalent to a valid **source-matched six-provider panel for both original frozen legs**; the archived audit finds none.
- V8's expected-goal parameters are statistical model estimates, **not measured shot-quality xG/xGA**. There is no licensed, audited historical xG feed.
- A verified injury/lineup endpoint exists for **on-demand display**, but player-level absence and XI have **no measured, trained per-match numerical impact model**. We do not invent one.
- v9/v10 can use timestamped authentic market odds when actually returned; there is insufficient matched archived market evidence to establish prospectively validated calibration.
- Retrospective exact five-double tests: 5 wins from 122 eligible historical days (4.10%) for V8 and V14; not untouched proof. Real completed original five-double sets: 0/1 at this initial snapshot; five pending.

This V17 release **does not purchase or enable new xG subscriptions, bookmaker feeds, AI requests, or raise GitHub Actions LLM_RESEARCH_MAX_MATCHES**. Those services require independently authenticated provider rights, data schemas, coverage, quota and user-approved costs. Do not describe them as connected until verified.

## Next substantive work

1. Confirm subscriptions for licensed timestamped xG and market quotes, including as-of and fixture identity; archive real provider responses instead of fictional values.
2. Build calibrated goal distributions with historical leagues, form, rest, verified team availability and de-vigged market consensus as *real, ablated independent features*.
3. Conduct forward-chaining, competition-stratified out-of-sample evaluation before choosing weights; inspect log loss, Brier, reliability diagrams and calibration by confidence bucket, and limit selection bias.
4. Collect hundreds of prospective original source records and compare exact-score, 1X2, double chance, and paired double chance separately, with no post-result changes.
5. Only promote a challenger if held-out performance and error bars support it; allow rejection/no bet on poorly evidenced days.

**Mathematical limitation:** Five exact-score double cells typically capture only a small fraction of the possible joint football final-score distribution. A 70% *correct exact-score double* win rate is not demonstrated or reasonably promised by using six language models or broad-outcome probabilities.

Tests: `python -m pytest -q tests/test_v17_confidence.py`. GitHub CI runs the full suite and compiles the app. Original records stay immutable.
