# Verified xG collection and independent Dixon–Coles comparison

## Purpose and current gate

Research only. Do not promote the xG challenger, increase live confidence, or claim
a 70% exact-score five-double hit rate without independently verified evidence.

The first step is **chronologically frozen, authenticated API-Football history**
and provider-qualified pre-kickoff xG evidence. A qualifying packet now
freezes the exact Dixon–Coles and xG probability matrices **before kickoff**;
those saved probabilities, not later refits, are graded when independently
verified results arrive. Packets without usable xG are kept as explicit
pre-kickoff abstentions and can never be retroactively filled after kickoff. The current evaluator does not
consider a post-match reconstruction of historical inputs a valid prospective
case. The pre-kickoff freeze happens on new `v21_capture_sources.py` captures.
Older API-Football packets can contain genuine historical xG but may have no
immutable, pre-kickoff historical-input archive. Never invent or backfill
those historical inputs after the result.

Sportmonks historical xG is a separate evidence stream. Its identity-matched
observations do **not** silently enter the API-Football xG challenger: a verified
cross-provider translation and frozen model specification would need a separate
review. Missing subscription access or query coverage is reported as missing.

## Running collection and auditing

- `.github/workflows/prematch-evidence.yml` schedules collection four times
  an hour on main. It uses existing `API_FOOTBALL_KEY` and bounded
  `V21_DAILY_CAPTURE_ATTEMPTS`; Sportmonks uses its separately bounded token.
  Do not add paid calls or loosen quotas simply to increase sample counts.
- `v21_capture_sources.py` preserves source packets under
  `data/v21_sources/` and, for new authenticated packets, calls
  `freeze_history` to preserve the exact as-of model inputs in
  `data/verified_xg_replay/history/`. Existing packets are write-once.
- Once each six-hour heavy refresh is due, the workflow writes
  `data/verified_xg_replay/latest_report.json` before committing data. The
  readout is not a production forecast and makes no external API calls.
- `.github/workflows/verified-xg-evidence.yml` also runs daily, on relevant
  PRs and manually, uploads `verified-xg-research-evidence`, and runs synthetic
  integrity tests. Use the workflow artifact to inspect coverage and exclusions.
- Local equivalent: `python verified_xg_evidence_pipeline.py --output verified-xg-evidence-report.json`
  after `pip install numpy pytest`. No provider API credentials are needed for
  the audit itself.

## Study preregistration

The held-out start date is `2026-12-15` UTC unless explicitly changed
**before** inspection. Require at least 30 authenticated training cases,
50 authenticated holdout cases, and 20 same-Sydney-day non-overlapping
five-double pairs as **minimum coverage**, not as a test of significance.
Report shortfalls numerically and label insufficient coverage explicitly.

For each eligible completed fixture, require: an as-of historical-input digest,
an intact provider snapshot seal, matching fixture and team IDs, captured
timestamp strictly before kickoff, and a regulation-time final score verified
independently after kickoff. One fixture contributes once; keep the earliest
eligible archived decision using a predefined rule that never consults the
final result. Missing evidence and failed joins are excluded, not imputed.

Compare **the same cases** across dynamic Dixon–Coles and the frozen
verified-xG algorithm. On the held-out cases report:

1. Top-1 exact-score hits and rate, with denominator.
2. Proper probability metrics: exact-score log loss and three-way-outcome
   Brier score; low losses are preferable.
3. Five-double hit rate and estimated five-ticket scoreline coverage on the
   identical, deterministic, same-day match pairs.
4. The paired xG-minus-Dixon–Coles differences for all matched cases
   and separately for confirmed starting XIs, partial, and unavailable.
   Lineup availability is an observational subgroup, **not** an estimate
   of the causal value of lineups: both algorithms currently give lineups
   zero numerical player-impact weight.
5. Eligible/excluded case counts and collector diagnostics. Different
   sample sizes, unverifiable data, cherry-picked windows, or post-kickoff
   information invalidate comparisons.

The researcher must separately examine statistical uncertainty,
multiple-testing/selection effects, data leakage, stability across leagues
and time, and genuinely forward-frozen model predictions before considering
promotion. A positive descriptive delta is not proof of improvement.

## Deployment gate

Until an independent audit explicitly approves a new protocol, leave
`live_prediction_model_changed=false`, `automatic_promotion=false`,
and `accuracy_improvement_verified=false`. Failed data capture is not
a reason to manufacture xG, buy access, or silently change the live engine.

## Post-merge capture verification (research workflow v2)

The collector's source archive is only a **provider-packet** count; it is not a
denominator for xG predictive accuracy. Every scheduled pre-match run now
publishes a read-only `data/verified_xg_replay/latest_report.json` even on
lightweight refreshes, including missing authenticated frozen inputs, rejected
post-match/identity-invalid records, and separate model-ready vs abstaining cases.
A synthetic collector-to-history-to-independent-result integration test verifies
the plumbing without spending API quota. It is **not real accuracy evidence**.

The pre-existing `data/professional_paired_study/` study is frozen v1. Its
strict implementation fingerprint no longer matches current collection code;
rewriting that protocol would falsify the original trial specification. The
optional matched odds/xG ablation runs under an independent
`data/professional_paired_study_v2/` protocol for new decisions only.
Neither study modifies the dynamic Dixon-Coles baseline, xG study holdout,
model serving, existing bets, or automatic-promotion gate.

A GitHub Actions green check confirms code/integrity only. To verify that the
collector is actually accumulating **new genuine evidence**, inspect the next
scheduled main-branch capture job and confirm `frozen_history_records` rises,
with a UTC pre-kickoff archived timestamp and intact original source hash.
Do not claim live evidence collection solely from synthetic tests or job setup.
