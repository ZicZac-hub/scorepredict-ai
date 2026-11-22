# ZicZac Continuous Five-Double Bot (V35)

The owner asked for a single bot to **continue researching and testing every day
even when the ChatGPT conversation is closed**, with verified five exact-score
double accuracy as the only forecasting priority.

## What is now automated

- GitHub Actions `continuous-five-double-bot.yml` checks the repository
  **four times per hour, around the clock** (a scheduled service, **not** a
  permanent live Python/Streamlit background process). GitHub schedule runs may
  be delayed, paused by GitHub, fail, or exhaust quotas. A startup push and
  manual dispatch are also supported.
- Every run verifies real V28 original source receipt/catalog integrity,
  recomputes the V34 missing-leg diagnosis and grades the genuinely frozen V35
  methods against independently identity-matched regulation-time final scores.
- The existing pre-match collection workflow freezes NEW V35 methods
  immediately after a new original V28 decision, provided original issuance
  is no more than 20 minutes old, both kickoffs are at least 45 minutes away,
  and all original input identities pass independent verification.
- V35 predeclares **six** same-fixture, same-five-ticket-budget policies:
  original V8, 75/25 V25 anchor, mild probability flattening, mild probability
  sharpening, fixed 0.15 high-goal tilt, and fixed -0.15 low-goal tilt.
  These are *new, unproven research methods*, not new production predictors.
- Each immutable new V35 archive keeps the original two selected fixtures,
  full per-method normalized probability matrices and five joint-score
  selections **before kickoff**, with the source hashes and timestamp.
  Every later audit reconstructs original source evidence and method choices
  rather than reading whatever a future recalibrated model says.
- The verified-results backfill periodically scores already frozen methods
  and preserves pending or unavailable results as pending. Duplicate physical
  matches and correlated candidate pairs are never counted as independent
  sample size.
- Source regressions and new synthetic integrity tests run under
  `prediction-tests.yml`. Failed CI cannot approve a model change.

## Direct ChatGPT/OpenAI research connection

The ChatGPT consumer login or Plus subscription **cannot be plugged into a
server-hosted website as a secret agent identity**. The supported integration
is OpenAI's separate paid **API**, using the owner's existing
`OPENAI_API_KEY` GitHub Actions secret (if installed). Keys stay exclusively
inside GitHub Secrets, never in Streamlit frontend or commit contents.

When a secret is available the controller sends **at most one small API request
per UTC day**, using the allowlisted `gpt-4o-mini` model by default (or an
explicitly allowlisted `AI_RESEARCH_MODEL` variable).
Its output is a bounded research memo describing **one falsifiable next
method hypothesis and test**. A successful memo is stored once in
`data/autobot/openai_research_YYYY-MM-DD.json`. The AI output is **never
executed** as code, never used to rewrite a forecast and cannot force a GitHub
merge. OpenAI API billing is separate from ChatGPT subscriptions. Set the
repository variable `AI_RESEARCH_BOT_ENABLED=false` to disable API charges.
If the API key is missing, deterministic research and verification continue.

Useful GitHub links:

- [Bot workflow](https://github.com/ZicZac-hub/scorepredict-ai/actions/workflows/continuous-five-double-bot.yml)
- [GitHub Actions secrets](https://github.com/ZicZac-hub/scorepredict-ai/settings/secrets/actions)
- [Website Bot dashboard](https://github.com/ZicZac-hub/scorepredict-ai/blob/main/continuous_bot_ui.py)

## Promotion safeguards and why the target is difficult

The V35 method review gate requires all of:

1. At least 100 fully independently verified **same-day, same-two-match**
   forecast sets and at least 100 distinct underlying fixtures.
2. At least 70% observed five-double hit rate, with a **95% Wilson lower
   confidence bound** no less than 70%.
3. Statistically positive paired wins over original V8 (exact sign test
   two-sided `p<0.05`).
4. Better mean exact-score log loss **and** multiclass Brier score on
   identical verified physical fixtures.
5. No tampered/missing original source evidence or altered original picks.

Passing this **opens a separate production review**; the bot does not
silently promote the live model or rewrite historic results. A safety and
regression review of a production model change still matters; a few early wins
or a test passing cannot demonstrate a genuine 70% generalizable win rate.

Existing original V8 five-score combination probabilities usually have only
single-digit percentage model-based combined coverage. A 70% real hit rate is
**not guaranteed and has not been demonstrated**. The bot will continue
collecting honest results and showing clear gaps rather than declaring
success by hiding misses, increasing ticket count, modifying forecasts after
kickoff, or cherry-picking verified days.

## Monitoring and troubleshooting

- Open the website's **⚙️ 24/7 Bot** section for saved status, method hit
  rates, pending days, integrity condition, and optional OpenAI memo.
- Check `data/autobot/status.json` in the repository, or use the GitHub
  Actions runs for actual execution success/failure and timing.
- If the first report is missing, confirm the scheduled workflow ran and
  GitHub Actions has write permission to publish research status.
- GitHub does not guarantee exact scheduled trigger times or uninterrupted
  execution; this isn't a 24/7 dedicated paid server.
- The bot only commits/generated research under `data/autobot/`,
  `data/v35_continuous_challengers/`, and the already-existing V34 diagnosis.
  Live score engines, production settings, and all original forecast paths
  remain unmodified by this scheduled bot.
- Rerun the original CI suite for code changes. The bot does not assert that
  code compilation or GitHub success proves any real football hit rate.

## How to test

```bash
python -m pytest -q tests/test_continuous_accuracy_bot.py
python v35_continuous_double_trials.py --verify-only
python continuous_accuracy_bot.py --no-ai
```

The bot's first prospective test can remain at zero verified sets until future
eligible original V28 decisions and independently verified full-time final
scores exist. That is a correct **pending/no-evidence** status, not a loss.
