# Optional Five-AI Pre-Match Research (not activated without credentials)

**Status:** Code and read-only dashboard are installed. GitHub Actions runs an opt-in
pre-match research step on six-hour heavy collection passes. Nothing is sent to
external AI providers unless you deliberately enable it and configure credentials.
Your existing v6/v7/v8/v9/v10 forecasts, frozen five doubles and settled results
are never overwritten, and the experimental AI panel cannot change the live scorer.

## What ScoreGPT research does and does not establish

ScoreGPT publicly describes independent model forecasts followed by an average
of home/draw/away outcome probabilities. Its *published pick and match-outcome*
success rates do not establish exact-score or five-double hit rates.
See https://scoregpt.app/methodology and https://scoregpt.app/accuracy.
This project implements its **own** optional independent research adapter;
it does not access ScoreGPT's proprietary predictions or pretend it has
connected to ScoreGPT.

## GitHub configuration

Go to repository **Settings -> Secrets and variables -> Actions**.

Create five repository **secrets** as needed (do not paste credentials into GitHub
source files, issue comments, ChatGPT, screenshots or this README):

- `OPENAI_API_KEY`
- `ANTHROPIC_API_KEY`
- `GEMINI_API_KEY`
- `XAI_API_KEY`
- `DEEPSEEK_API_KEY`

Create matching repository **variables** with model IDs supported by your
individual paid API accounts:

- `OPENAI_MODEL`
- `ANTHROPIC_MODEL`
- `GEMINI_MODEL`
- `XAI_MODEL`
- `DEEPSEEK_MODEL`

Create variable `LLM_RESEARCH_ENABLED` with value `true` **only after**
confirming provider pricing, spend limits, access and permitted data sharing.
To stop new requests, change it to `false` or remove the variable.

An OpenAI ChatGPT subscription does not by itself supply an OpenAI API key
or cover paid API calls. Claude, Gemini, Grok and DeepSeek require their
own provider credentials and suitable model IDs. Each provider's pricing,
models and quota can change. Calls are capped at two upcoming matches per
six-hour heavy pass, up to five providers each, with a 40-second timeout per
call. API usage may incur charges, and a completed match should never be
revisited to manufacture a retrospective AI prediction.

## What is actually frozen

- Original immutable v8/v8_near fixture ID, team IDs, kickoff, input digest,
  model version and score are retained as comparison evidence.
- The same previously collected form, schedule, estimated goals and top scores
  are sent independently to the configured providers.
- Each configured model returns its own 1X2 probabilities and a score suggestion.
  The panel **averages 1X2 probabilities only**, without calling that average
  an exact-score probability.
- A research panel is frozen in `data/llm_research/forecasts/` before kickoff.
  No predictions are created after kickoff or retroactively backfilled.
- Complete five-provider panels are compared on identical verified finished
  fixtures against the original statistical score. Partial panels are labeled
  as such and excluded from this comparison.
- No provider can browse or fetch additional live evidence through these calls.
  Missing injuries, true xG, bookmaker prices and lineups remain missing.
  Connecting licensed and timestamped data sources is a **separate** project.

## Operational status

The Deep Research and Performance pages show the connection state and real
saved panels using `llm_research_ui.py`. GitHub Actions emits
`data/llm_research/status.json` and `evaluation.json`. A missing key,
incompatible model, invalid JSON or provider error is disclosed without
fabricating votes. New output does not imply improved exact-score accuracy.

To verify the installation: GitHub Actions -> "Capture verified pre-match
evidence". The optional step should report `disabled` until settings are
configured. After opting in, look for write-once records and check the real
paired results as finished matches arrive. No model is auto-promoted.
