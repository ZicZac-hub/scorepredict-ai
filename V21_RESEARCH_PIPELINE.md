# V21 — Verified research and five-double accuracy upgrade

**Status:** implemented as a research-only challenger. V8 and all original
pre-match, manual, and automatic five-double choices remain immutable.
**No demonstrated increase in exact-score win rate yet.**

## 1. Real source collection (API-Football, bounded and opt-in)

- `v21_provider_evidence.py` calls only the existing authenticated
  API-Football endpoints `/fixtures/lineups`, `/injuries`,
  `/odds` and `/fixtures/statistics` for a few verified, already-finished
  historical fixtures. It does not scrape every football website.
- Each packet records provider/fixture/team identity, kickoff, actual
  capture time, SHA-256 fingerprint, which sources were returned and which
  were not. Injury reports and XI names are used as factual context;
  no untrained numeric player-weight is invented.
- Fixture statistics are accepted as provider-reported **historical expected
  goals only if both teams' Expected Goals fields exist**. API coverage
  varies by league and may not include actual xG. Model-estimated goals and
  provider-reported xG are never silently conflated.
- Pre-match 1X2 market prices require a named bookmaker, all three outcomes,
  a real source update timestamp no later than observation, no older than
  12 hours and a plausible overround. They are de-vigged but NEVER labelled
  as a three-bookmaker consensus or a fully archived HTTP proof.
- Server-side query cap: 3 fixture endpoints plus at most 2 historical
  fixture-statistics requests per team (currently scheduled at **one**
  historical match per team). Cached league history queries are separate.
- The optional scheduled step looks for eligible previously frozen double
  **legs 8–35 minutes before kickoff**, including the later leg after the
  first begins. It needs the owner to set GitHub Actions variable
  `V21_EVIDENCE_ENABLED=true`, and an existing usable
  `API_FOOTBALL_KEY` secret. It is deliberately disabled unless opted in.
  Requests consume provider quota; no additional subscription is purchased.
- When the public repository is used, scheduled evidence metadata may be
  committed to `data/v21_sources/`. Do not put private identifiers, API
  secrets, or raw vendor response bodies in this directory.
- The interactive AI Five-Double Studio also has a **separate checkbox**
  explicitly authorising two-match API evidence. It requires Private
  Access and a Streamlit `API_FOOTBALL_KEY`. Valid packets are stored
  append-only in private PostgreSQL if a working `DATABASE_URL` is
  configured. Without it, the site warns that packets are session-only.
- Confirmed starting XIs are often unavailable earlier than 20–40 minutes
  before kickoff; missing data means unknown, not a healthy full squad.
  If the feed lacks xG, a licensed provider with auditable timestamps
  must be connected separately. This update does **not** assert that such a
  subscription exists.

## 2. Goals and high-score testing

- `v21_goal_model_comparison.py` compares **unchanged V8** with independent
  Poisson marginals, fixed negative-binomial (r=3) marginals, and fixed
  bivariate Poisson covariance (0.12). These last three are *untuned*
  statistical hypotheses, not newer proven models.
- Tests measure top exact score, top five, goal MAE, exact-score log loss,
  1X2 Brier score, over-2.5 Brier and observed vs predicted 4+ goal
  frequency. Explicitly record probabilities of **2–2, 3–1 and 3–2**.
- `walk_forward(...)` excludes each fixture's result from its pre-kickoff
  data, with a 3-hour conservative delay for prior finished matches. If a
  genuine independent past-data archive is unavailable, the app cannot claim
  that a model has passed historical backtesting.
- A retrospective mathematical transformation of existing forecasts is
  **NOT** a newly frozen prospective prediction and may not be used to claim
  a model improvement.

## 3. Genuine multi-AI evidence

- V21 provider packets enter the **identical input prompt** sent to each
  selected AI API during an owner-approved manual research request.
- Earlier `llm_research_panel.py` collects source-matched frozen AI votes
  when `LLM_RESEARCH_ENABLED=true`, valid keys/models are present, and
  the user explicitly chooses `LLM_RESEARCH_MAX_MATCHES=2` for both legs.
  Those providers receive a shared frozen statistical packet; they do **not**
  independently browse the entire web. GitHub Actions secrets and Streamlit
  Secrets are separate.
- **V21 research scoring assigns zero performance weight to AI votes** until
  an independently verified, source-matched prospective comparison
  demonstrates a measurable advantage. Advice and errors remain visible.
  Existing legacy modes retain their prior, disclosed experimental behavior.

## 4. Strong pair selection and abstention

- The manual AI Studio now offers **V21 AI Auto Select**. It checks all
  eligible same-Sydney-day pairs, needs valid original full matrices and at
  least 8 previous matches/team with 4 at the relevant venue, computes the
  original and V20 form-model top-five joint coverage, and ranks by their
  **minimum** coverage. It requires >=5.5% estimated coverage.
- A 5.5% gate is an unvalidated *model probability* heuristic, NOT 5.5%
  observed success or a guarantee. When not qualified, the app clearly
  abstains; original manual mode remains available.
- `v21_prospective.py` separately compares every eligible frozen same-day
  pair. It chooses a first, ranked research pair per day and freezes the
  same pair's five combinations from each original/Poisson/NB/bivariate
  model **before both kickoffs**, without rewriting V8.

## 5. Immutable results tracking

- The website already stores first pre-match manual choices by method in
  private PostgreSQL when owner login and DATABASE_URL work. Without these,
  an interactive prediction cannot be guaranteed durable and the UI warns.
- V21 scheduled research forecasts and pairs persist under
  `data/v21_prospective/` and are evaluated automatically from the existing
  independently verified, provider/team/kickoff-matched result ledger.
- All V21 methods are graded on **the same actual two matches**, with
  separate wins, misses and pending. No pending result is marked as a loss.
  Every record is write-once. Model promotion is disabled, regardless of
  retrospective performance. Require hundreds of qualifying future
  forecasts and calibrated statistical tests before considering promotion.

## Verification

- Regression tests:
  `tests/test_v21_evidence_models.py`,
  `tests/test_v21_prospective.py`,
  `tests/test_v21_pair_selector.py`.
- GitHub Actions: `Prediction integrity tests` and scheduled
  `Capture verified pre-match evidence`.
- No API keys, paid subscriptions, licensed xG, full lineup/player-impact
  parameters or third-party browsing access were added by this commit.

**Important:** Exact-score five doubles have intrinsically small coverage.
The goal is demonstrably better, auditable prediction quality, not a
fabricated 70% success rate.
