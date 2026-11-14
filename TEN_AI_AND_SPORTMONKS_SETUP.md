# ZicZac: Ten Optional AI APIs + Sportmonks Correct-Score API

The source code and dashboard display for all services are installed, but
**no third-party connection or payment is activated by this release**.
These are research-only shadow adapters. The original frozen pre-match
scores, five doubles, historical models and verified outcomes are unchanged.

## Supported external AI research

| Research service | GitHub Actions **secret** | GitHub Actions **variable** |
|---|---|---|
| OpenAI (ChatGPT API) | `OPENAI_API_KEY` | `OPENAI_MODEL` |
| Anthropic Claude | `ANTHROPIC_API_KEY` | `ANTHROPIC_MODEL` |
| Google Gemini | `GEMINI_API_KEY` | `GEMINI_MODEL` |
| xAI Grok | `XAI_API_KEY` | `XAI_MODEL` |
| DeepSeek | `DEEPSEEK_API_KEY` | `DEEPSEEK_MODEL` |
| Perplexity Agent API | `PERPLEXITY_API_KEY` | `PERPLEXITY_MODEL` |
| Mistral | `MISTRAL_API_KEY` | `MISTRAL_MODEL` |
| Alibaba Qwen | `QWEN_API_KEY` | `QWEN_MODEL` |
| GroqCloud (Meta Llama) | `GROQ_API_KEY` | `GROQ_MODEL` |
| Cohere | `COHERE_API_KEY` | `COHERE_MODEL` |

Enter the model IDs provided by your actual provider accounts; do not
assume a model name from a marketing brochure is supported by your tier.
GroqCloud is a separate company from **xAI Grok**.

For Qwen, the **optional variable** `QWEN_API_URL` selects the
region/workspace-specific HTTPS compatible-mode chat URL:
`https://WORKSPACE.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1/chat/completions`.
Replace WORKSPACE with your own Alibaba Model Studio Workspace ID.
The old Singapore shared endpoint is used if this variable is absent.
The key's region and endpoint must correspond. Only Alibaba-owned
HTTPS compatible-mode URLs are accepted.

Perplexity uses the **current Agent API** at
`https://api.perplexity.ai/v1/agent`, NOT the sunset legacy Sonar
chat-completions API. Default panel calls do **not** enable external
web searching: all ten models receive the same recorded before-kickoff
source packet to avoid unavailable or retrospectively added match data.

## Sportmonks independent football probabilities

Sportmonks was already connected for fixture discovery via
`THIRD_FOOTBALL_API_KEY` in **Streamlit Secrets**, but that does
NOT mean the separate **Predictions API add-on** is available.

To use the scheduled independent correct-score comparison:
1. Check that your Sportmonks football subscription includes the
   **Predictions add-on**, especially `CORRECT_SCORE_PROBABILITY`.
2. Save your Sportmonks token as a **GitHub Actions secret** named
   `THIRD_FOOTBALL_API_KEY` or `SPORTMONKS_API_TOKEN`. A Streamlit
   secret is not automatically available in GitHub Actions.
3. Create GitHub Actions variable `SPORTMONKS_PREDICTIONS_ENABLED`
   with value `true` after confirming the subscription and API limits.

The adapter matches a Sportmonks fixture ID to an API-Football frozen
forecast only if normalized home and away team names and kickoff
(within five minutes) agree unambiguously. Unknown or ambiguous
matches are skipped. It fetches up to two eligible matches per heavy
pass and stores **unchanged Sportmonks score probabilities** and
their 1X2 probability source, if supplied, before kickoff.
It does not mix provider IDs or silently blend forecasts.

## Enable the optional services only when ready

1. Open GitHub `ZicZac-hub/scorepredict-ai` and go to
   **Settings → Secrets and variables → Actions**.
2. Under **Repository secrets**, create the keys you personally obtained
   from the relevant providers. **Never send them in a chat or screenshot,
   or commit secrets to source code.**
3. Under **Variables**, create the matching model IDs.
4. After reviewing the API prices and setting monthly spending and request
   limits with each provider, set `LLM_RESEARCH_ENABLED` to `true`.
   If not all ten providers are set up, partial votes are shown as partial;
   a full-panel evaluation requires the documented complete set.
5. Run the GitHub Actions workflow **Capture verified pre-match evidence**,
   or allow its scheduled heavy refresh. Open ZicZac **Deep Research** or
   **Performance** to view non-secret status and actually frozen research.
6. To stop billing calls, change `LLM_RESEARCH_ENABLED` and/or
   `SPORTMONKS_PREDICTIONS_ENABLED` back to `false` immediately.
   Stored historical records are not edited.

**Run limits:** One upcoming match per six-hour heavy pass for the
multi-AI panel, max one request to each configured AI for that match.
Sportmonks uses bounded schedule discovery and at most two matched
upcoming fixture predictions per pass. Providers can reject requests
due to plan, quota or model availability. These limits reduce but
do **not eliminate** costs. No API plan or billing is purchased by ZicZac.

## What appears in the website

- A ten-provider status panel, individual saved provider votes, an
  experimental averaged 1X2 outcome opinion and a suggested score.
- A separate Sportmonks correct-score probability panel, with a
  strictly verified same-fixture comparison after the actual match.
- Clear `disabled`, `missing key`, `provider unavailable`, and
  `pending verified results` states.
- Existing original ZicZac scores and same-day frozen five doubles
  are unchanged; **no automatic model promotion**.

There is **no demonstrated improvement in five-double exact-score
accuracy yet**, and market or AI predicted probabilities should not
be presented as verified wins. After enough matched, completed games,
compare observed same-fixture exact hits, log loss/calibration and
verified five-double hit rate before considering a new production model.

Official provider references:
- Perplexity Agent API: https://docs.perplexity.ai/docs/agent-api/quickstart
- Mistral: https://docs.mistral.ai/api
- Alibaba Qwen endpoint regions: https://help.aliyun.com/en/model-studio/base-url
- GroqCloud: https://console.groq.com/docs/api-reference
- Cohere v2: https://docs.cohere.com/reference/chat
- Sportmonks Predictions API: https://docs.sportmonks.com/v3/tutorials-and-guides/tutorials/odds-and-predictions/predictions/probabilities
