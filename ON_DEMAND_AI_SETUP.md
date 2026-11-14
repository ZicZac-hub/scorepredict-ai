# Enable external AI on the purple "Run Deep Research AI" button

## How this differs from the scheduled GitHub research

The GitHub Actions secrets you already added are used by the **scheduled
pre-match collector**. GitHub does not expose those private keys to the
running Streamlit website. This is intentional.

The new selected-match on-demand integration runs **only on the Streamlit
server**, **only after clicking the purple button**, **only before kickoff**,
and **only for an authenticated private app**. No credentials are published,
sent to the browser, committed to the repository, or shown in error details.

## Required owner setup

1. Open [Streamlit Community Cloud](https://share.streamlit.io/), locate
   your existing ZicZac app, then **Manage app → Settings → Secrets**.
2. Add a new strong, unique private-access password as a TOML secret:
   `ZICZAC_ACCESS_PASSWORD = "a unique long password only you know"`.
   Never send this password to ChatGPT. When enabled, you must sign in on
   your iPhone or MacBook before using any app page.
3. For **each provider you want to consult from the purple button**, add
   both its server-side API key and its model ID to the same **Streamlit
   Secrets** editor. Copy values from your private provider dashboards or
   secure password manager, not from screenshots or GitHub source. For
   example, the settings use this form (replace every example value):

   ```toml
   ZICZAC_ACCESS_PASSWORD = "REPLACE-WITH-YOUR-OWN-STRONG-PASSWORD"

   OPENAI_API_KEY = "REPLACE-WITH-YOUR-OPENAI-KEY"
   OPENAI_MODEL = "gpt-4o-mini"

   GEMINI_API_KEY = "REPLACE-WITH-YOUR-GEMINI-KEY"
   GEMINI_MODEL = "REPLACE-WITH-YOUR-AVAILABLE-MODEL-ID"

   ANTHROPIC_API_KEY = "REPLACE-WITH-YOUR-CLAUDE-KEY"
   ANTHROPIC_MODEL = "REPLACE-WITH-YOUR-AVAILABLE-MODEL-ID"
   ```

   Add other providers using the following exact names:

   | Provider | Streamlit secret key | Streamlit model name |
   |---|---|---|
   | OpenAI | `OPENAI_API_KEY` | `OPENAI_MODEL` |
   | Claude | `ANTHROPIC_API_KEY` | `ANTHROPIC_MODEL` |
   | Gemini | `GEMINI_API_KEY` | `GEMINI_MODEL` |
   | Grok | `XAI_API_KEY` | `XAI_MODEL` |
   | DeepSeek | `DEEPSEEK_API_KEY` | `DEEPSEEK_MODEL` |
   | Perplexity | `PERPLEXITY_API_KEY` | `PERPLEXITY_MODEL` |
   | Mistral | `MISTRAL_API_KEY` | `MISTRAL_MODEL` |
   | Qwen | `QWEN_API_KEY` | `QWEN_MODEL` |
   | GroqCloud | `GROQ_API_KEY` | `GROQ_MODEL` |
   | Cohere | `COHERE_API_KEY` | `COHERE_MODEL` |

   Qwen can additionally use the Streamlit secret `QWEN_API_URL`, which
   must point to an authorised Alibaba HTTPS compatible-mode endpoint.
   If omitted, the international default is used.

4. Save settings and restart your Streamlit app if the changes are not
   picked up immediately. Do not paste the keys into GitHub Variables.
5. In the ZicZac **Deep Research** page, choose an upcoming match.
   Make sure **Also consult my connected external AI models** is checked.
   Press **🧠 Run Deep Research AI** once.
6. Review the statistical report plus the private **External AI opinions
   for the selected match** panel. Each provider either returns a
   probability/score opinion or a safe error category. Missing votes are
   not invented.

## Charges and protections

- The integration makes **up to one model request per configured provider**
  for the selected match, concurrently in a bounded batch. Even failed
  requests may use quota or incur charges. Review billing/quotas first.
- A password gate is required to avoid allowing anyone on a publicly
  reachable website to generate paid calls.
- Only selected fixtures at least five minutes before kickoff can trigger
  requests. Models get the same limited historical results and statistical
  evidence available at that time.
- For the same fixture, repeat clicks **within the same Streamlit session**
  show the earlier response, not another paid batch. Starting a new
  browser session can allow another explicit batch; there is no
  cross-session durable charge cap.
- These on-demand responses are **session-only**, not immutable published
  pre-match forecasts. They do not alter previous scores, five-double
  picks, or verified hit-rate metrics. Only the separately frozen research
  collector contributes to prospective accuracy.
- To immediately disable paid calls from the purple button, remove the
  provider keys from **Streamlit Secrets**. Scheduled GitHub analysis
  remains separately governed by `LLM_RESEARCH_ENABLED` in GitHub.
- A 429 error can mean exhausted credits or rate limits. A 400 error can
  mean an unsupported model or invalid provider request. These must be
  resolved by account settings or model compatibility; changing the
  on-demand code cannot purchase API credits.

**Do not publish the app's API keys or password.** The private Streamlit
Secrets editor and GitHub Actions Secrets are different credential stores.
