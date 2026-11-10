# ZicZac Predict AI iPhone PWA

This is an installable **frontend shell** for the existing Streamlit app. It does not copy API keys, replace the prediction engine, or invent match results. The live predictions button opens the existing Streamlit service. An optional embedded iframe may be blocked by the Streamlit host, so it has a clear fallback.

## Publish on GitHub Pages
In GitHub repository Settings → Pages, select **Deploy from a branch**, choose the default branch and **/ (root)** if this folder is moved to root, or use GitHub Actions to publish the `pwa/` directory. GitHub Pages does not offer `/pwa` as a branch publishing folder. A workflow is provided in `.github/workflows/deploy-pwa.yml`.

Expected URL after successful Pages deployment: https://ziczac-hub.github.io/scorepredict-ai/

On iPhone: open the GitHub Pages URL in Safari → Share → Add to Home Screen. The installed app uses a standalone window where supported.

For full **in-app** predictions without leaving the PWA, migrate the prediction engine to a separate authenticated HTTP API and connect this frontend to it. Do not put Sportmonks or other private API keys in browser JavaScript.
