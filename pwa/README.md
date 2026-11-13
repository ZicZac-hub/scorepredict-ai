# ZicZac Predict AI iPhone PWA

This is the installable **Matchday Studio 2.0 launch page** for the existing Streamlit app. It does not copy API keys, replace the prediction engine, or invent match results. Every navigation item and feature card opens the matching section of the live website. Auto Select and Manual Select open `?view=doubles&mode=auto` and `?view=doubles&mode=manual`. Navigation uses the same browser window on iPhone and MacBook.

The service worker refreshes the shell from the network and removes older ZicZac shell caches. An offline copy is only a launch page; it contains no live forecasts or private records. Account access and prediction history remain on the existing website.

## Publish on GitHub Pages
In GitHub repository Settings → Pages, select **Deploy from a branch**, choose the default branch and **/ (root)** if this folder is moved to root, or use GitHub Actions to publish the `pwa/` directory. GitHub Pages does not offer `/pwa` as a branch publishing folder. A workflow is provided in `.github/workflows/deploy-pwa.yml`.

Expected URL after successful Pages deployment: https://ziczac-hub.github.io/scorepredict-ai/

On iPhone: open the GitHub Pages URL in Safari → Share → Add to Home Screen. The installed app uses a standalone window where supported.

For full **in-app** predictions without leaving the PWA, migrate the prediction engine to a separate authenticated HTTP API and connect this frontend to it. Do not put Sportmonks or other private API keys in browser JavaScript.
