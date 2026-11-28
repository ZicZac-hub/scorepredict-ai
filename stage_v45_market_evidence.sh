#!/usr/bin/env bash
# Only stage V45 market artifacts. Treat published receipts and quota use as write-once.
set -euo pipefail
shopt -s nullglob
market_requests=(data/v24_market_catalog/requests/*/v45_target_*.json)
if ((${#market_requests[@]})); then
  git add -- "${market_requests[@]}"
fi
for path in data/v45_market_capture/versions data/v45_market_capture/status.json data/v45_market_trial/protocol.json data/v45_market_trial/trials data/v45_market_trial/status.json; do
  if [[ -e "$path" ]]; then
    git add -- "$path"
  fi
done
if [[ -n "$(git diff --cached --name-only --diff-filter=DMRTUXB -- \
  data/v45_market_capture/versions data/v45_market_trial/protocol.json data/v45_market_trial/trials \
  ':(glob)data/v24_market_catalog/requests/*/v45_target_*.json')" ]]; then
  echo 'SAFE STOP: existing V45 market evidence, trials or quota records changed'
  exit 1
fi
