#!/usr/bin/env bash
# Push only the committed result-backfill change. Never stage, delete, or
# autostash the original checkout's untracked or unstaged research evidence.
set -euo pipefail
repo="$(git rev-parse --show-toplevel)"
work="$(mktemp -d)"
checkout="$work/publish"
cleanup() {
  git -C "$repo" worktree remove --force "$checkout" >/dev/null 2>&1 || true
  rmdir "$work" >/dev/null 2>&1 || true
}
trap cleanup EXIT
git -C "$repo" worktree add --detach "$checkout" HEAD
for attempt in 1 2 3 4; do
  git -C "$checkout" fetch origin main
  if ! git -C "$checkout" rebase origin/main; then
    git -C "$checkout" rebase --abort || true
    echo "::error::Conflicting committed evidence requires manual review"
    exit 1
  fi
  if [ -n "$(git -C "$checkout" status --porcelain)" ]; then
    echo "::error::Unexpected publication worktree changes require review"
    exit 1
  fi
  if git -C "$checkout" push origin HEAD:main; then
    echo "Verified results published"
    exit 0
  fi
  sleep 3
done
echo "::error::Verified results could not be published after four attempts"
exit 1
