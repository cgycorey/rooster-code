#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PARENT_DIR="$(cd -- "${REPO_ROOT}/.." && pwd)"
SDK_DIR="${PARENT_DIR}/open-agent-sdk-python"
SDK_REPO_URL="https://github.com/cgycorey/open-agent-sdk-python"

need_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    printf 'Error: required command not found: %s\n' "$1" >&2
    exit 1
  fi
}

need_cmd git
need_cmd uv

printf '==> Rooster Code bootstrap\n'
printf '    repo root: %s\n' "$REPO_ROOT"
printf '    sdk path : %s\n' "$SDK_DIR"

if [ ! -d "$SDK_DIR" ]; then
  printf '==> Cloning Open Agent SDK from %s\n' "$SDK_REPO_URL"
  git clone "$SDK_REPO_URL" "$SDK_DIR"
else
  printf '==> Reusing existing SDK checkout at %s\n' "$SDK_DIR"
fi

printf '==> Running uv sync\n'
uv sync --project "$REPO_ROOT"

printf '\nBootstrap complete. Try one of:\n'
printf '  uv run rooster-code --help\n'
printf '  uv run rooster-code ask "Summarize this repository"\n'
printf '  uv run rooster-code chat\n'
