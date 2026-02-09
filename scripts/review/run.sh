#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

node "$SCRIPT_DIR/scripts/run-review.mjs" --project-root "$PROJECT_ROOT" "$@"
