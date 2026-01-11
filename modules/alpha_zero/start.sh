#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

cargo run --bin demo_nim
