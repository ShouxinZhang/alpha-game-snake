#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PYTHON="$SCRIPT_DIR/.venv/bin/python"
TRAIN_SCRIPT="$SCRIPT_DIR/snake_rl_parallel.py"
LOG_FILE="$SCRIPT_DIR/train.log"
TRAIN_PID=""

cleanup() {
    local exit_code=$?
    trap - EXIT INT TERM

    if [[ -n "${TRAIN_PID:-}" ]]; then
        pkill -P "$TRAIN_PID" 2>/dev/null || true
        kill "$TRAIN_PID" 2>/dev/null || true
        wait "$TRAIN_PID" 2>/dev/null || true
    fi

    pkill -P $$ tail 2>/dev/null || true
    pkill -P $$ zenity 2>/dev/null || true

    echo "[restart.sh] Training stopped."
    exit "$exit_code"
}
trap cleanup EXIT INT TERM

if [[ ! -x "$VENV_PYTHON" ]]; then
    echo "[restart.sh] Missing Python executable: $VENV_PYTHON" >&2
    exit 1
fi

if [[ ! -f "$TRAIN_SCRIPT" ]]; then
    echo "[restart.sh] Training script not found: $TRAIN_SCRIPT" >&2
    exit 1
fi

# Kill any existing training process
pkill -f "$TRAIN_SCRIPT" 2>/dev/null || true
sleep 0.5

# Clear log
: > "$LOG_FILE"

# Start training in background and tee output to log.
# Use process substitution so TRAIN_PID is the Python process PID.
"$VENV_PYTHON" "$TRAIN_SCRIPT" "$@" > >(tee "$LOG_FILE") 2>&1 &
TRAIN_PID=$!

echo "[restart.sh] Training started (PID=$TRAIN_PID)"

if command -v zenity >/dev/null 2>&1; then
    # Avoid exiting early when zenity closes and tail gets SIGPIPE.
    set +e
    tail --pid="$TRAIN_PID" -f "$LOG_FILE" 2>/dev/null | zenity --text-info \
        --title="🐍 Snake RL Training" \
        --width=720 --height=480 \
        --auto-scroll \
        --font="monospace" \
        --ok-label="Stop Training" \
        2>/dev/null
    set -e
    echo "[restart.sh] GUI closed, stopping training..."
else
    echo "[restart.sh] zenity not found, following log in terminal."
    tail --pid="$TRAIN_PID" -f "$LOG_FILE" 2>/dev/null || true
fi
