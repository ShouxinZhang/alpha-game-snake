#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

TRAIN_CONFIG="configs/train/alpha_zero_vit.yaml"
RUN_TRAIN=1
RUN_UI=1
RUN_PROBE=0

usage() {
  cat <<'USAGE'
Usage: bash restart.sh [options]

Options:
  --config <path>   Training config path (default: configs/train/alpha_zero_vit.yaml)
  --train-only      Restart only training process
  --ui-only         Restart only UI process
  --skip-probe      Skip hardware probe before restart
  --probe           Force run hardware probe before restart
  -h, --help        Show this help message
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      TRAIN_CONFIG="${2:-}"
      shift 2
      ;;
    --train-only)
      RUN_TRAIN=1
      RUN_UI=0
      shift
      ;;
    --ui-only)
      RUN_TRAIN=0
      RUN_UI=1
      shift
      ;;
    --skip-probe)
      RUN_PROBE=0
      shift
      ;;
    --probe)
      RUN_PROBE=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[restart] Unknown option: $1"
      usage
      exit 1
      ;;
  esac
done

if [[ ! -x ".venv/bin/python" ]]; then
  echo "[restart] Missing Python runtime: .venv/bin/python"
  exit 1
fi

if [[ "$RUN_TRAIN" -eq 1 && ! -f "$TRAIN_CONFIG" ]]; then
  echo "[restart] Training config not found: $TRAIN_CONFIG"
  exit 1
fi

mkdir -p logs artifacts/metrics checkpoints

stop_pattern() {
  local pattern="$1"
  local label="$2"

  if pgrep -f "$pattern" >/dev/null 2>&1; then
    echo "[restart] Stopping $label ..."
    pkill -f "$pattern" || true
    sleep 1
  fi
}

stop_pattern "snake_rl.trainers.train_loop" "trainer"
stop_pattern "target/debug/snake-ui" "snake-ui"
stop_pattern "cargo run -p snake-ui" "snake-ui (cargo)"

if [[ "$RUN_PROBE" -eq 1 ]]; then
  echo "[restart] Running hardware probe ..."
  .venv/bin/python scripts/system_probe/probe_hardware.py
fi

TRAIN_PID=""
UI_PID=""

if [[ "$RUN_TRAIN" -eq 1 ]]; then
  echo "[restart] Starting trainer ..."
  PYTHONPATH=python nohup .venv/bin/python -m snake_rl.trainers.train_loop \
    --config "$TRAIN_CONFIG" > logs/train.log 2>&1 &
  TRAIN_PID="$!"
fi

if [[ "$RUN_UI" -eq 1 ]]; then
  echo "[restart] Starting UI ..."
  nohup cargo run -p snake-ui > logs/ui.log 2>&1 &
  UI_PID="$!"
fi

echo "[restart] Done"
if [[ -n "$TRAIN_PID" ]]; then
  echo "[restart] trainer pid=$TRAIN_PID log=logs/train.log"
fi
if [[ -n "$UI_PID" ]]; then
  echo "[restart] ui pid=$UI_PID log=logs/ui.log"
fi

echo "[restart] Tip: tail -f logs/train.log logs/ui.log"
