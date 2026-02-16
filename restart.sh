#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

TRAIN_CONFIG="configs/train/vit_mae_icm_ppo.yaml"
METRICS_PATH="artifacts/metrics/latest.jsonl"
RUN_TRAIN=1
RUN_UI=1
BUILD_RUST_EXT=1

usage() {
  cat <<'USAGE'
Usage: bash restart.sh [options]

Options:
  --config <path>         Training config path (default: configs/train/vit_mae_icm_ppo.yaml)
  --metrics <path>        Metrics jsonl path for GUI (default: artifacts/metrics/latest.jsonl)
  --train-only            Restart only training process
  --ui-only               Restart only UI process
  --skip-build-ext        Skip building snake-core python extension
  --build-ext             Force building snake-core python extension
  -h, --help              Show this help message
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      TRAIN_CONFIG="${2:-}"
      shift 2
      ;;
    --metrics)
      METRICS_PATH="${2:-}"
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
    --skip-build-ext)
      BUILD_RUST_EXT=0
      shift
      ;;
    --build-ext)
      BUILD_RUST_EXT=1
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

if [[ "$BUILD_RUST_EXT" -eq 1 && "$RUN_TRAIN" -eq 1 ]]; then
  echo "[restart] Building snake-core python extension ..."
  if cargo build -p snake-core --features python > logs/build_snake_core.log 2>&1; then
    echo "[restart] Extension build completed (log=logs/build_snake_core.log)"
  else
    echo "[restart] Extension build failed, training will fall back to python env implementation."
    echo "[restart] Check log: logs/build_snake_core.log"
  fi
fi

TRAIN_PID=""
UI_PID=""

# 清理函数：退出时杀死所有子进程
cleanup() {
  echo ""
  if [[ -n "$TRAIN_PID" ]] && kill -0 "$TRAIN_PID" 2>/dev/null; then
    echo "[restart] Stopping trainer (pid=$TRAIN_PID) ..."
    kill "$TRAIN_PID" 2>/dev/null || true
    wait "$TRAIN_PID" 2>/dev/null || true
  fi
  if [[ -n "$UI_PID" ]] && kill -0 "$UI_PID" 2>/dev/null; then
    echo "[restart] Stopping UI (pid=$UI_PID) ..."
    kill "$UI_PID" 2>/dev/null || true
    wait "$UI_PID" 2>/dev/null || true
  fi
  echo "[restart] All processes stopped."
}

if [[ "$RUN_TRAIN" -eq 1 ]]; then
  echo "[restart] Starting trainer ..."
  PYTHONPATH=python .venv/bin/python -m snake_rl.trainers.train_loop \
    --config "$TRAIN_CONFIG" > logs/train.log 2>&1 &
  TRAIN_PID="$!"
fi

if [[ "$RUN_UI" -eq 1 ]]; then
  echo "[restart] Starting UI ..."
  SNAKE_METRICS_PATH="$METRICS_PATH" cargo run -p snake-ui > logs/ui.log 2>&1 &
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

# 同时运行 UI + trainer 时，前台等待 UI，UI 退出后自动停止 trainer
if [[ "$RUN_TRAIN" -eq 1 && "$RUN_UI" -eq 1 ]]; then
  trap cleanup EXIT INT TERM
  echo "[restart] Waiting for UI to exit (close GUI window to stop all) ..."
  wait "$UI_PID" 2>/dev/null || true
  UI_PID=""  # 已退出，cleanup 中不再重复 kill
  # cleanup 会在 EXIT trap 中自动执行，杀死 trainer
elif [[ "$RUN_UI" -eq 1 ]]; then
  trap cleanup EXIT INT TERM
  echo "[restart] Waiting for UI to exit ..."
  wait "$UI_PID" 2>/dev/null || true
elif [[ "$RUN_TRAIN" -eq 1 ]]; then
  trap cleanup EXIT INT TERM
  echo "[restart] Waiting for trainer (Ctrl+C to stop) ..."
  wait "$TRAIN_PID" 2>/dev/null || true
fi
