#!/usr/bin/env bash
set -euo pipefail

# Minimal selfplay -> train -> export loop (Snake 5x5)
# Usage:
#   ./train/loop_snake_5x5.sh 2

ROUNDS="${1:-2}"
GAMES_PER_ROUND="${GAMES_PER_ROUND:-200}"
SIM="${SIM:-200}"
TEMP="${TEMP:-1.0}"
THREADS="${THREADS:-32}"

W="${W:-5}"
H="${H:-5}"

# Get the absolute path to the project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ALPHA_ZERO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT_ROOT="$(cd "$ALPHA_ZERO_DIR/../.." && pwd)"

# Always run from alpha_zero directory
cd "$ALPHA_ZERO_DIR"

DATA_DIR="train/data"
MODEL_DIR="train/models"

# Use virtualenv python if available
PYTHON_EXE="$PROJECT_ROOT/.venv/bin/python"
if [[ ! -f "$PYTHON_EXE" ]]; then
  PYTHON_EXE="python"
fi

mkdir -p "$DATA_DIR" "$MODEL_DIR"

ONNX_PATH="$MODEL_DIR/snake_policy_value.onnx"
PT_PATH="$MODEL_DIR/snake_policy_value.pt"

IN_DIM=$((8 * W * H))

for r in $(seq 1 "$ROUNDS"); do
  echo "=== round $r/$ROUNDS ==="

  if [[ -f "$ONNX_PATH" ]]; then
    echo "selfplay with ONNX priors/value: $ONNX_PATH"
    cargo run --bin selfplay_snake -- \
      --out "$DATA_DIR/snake_${W}x${H}_round_${r}.csv" \
      --games "$GAMES_PER_ROUND" --sim "$SIM" --temp "$TEMP" --threads "$THREADS" --w "$W" --h "$H" \
      --onnx "$ONNX_PATH"
  else
    echo "selfplay with uniform priors/value"
    cargo run --bin selfplay_snake -- \
      --out "$DATA_DIR/snake_${W}x${H}_round_${r}.csv" \
      --games "$GAMES_PER_ROUND" --sim "$SIM" --temp "$TEMP" --threads "$THREADS" --w "$W" --h "$H"
  fi

  "$PYTHON_EXE" -u train/train_snake.py \
    --data "$DATA_DIR/snake_${W}x${H}_round_${r}.csv" \
    --epochs 20 --batch 256 --lr 1e-3 \
    --out_dir "$MODEL_DIR" \
    --hidden 256

  "$PYTHON_EXE" -u train/export_onnx.py --ckpt "$PT_PATH" --out "$ONNX_PATH"

  echo "round $r done: $ONNX_PATH"
  echo

done
