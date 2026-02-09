#!/bin/bash
# AlphaZero Training Loop
# Usage: ./run_training.sh [num_iterations]

set -e

NUM_ITERATIONS=${1:-10}
GRID_SIZE=10
DATA_DIR="data/self_play"
MODELS_DIR="models"
GAMES_PER_ITER=100

echo "========================================"
echo "  Snake AlphaZero Training Loop"
echo "========================================"
echo "Iterations: $NUM_ITERATIONS"
echo "Grid Size: ${GRID_SIZE}x${GRID_SIZE}"
echo "Games per iteration: $GAMES_PER_ITER"
echo ""

cd "$(dirname "$0")"

# Step 0: Initialize model if not exists
if [ ! -f "$MODELS_DIR/latest_model.onnx" ]; then
    echo "[Step 0] Generating initial random model..."
    python3 trainer/init_model.py --output "$MODELS_DIR/latest_model.onnx" --width $GRID_SIZE --height $GRID_SIZE
    echo ""
fi

# Main training loop
for i in $(seq 1 $NUM_ITERATIONS); do
    echo "========================================"
    echo "  Iteration $i / $NUM_ITERATIONS"
    echo "========================================"
    
    # Step 1: Self-play (Rust)
    echo "[Step 1] Running self-play ($GAMES_PER_ITER games)..."
    rm -rf "$DATA_DIR"
    cargo run --release --bin self_play -- "$MODELS_DIR/latest_model.onnx" "$DATA_DIR" $GAMES_PER_ITER
    echo ""
    
    # Step 2: Train (Python)
    echo "[Step 2] Training on generated data..."
    python3 trainer/train.py --data_dir "$DATA_DIR" --output_dir "$MODELS_DIR" --width $GRID_SIZE --height $GRID_SIZE
    echo ""
    
    echo "Iteration $i complete!"
    echo ""
done

echo "========================================"
echo "  Training Complete!"
echo "========================================"
echo "Final model: $MODELS_DIR/latest_model.onnx"
