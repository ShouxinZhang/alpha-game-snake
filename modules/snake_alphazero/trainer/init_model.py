#!/usr/bin/env python3
"""
Generate an initial random model for AlphaZero self-play bootstrap.
This model has random weights and serves as the starting point for training.
"""

import torch
import os
import argparse
from network import SnakeNet

def main():
    parser = argparse.ArgumentParser(description="Generate initial ONNX model")
    parser.add_argument("--output", type=str, default="models/latest_model.onnx", help="Output path")
    parser.add_argument("--width", type=int, default=10, help="Grid width")
    parser.add_argument("--height", type=int, default=10, help="Grid height")
    args = parser.parse_args()

    print(f"Generating initial model for {args.width}x{args.height} grid...")
    
    # Create model with random weights
    model = SnakeNet(args.width, args.height)
    model.eval()
    
    # Create output directory
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    
    # Export to ONNX
    dummy_input = torch.randn(1, 4, args.height, args.width)
    torch.onnx.export(
        model,
        dummy_input,
        args.output,
        input_names=["input"],
        output_names=["policy", "value"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "policy": {0: "batch_size"},
            "value": {0: "batch_size"}
        },
        opset_version=14
    )
    
    print(f"Model exported to: {args.output}")
    print(f"  Input shape: (batch, 4, {args.height}, {args.width})")
    print(f"  Policy output: (batch, 4)")
    print(f"  Value output: (batch, 1)")

if __name__ == "__main__":
    main()
