import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
import os
import onnx
import onnxruntime

from network import SnakeNet
from dataset import SnakeDataset

def train(data_dir, output_dir, grid_w, grid_h, epochs=1, batch_size=64):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset = SnakeDataset(data_dir)
    if len(dataset) == 0:
        print("No data found, skipping training.")
        return

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = SnakeNet(grid_w, grid_h).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    mse_loss = nn.MSELoss()

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for inputs, policy_target, value_target in dataloader:
            inputs = inputs.view(-1, 4, grid_h, grid_w).to(device) # Reshape here
            policy_target = policy_target.to(device)
            value_target = value_target.to(device).unsqueeze(1) # (N) -> (N, 1)

            optimizer.zero_grad()
            
            p_out, v_out = model(inputs)
            
            # Policy Loss (Cross Entropy logic, but target is probability distribution)
            # -sum(target * log(pred))
            # p_out is logits from network? No, model policy head needs Softmax?
            # network.py policy head ends with Linear, comment says "Softmax applied in loss"
            # So p_out are logits.
            
            log_probs = torch.log_softmax(p_out, dim=1)
            policy_loss = -torch.sum(policy_target * log_probs, dim=1).mean()
            
            value_loss = mse_loss(v_out, value_target)
            
            loss = policy_loss + value_loss
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(dataloader):.4f}")

    # Export to ONNX (legacy mode for compatibility)
    os.makedirs(output_dir, exist_ok=True)
    onnx_path = os.path.join(output_dir, "latest_model.onnx")
    
    model.eval()
    model.cpu()  # Move to CPU for export
    dummy_input = torch.randn(1, 4, grid_h, grid_w)
    torch.onnx.export(
        model, 
        dummy_input, 
        onnx_path, 
        input_names=["input"], 
        output_names=["policy", "value"],
        dynamic_axes={"input": {0: "batch_size"}, "policy": {0: "batch_size"}, "value": {0: "batch_size"}},
        opset_version=14,
        dynamo=False  # Use legacy export
    )
    print(f"Model exported to {onnx_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--output_dir", type=str, default="models")
    parser.add_argument("--width", type=int, default=10)
    parser.add_argument("--height", type=int, default=10)
    args = parser.parse_args()
    
    train(args.data_dir, args.output_dir, args.width, args.height)
