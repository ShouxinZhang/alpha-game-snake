import argparse
import os
from typing import Tuple

import torch
import torch.nn as nn


class PolicyValueNet(nn.Module):
    def __init__(self, in_dim: int, hidden: int, actions: int):
        super().__init__()
        self.channels = 8
        self.hw = int((in_dim / self.channels) ** 0.5)

        self.conv_block = nn.Sequential(
            nn.Conv2d(self.channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        # Policy head
        self.policy_conv = nn.Sequential(
            nn.Conv2d(64, 2, kernel_size=1),
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2 * self.hw * self.hw, actions)
        )

        # Value head
        self.value_conv = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(1 * self.hw * self.hw, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = x.view(-1, self.channels, self.hw, self.hw)
        h = self.conv_block(x)
        logits = self.policy_conv(h)
        value = self.value_conv(h).squeeze(-1)
        return logits, value


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu")
    in_dim = int(ckpt.get("in_dim", 2))
    hidden = int(ckpt.get("hidden", 64))
    actions = int(ckpt.get("actions", 3))

    net = PolicyValueNet(in_dim=in_dim, hidden=hidden, actions=actions)
    net.load_state_dict(ckpt["model"], strict=True)
    net.eval()

    dummy = torch.zeros((1, in_dim), dtype=torch.float32)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    torch.onnx.export(
        net,
        dummy,
        args.out,
        input_names=["x"],
        output_names=["policy_logits", "value"],
        opset_version=12,
    )
    print(f"exported: {args.out}")


if __name__ == "__main__":
    main()
