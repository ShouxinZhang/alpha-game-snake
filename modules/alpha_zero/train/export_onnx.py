import argparse
import os
from typing import Tuple

import torch
import torch.nn as nn


class PolicyValueNet(nn.Module):
    def __init__(self, in_dim: int, hidden: int, actions: int):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(hidden, actions)
        self.value_head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(x)
        logits = self.policy_head(h)
        value = torch.tanh(self.value_head(h)).squeeze(-1)
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
        opset_version=18,
    )
    print(f"exported: {args.out}")


if __name__ == "__main__":
    main()
