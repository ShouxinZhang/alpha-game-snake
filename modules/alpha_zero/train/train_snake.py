import argparse
import csv
import os
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Batch:
    x: torch.Tensor
    pi: torch.Tensor
    z: torch.Tensor


def load_csv(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    feats: List[List[float]] = []
    pis: List[List[float]] = []
    zs: List[float] = []

    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        # infer feature columns
        feature_cols = [c for c in reader.fieldnames or [] if c.startswith("f")]
        feature_cols.sort(key=lambda x: int(x[1:]))

        for row in reader:
            feat = [float(row[c]) for c in feature_cols]
            pi = [float(row["pi0"]), float(row["pi1"]), float(row["pi2"]), float(row["pi3"])]
            z = float(row["z"])
            feats.append(feat)
            pis.append(pi)
            zs.append(z)

    x = np.asarray(feats, dtype=np.float32)
    pi_np = np.asarray(pis, dtype=np.float32)
    z_np = np.asarray(zs, dtype=np.float32)

    # normalize pi row-wise
    s = pi_np.sum(axis=1, keepdims=True)
    pi_np = np.where(s > 0.0, pi_np / s, np.full_like(pi_np, 0.25))

    return x, pi_np, z_np


class PolicyValueNet(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 256, actions: int = 4):
        super().__init__()
        # 8 channels: head, body, food, (unused), 4x direction broadcast
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
            # Remove Tanh to allow unbounded value prediction (regression)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Reshape flat input back to [B, C, H, W]
        x = x.view(-1, self.channels, self.hw, self.hw)
        h = self.conv_block(x)
        logits = self.policy_conv(h)
        # Linear value output
        value = self.value_conv(h).squeeze(-1)
        return logits, value


def iter_batches(x: np.ndarray, pi: np.ndarray, z: np.ndarray, batch_size: int, rng: np.random.Generator):
    n = x.shape[0]
    idx = np.arange(n)
    rng.shuffle(idx)

    for start in range(0, n, batch_size):
        sl = idx[start : start + batch_size]
        xb = torch.from_numpy(x[sl])
        pib = torch.from_numpy(pi[sl])
        zb = torch.from_numpy(z[sl])
        yield Batch(xb, pib, zb)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--out_dir", default="models")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    x, pi, z = load_csv(args.data)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = PolicyValueNet(in_dim=x.shape[1], hidden=args.hidden, actions=4).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)

    os.makedirs(args.out_dir, exist_ok=True)

    rng = np.random.default_rng(args.seed)

    # train/val split
    n = x.shape[0]
    n_val = max(1, int(0.1 * n))
    perm = rng.permutation(n)
    val_idx = perm[:n_val]
    tr_idx = perm[n_val:]

    x_tr, pi_tr, z_tr = x[tr_idx], pi[tr_idx], z[tr_idx]
    x_val, pi_val, z_val = x[val_idx], pi[val_idx], z[val_idx]

    def eval_loss() -> Tuple[float, float, float]:
        net.eval()
        with torch.no_grad():
            xv = torch.from_numpy(x_val).to(device)
            piv = torch.from_numpy(pi_val).to(device)
            zv = torch.from_numpy(z_val).to(device)
            logits, v = net(xv)
            logp = F.log_softmax(logits, dim=-1)
            policy_loss = -(piv * logp).sum(dim=-1).mean()
            # z is raw return; v is linear prediction. No tanh squash.
            value_loss = F.mse_loss(v, zv)
            return float(policy_loss.item()), float(value_loss.item()), float((policy_loss + value_loss).item())

    for epoch in range(1, args.epochs + 1):
        net.train()
        total_policy = 0.0
        total_value = 0.0
        total = 0.0
        steps = 0

        for batch in iter_batches(x_tr, pi_tr, z_tr, args.batch, rng):
            xb = batch.x.to(device)
            pib = batch.pi.to(device)
            zb = batch.z.to(device)

            logits, v = net(xb)
            logp = F.log_softmax(logits, dim=-1)
            policy_loss = -(pib * logp).sum(dim=-1).mean()
            # No tanh!
            value_loss = F.mse_loss(v, zb)
            loss = policy_loss + value_loss

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            total_policy += float(policy_loss.item())
            total_value += float(value_loss.item())
            total += float(loss.item())
            steps += 1

        vp, vv, vt = eval_loss()
        print(
            f"epoch {epoch:03d} | train: policy={total_policy/steps:.4f} value={total_value/steps:.4f} total={total/steps:.4f} | "
            f"val: policy={vp:.4f} value={vv:.4f} total={vt:.4f} | device={device}"
        )

    ckpt_path = os.path.join(args.out_dir, "snake_policy_value.pt")
    torch.save(
        {
            "model": net.state_dict(),
            "in_dim": int(x.shape[1]),
            "hidden": args.hidden,
            "actions": 4,
        },
        ckpt_path,
    )
    print(f"saved: {ckpt_path}")


if __name__ == "__main__":
    main()
