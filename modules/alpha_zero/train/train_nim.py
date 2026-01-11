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
    piles: List[float] = []
    players: List[float] = []
    pis: List[List[float]] = []
    zs: List[float] = []

    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pile = float(row["pile"])
            player = float(row["player"])
            pi = [
                float(row["pi_take1"]),
                float(row["pi_take2"]),
                float(row["pi_take3"]),
            ]
            z = float(row["z"])
            piles.append(pile)
            players.append(player)
            pis.append(pi)
            zs.append(z)

    piles_np = np.asarray(piles, dtype=np.float32)
    players_np = np.asarray(players, dtype=np.float32)
    pi_np = np.asarray(pis, dtype=np.float32)
    z_np = np.asarray(zs, dtype=np.float32)

    # Normalize features
    max_pile = max(1.0, float(piles_np.max(initial=1.0)))
    pile_feat = piles_np / max_pile
    player_feat = players_np

    x = np.stack([pile_feat, player_feat], axis=1)

    # Normalize pi row-wise
    s = pi_np.sum(axis=1, keepdims=True)
    pi_np = np.where(s > 0.0, pi_np / s, np.full_like(pi_np, 1.0 / 3.0))

    return x, pi_np, z_np


class PolicyValueNet(nn.Module):
    def __init__(self, in_dim: int = 2, hidden: int = 64, actions: int = 3):
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
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--out_dir", default="models")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval_games", type=int, default=10)
    ap.add_argument("--eval_every", type=int, default=1)
    ap.add_argument("--max_pile", type=int, default=12)
    args = ap.parse_args()

    x, pi, z = load_csv(args.data)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = PolicyValueNet(hidden=args.hidden).to(device)

    opt = torch.optim.Adam(net.parameters(), lr=args.lr)

    os.makedirs(args.out_dir, exist_ok=True)

    rng = np.random.default_rng(args.seed)

    x_t = torch.from_numpy(x).to(device)
    pi_t = torch.from_numpy(pi).to(device)
    z_t = torch.from_numpy(z).to(device)

    # simple train/val split
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
            value_loss = F.mse_loss(v, zv)
            return float(policy_loss.item()), float(value_loss.item()), float((policy_loss + value_loss).item())

    def eval_games() -> Tuple[float, float, float]:
        """Run self-play episodes and report (avg_return, avg_len, win_rate) for start player=+1."""
        net.eval()
        wins = 0
        total_return = 0.0
        total_len = 0.0
        with torch.no_grad():
            for _ in range(max(1, args.eval_games)):
                pile = int(rng.integers(1, args.max_pile + 1))
                player = 1
                steps = 0
                # terminal when pile==0; player_to_move loses.
                while pile > 0:
                    # legal actions are 1..min(3,pile)
                    x_in = np.array([[pile / float(args.max_pile), float(player)]], dtype=np.float32)
                    x_t = torch.from_numpy(x_in).to(device)
                    logits, _v = net(x_t)
                    probs = torch.softmax(logits[0, :], dim=-1).cpu().numpy()
                    legal = min(3, pile)
                    p = probs[:legal]
                    s = float(p.sum())
                    if s <= 0.0:
                        p = np.full((legal,), 1.0 / legal, dtype=np.float32)
                    else:
                        p = p / s
                    a = int(rng.choice(np.arange(1, legal + 1), p=p))
                    pile -= a
                    player = -player
                    steps += 1

                # now pile==0, player_to_move loses, winner is -player
                winner = -player
                ret = 1.0 if winner == 1 else -1.0
                if winner == 1:
                    wins += 1
                total_return += ret
                total_len += float(steps)
        n = float(max(1, args.eval_games))
        return total_return / n, total_len / n, wins / n

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
        msg = (
            f"epoch {epoch:03d} | train: policy={total_policy/steps:.4f} value={total_value/steps:.4f} total={total/steps:.4f} | "
            f"val: policy={vp:.4f} value={vv:.4f} total={vt:.4f} | device={device}"
        )

        if args.eval_every > 0 and (epoch % args.eval_every == 0):
            avg_ret, avg_len, win_rate = eval_games()
            msg += f" | eval({args.eval_games}): avg_return={avg_ret:.3f} avg_len={avg_len:.2f} win_rate={win_rate:.2f}"

        print(msg)

    ckpt_path = os.path.join(args.out_dir, "nim_policy_value.pt")
    torch.save(
        {
            "model": net.state_dict(),
            "in_dim": 2,
            "hidden": args.hidden,
            "actions": 3,
        },
        ckpt_path,
    )
    print(f"saved: {ckpt_path}")


if __name__ == "__main__":
    main()
