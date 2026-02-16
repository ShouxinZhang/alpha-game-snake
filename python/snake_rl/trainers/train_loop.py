from __future__ import annotations

import argparse
from collections import deque
import json
from pathlib import Path
import random

import numpy as np
import torch
import yaml

from snake_rl.mcts.alpha_zero_mcts import AlphaZeroPlanner, MCTSConfig
from snake_rl.models.vit_policy_value import ViTPolicyValueNet
from snake_rl.runtime.replay_buffer import ReplayBuffer
from snake_rl.runtime.rust_env import RustBatchEnv
from snake_rl.trainers.self_play import collect_self_play_batch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(cfg: dict) -> ViTPolicyValueNet:
    env_cfg = cfg["env"]
    model_cfg = cfg["model"]
    return ViTPolicyValueNet(
        in_channels=int(model_cfg.get("in_channels", 7)),
        board_h=int(env_cfg["board_h"]),
        board_w=int(env_cfg["board_w"]),
        patch_size=int(model_cfg["patch_size"]),
        embed_dim=int(model_cfg["embed_dim"]),
        depth=int(model_cfg["depth"]),
        heads=int(model_cfg["heads"]),
    )


def export_state_dict(model: torch.nn.Module) -> dict:
    if hasattr(model, "_orig_mod"):
        orig = getattr(model, "_orig_mod")
        return orig.state_dict()
    return model.state_dict()


def train(config_path: Path) -> None:
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    runtime_cfg = cfg["runtime"]
    optim_cfg = cfg["optim"]
    mcts_cfg = cfg["mcts"]

    seed = int(cfg.get("seed", 7))
    set_seed(seed)

    use_cuda = runtime_cfg.get("device", "cuda") == "cuda" and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    env = RustBatchEnv(cfg["env"])
    _ = env.reset()

    model = build_model(cfg).to(device)
    if bool(runtime_cfg.get("torch_compile", False)) and hasattr(torch, "compile"):
        model = torch.compile(model)  # type: ignore[assignment]

    planner = AlphaZeroPlanner(
        MCTSConfig(
            simulations=int(mcts_cfg["simulations"]),
            c_puct=float(mcts_cfg["c_puct"]),
            dirichlet_alpha=float(mcts_cfg["dirichlet_alpha"]),
            temperature=float(mcts_cfg["temperature"]),
        )
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(optim_cfg["lr"]),
        weight_decay=float(optim_cfg["weight_decay"]),
    )
    scaler = torch.amp.GradScaler(
        device="cuda" if use_cuda else "cpu",
        enabled=bool(runtime_cfg.get("amp", True) and use_cuda),
    )

    buffer = ReplayBuffer(capacity=int(runtime_cfg["replay_capacity"]))

    artifact_dir = Path(runtime_cfg.get("artifact_dir", "artifacts/metrics"))
    checkpoint_dir = Path(runtime_cfg.get("checkpoint_dir", "checkpoints"))
    artifact_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    latest_metrics = artifact_dir / "latest.jsonl"

    iterations = int(runtime_cfg["iterations"])
    steps_per_iter = int(runtime_cfg["steps_per_iter"])
    batch_size = int(optim_cfg["batch_size"])
    updates_per_iter = int(optim_cfg["updates_per_iter"])
    checkpoint_interval = int(runtime_cfg["checkpoint_interval"])
    gamma = float(optim_cfg.get("gamma", 0.99))
    metrics_window = int(runtime_cfg.get("metrics_window", 100))

    reward_window: deque[float] = deque(maxlen=metrics_window)
    score_window: deque[float] = deque(maxlen=metrics_window)
    steps_window: deque[float] = deque(maxlen=metrics_window)
    loss_window: deque[float] = deque(maxlen=metrics_window)

    for step in range(1, iterations + 1):
        model.eval()
        transitions, metrics = collect_self_play_batch(
            env=env,
            planner=planner,
            model=model,
            device=device,
            steps_per_iter=steps_per_iter,
            gamma=gamma,
        )
        buffer.push_batch(transitions)

        total_loss = 0.0
        update_count = 0
        model.train()
        for _ in range(updates_per_iter):
            if len(buffer) < batch_size:
                break
            obs, target_policy, target_value = buffer.sample(batch_size, device)
            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(
                device_type="cuda" if use_cuda else "cpu",
                enabled=bool(runtime_cfg.get("amp", True) and use_cuda),
            ):
                policy_logits, pred_value = model(obs)
                policy_loss = -(target_policy * torch.log_softmax(policy_logits, dim=-1)).sum(dim=-1).mean()
                value_loss = torch.nn.functional.mse_loss(
                    pred_value.squeeze(-1), target_value.squeeze(-1)
                )
                loss = policy_loss + float(optim_cfg.get("value_coef", 1.0)) * value_loss

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(optim_cfg.get("grad_clip", 1.0)))
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.detach().cpu())
            update_count += 1

        instant_loss = total_loss / max(update_count, 1)
        reward_window.append(metrics.avg_reward)
        score_window.append(metrics.avg_score)
        steps_window.append(metrics.avg_steps)
        loss_window.append(instant_loss)

        rolling_reward = float(sum(reward_window) / max(len(reward_window), 1))
        rolling_score = float(sum(score_window) / max(len(score_window), 1))
        rolling_steps = float(sum(steps_window) / max(len(steps_window), 1))
        rolling_loss = float(sum(loss_window) / max(len(loss_window), 1))

        record = {
            "rolling_avg_reward": rolling_reward,
            "avg_score": rolling_score,
            "avg_steps": rolling_steps,
            "loss": rolling_loss,
            "instant_reward": metrics.avg_reward,
            "instant_score": metrics.avg_score,
            "instant_steps": metrics.avg_steps,
            "instant_loss": instant_loss,
            "iter": step,
            "buffer_size": len(buffer),
            "episodes_finished": metrics.episodes_finished,
            "policy_up": metrics.policy_mean[0],
            "policy_down": metrics.policy_mean[1],
            "policy_left": metrics.policy_mean[2],
            "policy_right": metrics.policy_mean[3],
        }
        with latest_metrics.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        if step % checkpoint_interval == 0:
            ckpt_path = checkpoint_dir / f"alpha_zero_vit_step_{step}.pt"
            torch.save(
                {
                    "model": export_state_dict(model),
                    "optimizer": optimizer.state_dict(),
                    "step": step,
                    "config": cfg,
                },
                ckpt_path,
            )
            print(f"[checkpoint] saved {ckpt_path}")

        print(
            f"[iter={step}] reward={record['rolling_avg_reward']:.4f} "
            f"score={record['avg_score']:.4f} steps={record['avg_steps']:.2f} "
            f"loss={record['loss']:.4f} finished={record['episodes_finished']} "
            f"buffer={record['buffer_size']}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ViT + AlphaZero Snake")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/train/alpha_zero_vit.yaml"),
        help="training config path",
    )
    args = parser.parse_args()
    train(args.config)


if __name__ == "__main__":
    main()
