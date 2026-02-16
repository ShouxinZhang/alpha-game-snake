from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import random

import numpy as np
import torch
import yaml

from snake_rl.models.vit_mae_agent import ViTSnakeAgent
from snake_rl.runtime.rust_env import RustBatchEnv
from snake_rl.trainers.curriculum import CurriculumZeroPadWrapper
from snake_rl.trainers.joint_loss import compute_joint_loss


@dataclass
class RolloutMetrics:
    avg_env_reward: float
    avg_intrinsic_reward: float
    avg_score: float
    recent_avg_length: float
    episodes_finished: int


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_env_config(base_env_cfg: dict, board_size: int, seed: int) -> dict:
    cfg = dict(base_env_cfg)
    cfg["board_w"] = int(board_size)
    cfg["board_h"] = int(board_size)
    cfg["seed"] = int(seed)
    return cfg


def compute_gae(
    rewards: torch.Tensor,
    dones: torch.Tensor,
    values: torch.Tensor,
    last_values: torch.Tensor,
    gamma: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    steps, num_envs = rewards.shape
    advantages = torch.zeros_like(rewards)
    gae = torch.zeros((num_envs,), dtype=values.dtype, device=values.device)

    for t in reversed(range(steps)):
        next_value = last_values if t == steps - 1 else values[t + 1]
        not_done = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * not_done - values[t]
        gae = delta + gamma * gae_lambda * not_done * gae
        advantages[t] = gae

    returns = advantages + values
    return returns, advantages


def collect_rollout(
    agent: ViTSnakeAgent,
    env: RustBatchEnv,
    curriculum: CurriculumZeroPadWrapper,
    device: torch.device,
    steps_per_iter: int,
    gamma: float,
    gae_lambda: float,
    intrinsic_coef: float,
) -> tuple[tuple[torch.Tensor, ...], RolloutMetrics]:
    obs_np = env.get_obs_tensor()
    num_envs = int(obs_np.shape[0])

    states: list[torch.Tensor] = []
    actions: list[torch.Tensor] = []
    old_log_probs: list[torch.Tensor] = []
    values: list[torch.Tensor] = []
    rewards: list[torch.Tensor] = []
    dones: list[torch.Tensor] = []
    next_states: list[torch.Tensor] = []

    env_reward_trace: list[float] = []
    intrinsic_trace: list[float] = []
    score_trace: list[float] = []
    episode_finished = 0

    for _ in range(steps_per_iter):
        obs_t = torch.from_numpy(obs_np).to(device=device, dtype=torch.float32)
        obs_pad = curriculum.pad_batch(obs_t)

        with torch.no_grad():
            action, log_prob, value, cls_feat = agent.get_action(obs_pad)

        next_obs_np, env_rewards_np, dones_np, info = env.step(action.detach().cpu().numpy().astype(np.int64))

        next_obs_t = torch.from_numpy(next_obs_np).to(device=device, dtype=torch.float32)
        next_obs_pad = curriculum.pad_batch(next_obs_t)

        with torch.no_grad():
            next_cls_feat, _, _ = agent.encoder(next_obs_pad, mask_ratio=0.0)
            intrinsic_reward = agent.compute_intrinsic_reward(cls_feat, next_cls_feat, action)

        env_reward_t = torch.from_numpy(env_rewards_np.astype(np.float32)).to(device)
        total_reward = env_reward_t + intrinsic_coef * intrinsic_reward
        done_t = torch.from_numpy(dones_np.astype(np.float32)).to(device)

        states.append(obs_pad.detach())
        actions.append(action.detach())
        old_log_probs.append(log_prob.detach())
        values.append(value.squeeze(-1).detach())
        rewards.append(total_reward.detach())
        dones.append(done_t.detach())
        next_states.append(next_obs_pad.detach())

        env_reward_trace.append(float(env_rewards_np.mean()))
        intrinsic_trace.append(float(intrinsic_reward.detach().mean().cpu()))
        score_trace.append(float(info["score"].mean()))
        episode_finished += int(dones_np.sum())

        if bool(dones_np.any()):
            _ = env.reset_done()
        obs_np = next_obs_np

    obs_t = torch.from_numpy(obs_np).to(device=device, dtype=torch.float32)
    obs_pad = curriculum.pad_batch(obs_t)
    with torch.no_grad():
        next_values = agent.critic(agent.encoder(obs_pad, mask_ratio=0.0)[0]).squeeze(-1)

    rewards_t = torch.stack(rewards, dim=0)
    dones_t = torch.stack(dones, dim=0)
    values_t = torch.stack(values, dim=0)

    returns_t, advantages_t = compute_gae(
        rewards=rewards_t,
        dones=dones_t,
        values=values_t,
        last_values=next_values,
        gamma=gamma,
        gae_lambda=gae_lambda,
    )

    advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std(unbiased=False) + 1e-8)

    states_flat = torch.cat(states, dim=0)
    next_states_flat = torch.cat(next_states, dim=0)
    actions_flat = torch.cat(actions, dim=0)
    old_log_probs_flat = torch.cat(old_log_probs, dim=0)
    returns_flat = returns_t.reshape(-1)
    advantages_flat = advantages_t.reshape(-1)

    batch = (
        states_flat,
        actions_flat,
        returns_flat,
        advantages_flat,
        old_log_probs_flat,
        next_states_flat,
    )

    avg_score = float(np.mean(score_trace)) if score_trace else 0.0
    metrics = RolloutMetrics(
        avg_env_reward=float(np.mean(env_reward_trace)) if env_reward_trace else 0.0,
        avg_intrinsic_reward=float(np.mean(intrinsic_trace)) if intrinsic_trace else 0.0,
        avg_score=avg_score,
        recent_avg_length=avg_score + 3.0,
        episodes_finished=episode_finished,
    )
    return batch, metrics


def train(config_path: Path) -> None:
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    seed = int(cfg.get("seed", 7))
    set_seed(seed)

    runtime_cfg = cfg["runtime"]
    env_cfg = cfg["env"]
    model_cfg = cfg["model"]
    optim_cfg = cfg["optim"]
    curriculum_cfg = cfg["curriculum"]

    use_cuda = bool(runtime_cfg.get("device", "cuda") == "cuda" and torch.cuda.is_available())
    device = torch.device("cuda" if use_cuda else "cpu")

    curriculum = CurriculumZeroPadWrapper(
        max_grid=int(curriculum_cfg.get("max_grid", 15)),
        stages=[int(v) for v in curriculum_cfg.get("stages", [5, 8, 11, 15])],
    )

    current_env_cfg = build_env_config(env_cfg, curriculum.current_size, seed)
    env = RustBatchEnv(current_env_cfg)
    obs = env.reset()

    in_channels = int(model_cfg.get("in_channels", obs.shape[1]))
    if in_channels != int(obs.shape[1]):
        raise ValueError(
            f"model.in_channels={in_channels} but env provides {obs.shape[1]} channels; please align config"
        )

    agent = ViTSnakeAgent(
        action_dim=4,
        in_channels=in_channels,
        patch_size=int(model_cfg.get("patch_size", 3)),
        embed_dim=int(model_cfg.get("embed_dim", 128)),
    ).to(device)

    optimizer = torch.optim.AdamW(
        agent.parameters(),
        lr=float(optim_cfg.get("lr", 3e-4)),
        weight_decay=float(optim_cfg.get("weight_decay", 1e-4)),
    )

    artifact_dir = Path(runtime_cfg.get("artifact_dir", "artifacts/metrics"))
    checkpoint_dir = Path(runtime_cfg.get("checkpoint_dir", "checkpoints"))
    artifact_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    latest_metrics = artifact_dir / "latest.jsonl"

    iterations = int(runtime_cfg.get("iterations", 300))
    steps_per_iter = int(runtime_cfg.get("steps_per_iter", 128))
    ppo_epochs = int(optim_cfg.get("ppo_epochs", 2))
    gamma = float(optim_cfg.get("gamma", 0.99))
    gae_lambda = float(optim_cfg.get("gae_lambda", 0.95))
    grad_clip = float(optim_cfg.get("grad_clip", 1.0))
    intrinsic_coef = float(optim_cfg.get("intrinsic_coef", 0.05))

    clip_ratio = float(optim_cfg.get("clip_ratio", 0.2))
    mae_weight = float(optim_cfg.get("mae_weight", 0.5))
    icm_weight = float(optim_cfg.get("icm_weight", 0.1))

    checkpoint_interval = int(runtime_cfg.get("checkpoint_interval", 50))

    for step in range(1, iterations + 1):
        agent.train()
        batch, rollout_metrics = collect_rollout(
            agent=agent,
            env=env,
            curriculum=curriculum,
            device=device,
            steps_per_iter=steps_per_iter,
            gamma=gamma,
            gae_lambda=gae_lambda,
            intrinsic_coef=intrinsic_coef,
        )

        total_loss = torch.zeros((), device=device)
        loss_metrics = {"ppo_loss": 0.0, "mae_loss": 0.0, "icm_loss": 0.0}
        for _ in range(ppo_epochs):
            loss, metrics = compute_joint_loss(
                agent=agent,
                batch=batch,
                clip_ratio=clip_ratio,
                mae_weight=mae_weight,
                icm_weight=icm_weight,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.parameters(), grad_clip)
            optimizer.step()
            total_loss = total_loss + loss.detach()
            for key, value in metrics.items():
                loss_metrics[key] += float(value)

        for key in loss_metrics:
            loss_metrics[key] /= max(ppo_epochs, 1)

        record = {
            "iter": step,
            "stage_size": curriculum.current_size,
            "mode": env.mode,
            "total_loss": float((total_loss / max(ppo_epochs, 1)).cpu()),
            "ppo_loss": loss_metrics["ppo_loss"],
            "mae_loss": loss_metrics["mae_loss"],
            "icm_loss": loss_metrics["icm_loss"],
            "avg_env_reward": rollout_metrics.avg_env_reward,
            "avg_intrinsic_reward": rollout_metrics.avg_intrinsic_reward,
            "avg_score": rollout_metrics.avg_score,
            "recent_avg_length": rollout_metrics.recent_avg_length,
            "episodes_finished": rollout_metrics.episodes_finished,
        }

        with latest_metrics.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        if step % checkpoint_interval == 0:
            ckpt_path = checkpoint_dir / f"ppo_vit_mae_icm_step_{step}.pt"
            torch.save(
                {
                    "model": agent.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "step": step,
                    "config": cfg,
                },
                ckpt_path,
            )
            print(f"[checkpoint] saved {ckpt_path}")

        print(
            f"[iter={step}] stage={curriculum.current_size} mode={env.mode} "
            f"env_reward={record['avg_env_reward']:.4f} intrinsic={record['avg_intrinsic_reward']:.4f} "
            f"score={record['avg_score']:.4f} loss={record['total_loss']:.4f}"
        )

        if curriculum.step_curriculum(rollout_metrics.recent_avg_length):
            current_env_cfg = build_env_config(env_cfg, curriculum.current_size, seed + step)
            env = RustBatchEnv(current_env_cfg)
            _ = env.reset()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PPO + ViT/MAE + ICM Snake agent")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/train/vit_mae_icm_ppo.yaml"),
        help="training config path",
    )
    args = parser.parse_args()
    train(args.config)


if __name__ == "__main__":
    main()
