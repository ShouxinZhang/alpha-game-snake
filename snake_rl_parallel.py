#!/usr/bin/env python3
import argparse
import json
import os
import platform
import random
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_cpu_model() -> str:
    model = platform.processor()
    if model:
        return model
    cpuinfo_path = "/proc/cpuinfo"
    if os.path.exists(cpuinfo_path):
        with open(cpuinfo_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if "model name" in line:
                    return line.split(":", 1)[1].strip()
    return "unknown"


def collect_hardware_info(device: torch.device) -> Dict:
    info = {
        "timestamp": datetime.now().isoformat(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": get_cpu_model(),
        "cpu_logical_cores": os.cpu_count(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "selected_device": str(device),
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
    }
    if torch.cuda.is_available():
        gpus = []
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            gpus.append(
                {
                    "id": i,
                    "name": p.name,
                    "total_memory_gb": round(p.total_memory / (1024 ** 3), 3),
                    "multi_processor_count": p.multi_processor_count,
                    "major": p.major,
                    "minor": p.minor,
                }
            )
        info["gpu_count"] = torch.cuda.device_count()
        info["gpus"] = gpus
    else:
        info["gpu_count"] = 0
        info["gpus"] = []
    return info


def save_hardware_info(path: str, info: Dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)


@dataclass
class TrainConfig:
    grid_size: int = 6
    num_envs: int = 1024
    rollout_steps: int = 64
    updates: int = 500
    ppo_epochs: int = 4
    mini_batch_size: int = 4096
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    learning_rate: float = 3e-4
    step_penalty: float = -0.01
    food_reward: float = 1.0
    death_penalty: float = -1.0
    max_episode_steps: int = 200
    seed: int = 42
    log_interval: int = 10
    save_interval: int = 100
    output_model: str = "snake_ppo.pt"
    output_hwinfo: str = "hardware_info.json"
    force_cpu: bool = False
    use_compile: bool = True


class SnakeVectorEnv:
    def __init__(self, cfg: TrainConfig, device: torch.device):
        self.grid = cfg.grid_size
        self.num_envs = cfg.num_envs
        self.max_cells = self.grid * self.grid
        self.max_steps = cfg.max_episode_steps
        self.step_penalty = cfg.step_penalty
        self.food_reward = cfg.food_reward
        self.death_penalty = cfg.death_penalty
        self.device = device

        self.body = torch.zeros((self.num_envs, self.max_cells, 2), dtype=torch.long, device=device)
        self.lengths = torch.full((self.num_envs,), 2, dtype=torch.long, device=device)
        self.dirs = torch.full((self.num_envs,), 3, dtype=torch.long, device=device)
        self.food = torch.zeros((self.num_envs, 2), dtype=torch.long, device=device)
        self.steps = torch.zeros((self.num_envs,), dtype=torch.long, device=device)
        self.ep_returns = torch.zeros((self.num_envs,), dtype=torch.float32, device=device)
        self.ep_lengths = torch.zeros((self.num_envs,), dtype=torch.long, device=device)
        self.ep_max_snake_len = torch.full((self.num_envs,), 2, dtype=torch.long, device=device)

        self.deltas = torch.tensor([[-1, 0], [1, 0], [0, -1], [0, 1]], dtype=torch.long, device=device)
        self.opposite = torch.tensor([1, 0, 3, 2], dtype=torch.long, device=device)

        self.reset()

    def reset(self, env_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        if env_ids.numel() == 0:
            return self._observe()

        r = self.grid // 2
        c = self.grid // 2

        self.body[env_ids] = 0
        self.lengths[env_ids] = 2
        self.dirs[env_ids] = 3
        self.steps[env_ids] = 0
        self.ep_returns[env_ids] = 0.0
        self.ep_lengths[env_ids] = 0
        self.ep_max_snake_len[env_ids] = 2
        self.body[env_ids, 0] = torch.tensor([r, c - 1], device=self.device)
        self.body[env_ids, 1] = torch.tensor([r, c], device=self.device)

        self._spawn_food(env_ids)
        return self._observe()

    def _spawn_food(self, env_ids: torch.Tensor) -> None:
        # 6x6 网格很小，这里按环境循环放食物，逻辑更稳健。
        for env in env_ids.tolist():
            length = int(self.lengths[env].item())
            occ = torch.zeros((self.grid, self.grid), dtype=torch.bool, device=self.device)
            segs = self.body[env, :length]
            occ[segs[:, 0], segs[:, 1]] = True
            empty = torch.nonzero(~occ, as_tuple=False)
            if empty.numel() == 0:
                self.food[env] = torch.tensor([0, 0], device=self.device)
            else:
                idx = torch.randint(0, empty.shape[0], (1,), device=self.device)
                self.food[env] = empty[idx].squeeze(0)

    def _observe(self) -> torch.Tensor:
        obs = torch.zeros((self.num_envs, 3, self.grid, self.grid), dtype=torch.float32, device=self.device)
        arange_env = torch.arange(self.num_envs, device=self.device)
        cell_idx = torch.arange(self.max_cells, device=self.device).unsqueeze(0).expand(self.num_envs, self.max_cells)
        valid_mask = cell_idx < self.lengths.unsqueeze(1)
        env_idx = arange_env.unsqueeze(1).expand(self.num_envs, self.max_cells)[valid_mask]
        segs = self.body[valid_mask]

        obs[env_idx, 0, segs[:, 0], segs[:, 1]] = 1.0
        heads = self.body[arange_env, self.lengths - 1]
        obs[arange_env, 1, heads[:, 0], heads[:, 1]] = 1.0
        obs[arange_env, 0, heads[:, 0], heads[:, 1]] = 0.0
        obs[arange_env, 2, self.food[:, 0], self.food[:, 1]] = 1.0
        return obs

    def step(self, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        actions = actions.long()
        reverse = actions == self.opposite[self.dirs]
        actions = torch.where(reverse, self.dirs, actions)

        env_idx = torch.arange(self.num_envs, device=self.device)
        heads = self.body[env_idx, self.lengths - 1]
        new_heads = heads + self.deltas[actions]

        out_of_bounds = (
            (new_heads[:, 0] < 0)
            | (new_heads[:, 0] >= self.grid)
            | (new_heads[:, 1] < 0)
            | (new_heads[:, 1] >= self.grid)
        )
        will_eat = (new_heads == self.food).all(dim=1)

        cell_idx = torch.arange(self.max_cells, device=self.device).unsqueeze(0).expand(self.num_envs, self.max_cells)
        in_body = cell_idx < self.lengths.unsqueeze(1)
        body_if_move = in_body & (cell_idx >= 1)
        body_if_eat = in_body
        check_mask = torch.where(will_eat.unsqueeze(1), body_if_eat, body_if_move)

        hit_body = (((self.body == new_heads.unsqueeze(1)).all(dim=2)) & check_mask).any(dim=1)
        done = out_of_bounds | hit_body | (self.steps + 1 >= self.max_steps)

        rewards = torch.full((self.num_envs,), self.step_penalty, dtype=torch.float32, device=self.device)
        rewards = rewards + will_eat.float() * self.food_reward
        rewards[done] = self.death_penalty

        alive = ~done
        move_alive = alive & (~will_eat)
        eat_alive = alive & will_eat

        self.dirs[alive] = actions[alive]
        self.steps[alive] += 1

        move_ids = torch.nonzero(move_alive, as_tuple=False).squeeze(-1)
        if move_ids.numel() > 0:
            self.body[move_ids, :-1] = self.body[move_ids, 1:].clone()
            tail_new = self.lengths[move_ids] - 1
            self.body[move_ids, tail_new] = new_heads[move_ids]

        eat_ids = torch.nonzero(eat_alive, as_tuple=False).squeeze(-1)
        if eat_ids.numel() > 0:
            insert_idx = self.lengths[eat_ids]
            self.body[eat_ids, insert_idx] = new_heads[eat_ids]
            self.lengths[eat_ids] += 1

            full = self.lengths[eat_ids] >= self.max_cells
            if full.any():
                full_ids = eat_ids[full]
                done[full_ids] = True
                rewards[full_ids] = self.food_reward * 2.0

            not_full_ids = eat_ids[~full]
            if not_full_ids.numel() > 0:
                self._spawn_food(not_full_ids)

        self.ep_returns += rewards
        self.ep_lengths += 1
        self.ep_max_snake_len = torch.max(self.ep_max_snake_len, self.lengths)

        done_ids = torch.nonzero(done, as_tuple=False).squeeze(-1)
        info = {
            "episode_returns": torch.empty(0, dtype=torch.float32),
            "episode_lengths": torch.empty(0, dtype=torch.long),
            "episode_snake_lens": torch.empty(0, dtype=torch.long),
        }
        if done_ids.numel() > 0:
            info["episode_returns"] = self.ep_returns[done_ids].detach().cpu()
            info["episode_lengths"] = self.ep_lengths[done_ids].detach().cpu()
            info["episode_snake_lens"] = self.ep_max_snake_len[done_ids].detach().cpu()
            self.reset(done_ids)

        return self._observe(), rewards, done, info


class ActorCritic(nn.Module):
    def __init__(self, grid: int):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * grid * grid, 256),
            nn.ReLU(),
        )
        self.policy = nn.Linear(256, 4)
        self.value = nn.Linear(256, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.backbone(x)
        return self.policy(h), self.value(h).squeeze(-1)


def auto_num_envs(device: torch.device) -> int:
    cpu_cores = os.cpu_count() or 8
    if device.type == "cuda":
        return min(4096, max(1024, cpu_cores * 16))
    return min(2048, max(128, cpu_cores * 8))


def train(cfg: TrainConfig) -> None:
    if cfg.force_cpu:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.set_num_threads(max(1, os.cpu_count() or 1))
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    seed_everything(cfg.seed)
    hw_info = collect_hardware_info(device)
    save_hardware_info(cfg.output_hwinfo, hw_info)

    env = SnakeVectorEnv(cfg, device=device)
    model = ActorCritic(cfg.grid_size).to(device)
    if cfg.use_compile and hasattr(torch, "compile"):
        compiled_model = torch.compile(model, dynamic=True)
    else:
        compiled_model = model

    optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate, eps=1e-5)
    use_amp = False  # AMP disabled: model is tiny and FP16 cuBLAS can fail on newer GPUs
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    obs = env._observe()
    episode_returns_window = deque(maxlen=200)
    episode_lengths_window = deque(maxlen=200)
    episode_snake_lens_window = deque(maxlen=200)
    start_time = time.time()

    num_steps = cfg.rollout_steps
    num_envs = cfg.num_envs
    batch_size = num_steps * num_envs
    mini_batch = min(cfg.mini_batch_size, batch_size)

    print(f"[Device] {device}")
    print(f"[Parallel] num_envs={cfg.num_envs}, rollout_steps={cfg.rollout_steps}, batch_size={batch_size}")
    print(f"[HW Info Saved] {cfg.output_hwinfo}")

    for update in range(1, cfg.updates + 1):
        obs_buf = torch.zeros((num_steps, num_envs, 3, cfg.grid_size, cfg.grid_size), dtype=torch.float32, device=device)
        actions_buf = torch.zeros((num_steps, num_envs), dtype=torch.long, device=device)
        logprob_buf = torch.zeros((num_steps, num_envs), dtype=torch.float32, device=device)
        rewards_buf = torch.zeros((num_steps, num_envs), dtype=torch.float32, device=device)
        dones_buf = torch.zeros((num_steps, num_envs), dtype=torch.float32, device=device)
        values_buf = torch.zeros((num_steps, num_envs), dtype=torch.float32, device=device)

        for t in range(num_steps):
            obs_buf[t] = obs
            with torch.no_grad():
                logits, value = compiled_model(obs)
                dist = Categorical(logits=logits)
                action = dist.sample()
                logprob = dist.log_prob(action)

            next_obs, reward, done, info = env.step(action)

            actions_buf[t] = action
            logprob_buf[t] = logprob
            rewards_buf[t] = reward
            dones_buf[t] = done.float()
            values_buf[t] = value
            obs = next_obs

            if info["episode_returns"].numel() > 0:
                episode_returns_window.extend(info["episode_returns"].tolist())
                episode_lengths_window.extend(info["episode_lengths"].tolist())
                episode_snake_lens_window.extend(info["episode_snake_lens"].tolist())

        with torch.no_grad():
            _, last_value = compiled_model(obs)

        advantages = torch.zeros_like(rewards_buf, device=device)
        last_gae = torch.zeros((num_envs,), dtype=torch.float32, device=device)
        for t in reversed(range(num_steps)):
            next_non_terminal = 1.0 - dones_buf[t]
            next_value = last_value if t == num_steps - 1 else values_buf[t + 1]
            delta = rewards_buf[t] + cfg.gamma * next_value * next_non_terminal - values_buf[t]
            last_gae = delta + cfg.gamma * cfg.gae_lambda * next_non_terminal * last_gae
            advantages[t] = last_gae
        returns = advantages + values_buf

        b_obs = obs_buf.reshape((-1, 3, cfg.grid_size, cfg.grid_size))
        b_actions = actions_buf.reshape(-1)
        b_logprobs = logprob_buf.reshape(-1)
        b_adv = advantages.reshape(-1)
        b_returns = returns.reshape(-1)

        b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)

        idx = torch.arange(batch_size, device=device)
        for _ in range(cfg.ppo_epochs):
            perm = idx[torch.randperm(batch_size, device=device)]
            for start in range(0, batch_size, mini_batch):
                mb_idx = perm[start : start + mini_batch]
                with torch.amp.autocast(device.type, enabled=use_amp):
                    new_logits, new_values = compiled_model(b_obs[mb_idx])
                    new_dist = Categorical(logits=new_logits)
                    new_logprobs = new_dist.log_prob(b_actions[mb_idx])
                    entropy = new_dist.entropy().mean()

                    ratio = (new_logprobs - b_logprobs[mb_idx]).exp()
                    pg_loss_1 = -b_adv[mb_idx] * ratio
                    pg_loss_2 = -b_adv[mb_idx] * torch.clamp(ratio, 1.0 - cfg.clip_coef, 1.0 + cfg.clip_coef)
                    pg_loss = torch.max(pg_loss_1, pg_loss_2).mean()
                    v_loss = 0.5 * (new_values - b_returns[mb_idx]).pow(2).mean()
                    loss = pg_loss + cfg.vf_coef * v_loss - cfg.ent_coef * entropy

                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()

        if update % cfg.log_interval == 0 or update == 1:
            elapsed = time.time() - start_time
            fps = int((update * batch_size) / max(elapsed, 1e-6))
            mean_return = float(sum(episode_returns_window) / len(episode_returns_window)) if episode_returns_window else 0.0
            mean_len = float(sum(episode_lengths_window) / len(episode_lengths_window)) if episode_lengths_window else 0.0
            mean_snake_len = float(sum(episode_snake_lens_window) / len(episode_snake_lens_window)) if episode_snake_lens_window else 0.0
            print(
                f"[{update:04d}/{cfg.updates}] "
                f"fps={fps} "
                f"mean_return={mean_return:.3f} "
                f"mean_len={mean_len:.2f} "
                f"mean_snake_len={mean_snake_len:.2f}"
            )

        if update % cfg.save_interval == 0 or update == cfg.updates:
            ckpt = {
                "model_state_dict": model.state_dict(),
                "config": cfg.__dict__,
                "update": update,
                "device": str(device),
            }
            torch.save(ckpt, cfg.output_model)

    print(f"[Model Saved] {cfg.output_model}")


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="6x6 Snake RL (PPO) with massive parallel environments.")
    parser.add_argument("--grid-size", type=int, default=6, help="Grid size. Requirement is 6.")
    parser.add_argument("--num-envs", type=int, default=0, help="Parallel env count. 0 means auto.")
    parser.add_argument("--rollout-steps", type=int, default=64)
    parser.add_argument("--updates", type=int, default=500)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--mini-batch-size", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force-cpu", action="store_true")
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--output-model", type=str, default="snake_ppo.pt")
    parser.add_argument("--output-hwinfo", type=str, default="hardware_info.json")
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--save-interval", type=int, default=100)

    args = parser.parse_args()

    cfg = TrainConfig(
        grid_size=args.grid_size,
        rollout_steps=args.rollout_steps,
        updates=args.updates,
        ppo_epochs=args.ppo_epochs,
        mini_batch_size=args.mini_batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        output_model=args.output_model,
        output_hwinfo=args.output_hwinfo,
        force_cpu=args.force_cpu,
        use_compile=(not args.no_compile),
        log_interval=args.log_interval,
        save_interval=args.save_interval,
    )

    if cfg.grid_size != 6:
        raise ValueError("This script is fixed for a 6x6 grid. Please use --grid-size 6.")

    device = torch.device("cpu") if cfg.force_cpu else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg.num_envs = args.num_envs if args.num_envs > 0 else auto_num_envs(device)
    cfg.max_episode_steps = cfg.grid_size * cfg.grid_size * 4
    return cfg


if __name__ == "__main__":
    config = parse_args()
    train(config)
