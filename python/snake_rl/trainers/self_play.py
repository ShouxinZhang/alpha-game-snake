from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from snake_rl.mcts.alpha_zero_mcts import AlphaZeroPlanner
from snake_rl.runtime.replay_buffer import Transition
from snake_rl.runtime.rust_env import RustBatchEnv


@dataclass
class SelfPlayMetrics:
    avg_reward: float
    avg_score: float
    avg_steps: float
    policy_mean: tuple[float, float, float, float]
    episodes_finished: int


@torch.no_grad()
def collect_self_play_batch(
    env: RustBatchEnv,
    planner: AlphaZeroPlanner,
    model: torch.nn.Module,
    device: torch.device,
    steps_per_iter: int,
    gamma: float,
) -> tuple[list[Transition], SelfPlayMetrics]:
    obs_np = env.get_obs_tensor()
    transitions: list[Transition] = []
    num_envs = obs_np.shape[0]
    legal_mask = torch.ones((num_envs, 4), device=device)
    policy_acc = torch.zeros(4, device=device)
    policy_count = 0

    episode_reward = np.zeros((num_envs,), dtype=np.float32)
    episode_steps = np.zeros((num_envs,), dtype=np.int32)
    completed_rewards: list[float] = []
    completed_scores: list[float] = []
    completed_steps: list[float] = []
    latest_scores = np.zeros((num_envs,), dtype=np.float32)

    for _ in range(steps_per_iter):
        obs_t = torch.from_numpy(obs_np).to(device)
        policy, actions, _ = planner.plan(model, obs_t, legal_mask, training=True)
        policy_acc += policy.mean(dim=0)
        policy_count += 1

        actions_np = actions.detach().cpu().numpy().astype(np.int64)
        next_obs, rewards, dones, info = env.step(actions_np)
        latest_scores = info["score"].astype(np.float32)
        legal_np = info["legal_actions_mask"]
        legal_mask = torch.from_numpy(legal_np.astype(np.float32)).to(device)
        next_obs_t = torch.from_numpy(next_obs).to(device)
        _, next_values = model(next_obs_t)

        rewards_t = torch.from_numpy(rewards.astype(np.float32)).to(device)
        dones_t = torch.from_numpy(dones.astype(np.float32)).to(device)
        td_target = rewards_t + gamma * next_values.squeeze(-1) * (1.0 - dones_t)

        for env_idx in range(obs_t.shape[0]):
            transitions.append(
                Transition(
                    obs=obs_t[env_idx].detach().cpu(),
                    policy=policy[env_idx].detach().cpu(),
                    value=td_target[env_idx].detach().cpu(),
                )
            )
            episode_reward[env_idx] += float(rewards[env_idx])
            episode_steps[env_idx] += 1

            if bool(dones[env_idx]):
                completed_rewards.append(float(episode_reward[env_idx]))
                completed_scores.append(float(info["score"][env_idx]))
                completed_steps.append(float(episode_steps[env_idx]))
                episode_reward[env_idx] = 0.0
                episode_steps[env_idx] = 0

        if bool(dones.any()):
            _ = env.reset_done()
        obs_np = next_obs

    if completed_rewards:
        avg_reward = float(np.mean(completed_rewards))
        avg_score = float(np.mean(completed_scores))
        avg_steps = float(np.mean(completed_steps))
        episodes_finished = len(completed_rewards)
    else:
        avg_reward = float(np.mean(episode_reward))
        avg_score = float(np.mean(latest_scores))
        avg_steps = float(np.mean(episode_steps))
        episodes_finished = 0

    policy_mean = (policy_acc / max(policy_count, 1)).detach().cpu().tolist()
    return (
        transitions,
        SelfPlayMetrics(
            avg_reward=avg_reward,
            avg_score=avg_score,
            avg_steps=avg_steps,
            policy_mean=(float(policy_mean[0]), float(policy_mean[1]), float(policy_mean[2]), float(policy_mean[3])),
            episodes_finished=episodes_finished,
        ),
    )
