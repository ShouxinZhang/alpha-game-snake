from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class MCTSConfig:
    simulations: int
    c_puct: float
    dirichlet_alpha: float
    temperature: float


class AlphaZeroPlanner:
    def __init__(self, config: MCTSConfig) -> None:
        self.config = config

    @torch.no_grad()
    def plan(
        self,
        model: torch.nn.Module,
        obs: torch.Tensor,
        legal_actions_mask: torch.Tensor,
        training: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, values = model(obs)
        masked_logits = logits.masked_fill(legal_actions_mask <= 0, -1e9)
        policy = torch.softmax(masked_logits / max(self.config.temperature, 1e-3), dim=-1)

        if training:
            noise = torch.distributions.Dirichlet(
                torch.full((4,), self.config.dirichlet_alpha, device=obs.device)
            ).sample((obs.shape[0],))
            policy = 0.75 * policy + 0.25 * noise
            policy = policy * legal_actions_mask
            policy = policy / (policy.sum(dim=-1, keepdim=True) + 1e-8)

        for _ in range(max(self.config.simulations // 16, 1)):
            sharpen = torch.pow(policy + 1e-8, 1.0 + self.config.c_puct * 0.05)
            sharpen = sharpen * legal_actions_mask
            policy = sharpen / (sharpen.sum(dim=-1, keepdim=True) + 1e-8)

        actions = torch.multinomial(policy, num_samples=1).squeeze(-1)
        return policy, actions, values.squeeze(-1)
