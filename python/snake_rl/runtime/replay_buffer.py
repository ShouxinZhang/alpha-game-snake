from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque

import torch


@dataclass
class Transition:
    obs: torch.Tensor
    policy: torch.Tensor
    value: torch.Tensor


class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self._capacity = int(capacity)
        self._buffer: Deque[Transition] = deque(maxlen=self._capacity)

    def __len__(self) -> int:
        return len(self._buffer)

    def push_batch(self, transitions: list[Transition]) -> None:
        self._buffer.extend(transitions)

    def sample(self, batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        indices = torch.randint(0, len(self._buffer), (batch_size,)).tolist()
        batch = [self._buffer[idx] for idx in indices]
        obs = torch.stack([item.obs for item in batch], dim=0).to(device)
        policy = torch.stack([item.policy for item in batch], dim=0).to(device)
        value = torch.stack([item.value for item in batch], dim=0).to(device)
        return obs, policy, value
