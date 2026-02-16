from __future__ import annotations

import torch
import torch.nn.functional as F


class CurriculumZeroPadWrapper:
    """Progressive curriculum with fixed-size zero-shot input alignment."""

    def __init__(self, max_grid: int = 15, stages: list[int] | None = None) -> None:
        self.max_grid = max_grid
        self.stages = stages[:] if stages is not None else [5, 8, 11, 15]
        if not self.stages:
            raise ValueError("stages must not be empty")
        if self.max_grid not in self.stages:
            self.stages.append(self.max_grid)
            self.stages.sort()
        self.current_stage_idx = 0

    @property
    def current_size(self) -> int:
        return self.stages[self.current_stage_idx]

    def pad_state(self, state: torch.Tensor) -> torch.Tensor:
        if state.dim() != 3:
            raise ValueError("state must have shape (C, H, W)")

        size = self.current_size
        if state.shape[-2] != size or state.shape[-1] != size:
            raise ValueError(
                f"state size mismatch: expected {size}x{size}, got {state.shape[-2]}x{state.shape[-1]}"
            )

        if size == self.max_grid:
            return state.float()

        pad_total = self.max_grid - size
        pad_lt = pad_total // 2
        pad_rb = pad_total - pad_lt
        return F.pad(state.float(), (pad_lt, pad_rb, pad_lt, pad_rb), mode="constant", value=-1.0)

    def pad_batch(self, states: torch.Tensor) -> torch.Tensor:
        if states.dim() != 4:
            raise ValueError("states must have shape (B, C, H, W)")

        size = self.current_size
        if states.shape[-2] != size or states.shape[-1] != size:
            raise ValueError(
                f"batch size mismatch: expected {size}x{size}, got {states.shape[-2]}x{states.shape[-1]}"
            )

        if size == self.max_grid:
            return states.float()

        pad_total = self.max_grid - size
        pad_lt = pad_total // 2
        pad_rb = pad_total - pad_lt
        return F.pad(states.float(), (pad_lt, pad_rb, pad_lt, pad_rb), mode="constant", value=-1.0)

    def step_curriculum(self, recent_avg_length: float) -> bool:
        capacity = float(self.current_size * self.current_size)
        if recent_avg_length >= capacity * 0.6 and self.current_stage_idx < len(self.stages) - 1:
            self.current_stage_idx += 1
            print(f"[curriculum] upgraded to {self.current_size}x{self.current_size}")
            return True
        return False
