from __future__ import annotations

import torch
from torch import nn


class ViTPolicyValueNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        board_h: int,
        board_w: int,
        patch_size: int,
        embed_dim: int,
        depth: int,
        heads: int,
    ) -> None:
        super().__init__()
        if board_h % patch_size != 0 or board_w % patch_size != 0:
            raise ValueError("board size must be divisible by patch_size")

        self.board_h = board_h
        self.board_w = board_w
        self.patch_h = board_h // patch_size
        self.patch_w = board_w // patch_size
        self.patch_count = self.patch_h * self.patch_w

        self.patch_embed = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_count, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)

        self.policy_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 4),
        )
        self.value_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 1),
            nn.Tanh(),
        )

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = self.patch_embed(obs)
        tokens = tokens.flatten(2).transpose(1, 2)
        tokens = tokens + self.pos_embed
        encoded = self.encoder(tokens)
        pooled = self.norm(encoded.mean(dim=1))
        policy_logits = self.policy_head(pooled)
        value = self.value_head(pooled)
        return policy_logits, value

    @torch.no_grad()
    def attention_map(self, obs: torch.Tensor) -> torch.Tensor:
        tokens = self.patch_embed(obs)
        tokens = tokens.flatten(2).transpose(1, 2)
        norm = torch.norm(tokens, dim=-1)
        norm = norm / (norm.sum(dim=1, keepdim=True) + 1e-8)
        return norm.view(obs.shape[0], self.patch_h, self.patch_w)
