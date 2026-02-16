from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class ViT_MAE_Encoder(nn.Module):
    """ViT encoder with optional MAE-style patch masking."""

    def __init__(
        self,
        in_channels: int = 7,
        max_grid: int = 15,
        patch_size: int = 3,
        embed_dim: int = 128,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.max_grid = max_grid
        self.num_patches = (max_grid // patch_size) ** 2

        self.patch_embed = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches + 1, embed_dim))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=4,
            dim_feedforward=256,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)

    def forward(self, x: torch.Tensor, mask_ratio: float = 0.0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        batch_size = x.size(0)
        patches = self.patch_embed(x).flatten(2).transpose(1, 2)

        mask_indices = None
        if mask_ratio > 0:
            num_mask = max(int(self.num_patches * mask_ratio), 1)
            noise = torch.rand(batch_size, self.num_patches, device=x.device)
            mask_indices = torch.argsort(noise, dim=1)[:, :num_mask]

            mask = torch.zeros(batch_size, self.num_patches, dtype=torch.bool, device=x.device)
            mask.scatter_(1, mask_indices, True)
            token_bank = self.mask_token.expand(batch_size, self.num_patches, -1)
            patches = torch.where(mask.unsqueeze(-1), token_bank, patches)

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        seq = torch.cat((cls_tokens, patches), dim=1) + self.pos_embed

        features = self.transformer(seq)
        cls_feat = features[:, 0]
        patch_feats = features[:, 1:]
        return cls_feat, patch_feats, mask_indices


class ViTSnakeAgent(nn.Module):
    def __init__(
        self,
        action_dim: int = 4,
        in_channels: int = 7,
        patch_size: int = 3,
        embed_dim: int = 128,
    ) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.encoder = ViT_MAE_Encoder(
            in_channels=in_channels,
            max_grid=15,
            patch_size=patch_size,
            embed_dim=embed_dim,
        )

        self.actor = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
        )
        self.critic = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        patch_dim = in_channels * patch_size * patch_size
        self.mae_decoder = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Linear(128, patch_dim),
        )

        self.icm_forward = nn.Sequential(
            nn.Linear(embed_dim + action_dim, 128),
            nn.ReLU(),
            nn.Linear(128, embed_dim),
        )
        self.icm_inverse = nn.Sequential(
            nn.Linear(embed_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )

    def get_action(
        self, state: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        cls_feat, _, _ = self.encoder(state, mask_ratio=0.0)
        logits = self.actor(cls_feat)
        value = self.critic(cls_feat)

        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action), value, cls_feat

    def compute_intrinsic_reward(
        self,
        cls_feat: torch.Tensor,
        next_cls_feat: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        action_onehot = F.one_hot(action, num_classes=self.action_dim).float()
        pred_next_feat = self.icm_forward(torch.cat([cls_feat, action_onehot], dim=-1))
        return F.mse_loss(pred_next_feat, next_cls_feat.detach(), reduction="none").mean(dim=-1)
