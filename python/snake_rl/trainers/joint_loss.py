from __future__ import annotations

import torch
import torch.nn.functional as F


def compute_joint_loss(
    agent,
    batch,
    clip_ratio: float = 0.2,
    mae_weight: float = 0.5,
    icm_weight: float = 0.1,
) -> tuple[torch.Tensor, dict[str, float]]:
    """PPO + MAE + ICM composite loss."""

    states, actions, returns, advantages, old_log_probs, next_states = batch
    batch_size = int(states.shape[0])
    if batch_size <= 0:
        raise ValueError("empty batch")

    patch_size = agent.encoder.patch_size

    cls_feat, patch_feats, mask_indices = agent.encoder(states, mask_ratio=0.40)
    with torch.no_grad():
        next_cls_feat, _, _ = agent.encoder(next_states, mask_ratio=0.0)

    logits = agent.actor(cls_feat)
    values = agent.critic(cls_feat).squeeze(-1)
    dist = torch.distributions.Categorical(logits=logits)
    new_log_probs = dist.log_prob(actions)
    entropy = dist.entropy().mean()

    ratio = torch.exp(new_log_probs - old_log_probs)
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * advantages
    actor_loss = -torch.min(surr1, surr2).mean()
    critic_loss = F.mse_loss(values, returns)
    ppo_loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy

    pred_patches = agent.mae_decoder(patch_feats)
    target_patches = F.unfold(states, kernel_size=patch_size, stride=patch_size).transpose(1, 2)

    if mask_indices is None:
        mae_loss = torch.zeros((), device=states.device)
    else:
        batch_index = torch.arange(batch_size, device=states.device).unsqueeze(-1)
        masked_preds = pred_patches[batch_index, mask_indices]
        masked_targets = target_patches[batch_index, mask_indices]
        mae_loss = F.mse_loss(masked_preds, masked_targets)

    action_onehot = F.one_hot(actions, num_classes=agent.action_dim).float()
    pred_next_feat = agent.icm_forward(torch.cat([cls_feat, action_onehot], dim=-1))
    pred_action_logits = agent.icm_inverse(torch.cat([cls_feat, next_cls_feat], dim=-1))

    icm_forward_loss = F.mse_loss(pred_next_feat, next_cls_feat)
    icm_inverse_loss = F.cross_entropy(pred_action_logits, actions)
    icm_loss = icm_forward_loss + icm_inverse_loss

    total_loss = ppo_loss + mae_weight * mae_loss + icm_weight * icm_loss

    metrics = {
        "ppo_loss": float(ppo_loss.detach().cpu()),
        "mae_loss": float(mae_loss.detach().cpu()),
        "icm_loss": float(icm_loss.detach().cpu()),
    }
    return total_loss, metrics
