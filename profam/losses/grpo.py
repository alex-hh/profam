"""GRPO (PPO-clipped) policy-gradient loss + per-token log-prob helper.

Adapted from the inline implementation in pref_opt_profam/scripts/run_grpo_profam.py.
The per-token log-prob helper uses ProFam's frozen-context KV-cache pattern
(see profam.scoring.cache_context / score_variants_differentiable).
"""

from typing import Tuple

import torch
import torch.nn.functional as F

from profam.models.utils import InputAwareDynamicCache


def compute_per_token_log_probs(
    model,
    past_key_values,
    completion_ids: torch.Tensor,
    sub_batch_size: int = 4,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute per-token log-probabilities for each variant under the current policy.

    Args:
        model: A ProFam Lightning module exposing ``.model`` and ``.tokenizer``.
        past_key_values: Cached KV states for the (frozen) conditioning context.
        completion_ids: (1, N, L) tensor of tokenized variant sequences.
        sub_batch_size: Number of variants per forward pass (memory control).

    Returns:
        log_probs: (N, L-1) per-token log-probs (zero-padded to L-1).
        valid_mask: (N, L-1) bool, True where the token is non-padding.
    """
    pad_token_id = model.tokenizer.pad_token_id
    N = completion_ids.shape[1]
    L = completion_ids.shape[2]
    all_lps = []
    all_masks = []

    for batch_start in range(0, N, sub_batch_size):
        batch_end = min(batch_start + sub_batch_size, N)
        this_ids = completion_ids[0, batch_start:batch_end]
        actual_bs = this_ids.shape[0]

        # Trim trailing padding for efficiency
        mask = this_ids != pad_token_id
        indices = torch.arange(L, device=this_ids.device).expand(actual_bs, -1)
        indices = torch.where(mask, indices, torch.zeros_like(indices))
        max_len = indices.max().item() + 1
        this_ids = this_ids[:, :max_len]

        cache = InputAwareDynamicCache.from_legacy_cache(past_key_values)
        cache.batch_repeat_interleave(actual_bs)

        outputs = model.model(
            input_ids=this_ids,
            past_key_values=cache,
            use_cache=False,
        )

        logits = outputs.logits[:, :-1, :]
        targets = this_ids[:, 1:]
        log_probs = -F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            reduction="none",
        ).reshape(targets.shape)
        valid_mask = targets != pad_token_id

        if max_len - 1 < L - 1:
            pad_size = (L - 1) - (max_len - 1)
            log_probs = F.pad(log_probs, (0, pad_size), value=0.0)
            valid_mask = F.pad(valid_mask, (0, pad_size), value=False)

        all_lps.append(log_probs)
        all_masks.append(valid_mask)

    return torch.cat(all_lps), torch.cat(all_masks)


def grpo_loss(
    new_lps: torch.Tensor,
    old_lps: torch.Tensor,
    valid_mask: torch.Tensor,
    advantages: torch.Tensor,
    clip_ratio: float = 0.2,
) -> Tuple[torch.Tensor, dict]:
    """PPO-clipped GRPO loss.

    Args:
        new_lps: (N, L) per-token log-probs under the current policy.
        old_lps: (N, L) per-token log-probs under the reference policy (detached).
        valid_mask: (N, L) bool mask of valid (non-padding) tokens.
        advantages: (N,) per-sequence advantages (e.g. group-normalised rewards).
        clip_ratio: PPO clipping epsilon.

    Returns:
        loss: scalar tensor with gradients.
        metrics: dict with diagnostics (grpo_loss, clip_fraction, mean_ratio, mean_advantage).
    """
    log_ratio = new_lps - old_lps
    ratio = torch.exp(log_ratio)
    clipped_ratio = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio)

    adv = advantages.unsqueeze(1)
    surr1 = ratio * adv
    surr2 = clipped_ratio * adv
    per_token_obj = torch.min(surr1, surr2)

    num_valid = valid_mask.float().sum(dim=1).clamp(min=1)
    per_seq_obj = (per_token_obj * valid_mask.float()).sum(dim=1) / num_valid
    loss = -per_seq_obj.mean()

    with torch.no_grad():
        ratio_valid = ratio.detach()[valid_mask]
        if ratio_valid.numel() > 0:
            clip_frac = (
                (
                    (ratio_valid < 1.0 - clip_ratio)
                    | (ratio_valid > 1.0 + clip_ratio)
                )
                .float()
                .mean()
                .item()
            )
            mean_ratio = ratio_valid.mean().item()
        else:
            clip_frac = 0.0
            mean_ratio = 1.0

    metrics = {
        "grpo_loss": loss.item(),
        "clip_fraction": clip_frac,
        "mean_ratio": mean_ratio,
        "mean_advantage": advantages.mean().item(),
    }
    return loss, metrics
