"""Bradley-Terry pairwise ranking loss."""

import torch
import torch.nn.functional as F


def bradley_terry_loss(
    scores: torch.Tensor, fitness_values: torch.Tensor
) -> torch.Tensor:
    """Bradley-Terry pairwise ranking loss.

    Args:
        scores: (B,) differentiable log-likelihood scores from the model.
        fitness_values: (B,) ground-truth fitness values (any monotonic scale).

    Returns:
        Scalar loss tensor with gradients.
    """
    B = scores.shape[0]
    score_diffs = scores.unsqueeze(1) - scores.unsqueeze(0)
    targets = (fitness_values.unsqueeze(1) > fitness_values.unsqueeze(0)).float()
    mask = 1.0 - torch.eye(B, device=scores.device)
    loss = F.binary_cross_entropy_with_logits(score_diffs, targets, reduction="none")
    return 0.5 * (loss * mask).sum() / mask.sum()
