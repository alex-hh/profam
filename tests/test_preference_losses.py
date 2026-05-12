"""Unit tests for Bradley-Terry and GRPO loss math (CPU, no model)."""

import torch

from profam.losses import bradley_terry_loss, grpo_loss


def test_preference_module_imports():
    """Smoke: ensure new preference LitModules import cleanly (no model needed)."""
    from profam.models.preference import BTLitModule, GRPOLitModule  # noqa: F401
    from profam.scoring import (  # noqa: F401
        build_prompt_ids,
        cache_context,
        score_variants_differentiable,
        score_variants_no_grad,
    )


def test_build_prompt_ids_respects_max_context_tokens():
    from profam.scoring import build_prompt_ids

    msa = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    sep = 99
    prompt = build_prompt_ids(
        msa, sep_token_id=sep, start_tokens=[10], max_context_tokens=6
    )
    assert prompt[0] == 10
    assert sep not in prompt[1:] or len(prompt) <= 6
    assert len(prompt) <= 6


def test_build_prompt_ids_empty_input():
    from profam.scoring import build_prompt_ids

    assert build_prompt_ids([], sep_token_id=99, start_tokens=[10, 20]) == [10, 20]


def test_bradley_terry_loss_zero_when_scores_match_fitness_order():
    # Perfectly correlated scores and fitness → all pairwise BCE logits are
    # large positive when target is 1 and large negative when target is 0,
    # so loss is near zero.
    scores = torch.tensor([0.0, 1.0, 2.0, 3.0]) * 10.0
    fitness = torch.tensor([0.0, 1.0, 2.0, 3.0])
    loss = bradley_terry_loss(scores, fitness)
    assert loss.item() < 1e-3


def test_bradley_terry_loss_high_when_scores_invert_fitness_order():
    scores = torch.tensor([3.0, 2.0, 1.0, 0.0]) * 10.0
    fitness = torch.tensor([0.0, 1.0, 2.0, 3.0])
    loss = bradley_terry_loss(scores, fitness)
    assert loss.item() > 5.0


def test_bradley_terry_loss_log2_for_ties():
    # All zero scores, all unique fitness → each off-diagonal pair contributes
    # -log(sigmoid(0)) = log(2). Loss is 0.5 * (B*(B-1)*log(2)) / (B*(B-1))
    # = 0.5 * log(2).
    scores = torch.zeros(5)
    fitness = torch.arange(5, dtype=torch.float)
    loss = bradley_terry_loss(scores, fitness)
    assert abs(loss.item() - 0.5 * torch.log(torch.tensor(2.0)).item()) < 1e-6


def test_bradley_terry_loss_backprops():
    scores = torch.zeros(4, requires_grad=True)
    fitness = torch.tensor([0.0, 1.0, 2.0, 3.0])
    loss = bradley_terry_loss(scores, fitness)
    loss.backward()
    assert scores.grad is not None
    assert torch.isfinite(scores.grad).all()


def test_grpo_loss_zero_when_ratios_are_one_and_advantages_zero():
    # new_lps == old_lps → ratio = 1, advantage = 0 → loss = 0
    new = torch.zeros(3, 5, requires_grad=True)
    old = torch.zeros(3, 5)
    mask = torch.ones(3, 5, dtype=torch.bool)
    adv = torch.zeros(3)
    loss, metrics = grpo_loss(new, old, mask, adv)
    assert loss.item() == 0.0
    assert metrics["clip_fraction"] == 0.0
    assert abs(metrics["mean_ratio"] - 1.0) < 1e-6


def test_grpo_loss_negative_objective_when_advantage_positive_and_ratio_above_one():
    # ratio > 1 with positive advantage → surrogate objective is positive →
    # loss (negative of objective) is negative.
    new = torch.full((2, 4), 0.1, requires_grad=True)
    old = torch.zeros(2, 4)
    mask = torch.ones(2, 4, dtype=torch.bool)
    adv = torch.tensor([1.0, 1.0])
    loss, metrics = grpo_loss(new, old, mask, adv, clip_ratio=0.5)
    assert loss.item() < 0.0
    assert metrics["mean_ratio"] > 1.0


def test_grpo_loss_clip_fraction_with_extreme_ratio():
    new = torch.full((1, 3), 2.0)
    old = torch.zeros(1, 3)
    mask = torch.ones(1, 3, dtype=torch.bool)
    adv = torch.tensor([1.0])
    _, metrics = grpo_loss(new, old, mask, adv, clip_ratio=0.2)
    # ratio = exp(2) ≈ 7.39, way outside (0.8, 1.2)
    assert metrics["clip_fraction"] == 1.0


def test_grpo_loss_backprops_through_new_lps_only():
    new = torch.zeros(2, 3, requires_grad=True)
    old = torch.zeros(2, 3)
    mask = torch.ones(2, 3, dtype=torch.bool)
    adv = torch.tensor([1.0, -1.0])
    loss, _ = grpo_loss(new, old, mask, adv)
    loss.backward()
    assert new.grad is not None
    assert torch.isfinite(new.grad).all()
