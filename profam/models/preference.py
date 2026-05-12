"""Preference-optimisation Lightning modules for ProFam.

Two training modes:

- :class:`BTLitModule` — Bradley-Terry pairwise ranking loss on a batch of
  variants annotated with scalar fitness values.
- :class:`GRPOLitModule` — PPO-clipped policy-gradient loss using
  group-normalised fitness scores as rewards.

Both fine-tune the *decoder* against a frozen MSA-conditioned KV cache,
following Hawkins-Hooker et al. (2025) and the source implementation in
``pref_opt_profam``: a deep-copied snapshot of the model encodes the prompt
once, the cache is detached, and only the trainable model parameters
receive gradients during the variant forward pass.

Training-batch contract::

    batch = {
        "completion_ids": (1, B, L)  tokenised candidate sequences,
        "fitness":        (B,)       scalar fitness values,
        # GRPO only:
        "old_lps":        (B, L-1)   per-token log-probs under the reference policy,
        "old_masks":      (B, L-1)   bool mask for valid (non-padding) tokens,
    }

The frozen MSA KV cache is built lazily via :meth:`setup_frozen_context` —
typically called once after loading conditioning data, before
``trainer.fit``.
"""

import copy
from typing import List, Optional

import torch

from profam.losses.bradley_terry import bradley_terry_loss
from profam.losses.grpo import compute_per_token_log_probs, grpo_loss
from profam.models.llama import LlamaLitModule
from profam.scoring import (
    build_prompt_ids,
    cache_context,
    score_variants_differentiable,
)


class _FrozenContextMixin:
    """Shared frozen-context plumbing for BT and GRPO modules.

    Subclasses must inherit alongside :class:`LlamaLitModule` (or a compatible
    base exposing ``.model`` and ``.tokenizer``).
    """

    def setup_frozen_context(
        self,
        tokenized_conditioning_sequences: List[List[int]],
        start_tokens: Optional[List[int]] = None,
        max_context_tokens: int = 8000,
        weights=None,
        seed: int = 42,
    ) -> None:
        """Build the frozen MSA prompt and its detached KV cache.

        Uses a deep-copied snapshot of ``self`` to compute the cache so the
        live model can be fine-tuned without contaminating the frozen context
        (the cache is then detached and the copy released).
        """
        import numpy as np

        rng = np.random.default_rng(seed)
        prompt_ids = build_prompt_ids(
            tokenized_conditioning_sequences,
            sep_token_id=self.tokenizer.sep_token_id,
            start_tokens=start_tokens,
            max_context_tokens=max_context_tokens,
            weights=weights,
            rng=rng,
        )

        device = next(self.parameters()).device
        frozen_model = copy.deepcopy(self)
        frozen_model.eval()
        for p in frozen_model.parameters():
            p.requires_grad = False

        past_kv = cache_context(frozen_model, prompt_ids, device)
        detached_kv = tuple(
            tuple(t.detach().clone() for t in layer_kv) for layer_kv in past_kv
        )

        del frozen_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self._frozen_kv = detached_kv
        self._frozen_prompt_ids = prompt_ids

    @property
    def frozen_kv(self):
        if getattr(self, "_frozen_kv", None) is None:
            raise RuntimeError(
                "Frozen context not initialised — call setup_frozen_context(...) "
                "before training."
            )
        return self._frozen_kv


class BTLitModule(_FrozenContextMixin, LlamaLitModule):
    """ProFam fine-tuning with the Bradley-Terry pairwise ranking loss."""

    def __init__(self, *args, sub_batch_size: int = 4, **kwargs):
        super().__init__(*args, **kwargs)
        self.sub_batch_size = sub_batch_size
        self._frozen_kv = None
        self._frozen_prompt_ids = None

    def training_step(self, batch, batch_idx):
        completion_ids = batch["completion_ids"]
        fitness = batch["fitness"]

        scores = score_variants_differentiable(
            self,
            self.frozen_kv,
            completion_ids,
            sub_batch_size=self.sub_batch_size,
        )
        loss = bradley_terry_loss(scores, fitness)

        self.log("train/bt_loss", loss.item(), on_step=True, prog_bar=True)
        return loss


class GRPOLitModule(_FrozenContextMixin, LlamaLitModule):
    """ProFam fine-tuning with PPO-clipped GRPO on fitness-derived rewards."""

    def __init__(
        self,
        *args,
        sub_batch_size: int = 4,
        clip_ratio: float = 0.2,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.sub_batch_size = sub_batch_size
        self.clip_ratio = clip_ratio
        self._frozen_kv = None
        self._frozen_prompt_ids = None

    def precompute_reference_log_probs(
        self,
        completion_ids: torch.Tensor,
    ):
        """Compute per-token log-probs under the current (reference) policy.

        Call once before fine-tuning to snapshot the initial policy; the
        returned tensors feed ``batch["old_lps"]`` / ``batch["old_masks"]``.
        """
        self.eval()
        with torch.no_grad():
            old_lps, old_masks = compute_per_token_log_probs(
                self,
                self.frozen_kv,
                completion_ids,
                sub_batch_size=self.sub_batch_size,
            )
        return old_lps.detach(), old_masks.detach()

    def training_step(self, batch, batch_idx):
        completion_ids = batch["completion_ids"]
        fitness = batch["fitness"]
        old_lps = batch["old_lps"]
        old_masks = batch["old_masks"]

        # Group-relative advantages
        advantages = (fitness - fitness.mean()) / (fitness.std() + 1e-8)

        new_lps, new_masks = compute_per_token_log_probs(
            self,
            self.frozen_kv,
            completion_ids,
            sub_batch_size=self.sub_batch_size,
        )
        valid = old_masks & new_masks

        loss, metrics = grpo_loss(
            new_lps, old_lps, valid, advantages, clip_ratio=self.clip_ratio
        )

        for k, v in metrics.items():
            self.log(f"train/{k}", v, on_step=True, prog_bar=(k == "grpo_loss"))
        return loss
