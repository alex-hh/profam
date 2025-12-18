import os
from typing import List, Optional

import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from transformers.cache_utils import DynamicCache

from src.constants import BASEDIR


def calc_grad_norm(params):
    grad_norm = torch.norm(
        torch.stack(
            [torch.norm(p.grad.detach(), 2) for p in params if p.grad is not None]
        ),
        2,
    )

    return grad_norm


class InputAwareDynamicCache(DynamicCache):
    """A DynamicCache that allows for batched key-value caching.
    Manually implements latest version of DynamicCache from transformers.
    (once this is released we can remove this class)

    If we use this in a wrapper, we can call update_inputs in forward,
    then pass the cache on to the model.

    Notes on various cache types:
    https://huggingface.co/docs/transformers/main/en/kv_cache

    Q. why doesn't HF already have an attention mask cache? surely
    2D attention mask could be inconsistent with causal mask?

    Oh - is it because attention_mask is expected to align with target length?
    Seemingly yes
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.input_ids_cache = None

    def batch_repeat_interleave(self, repeats: int):
        """Repeat the cache `repeats` times in the batch dimension. Used in contrastive search."""
        super().batch_repeat_interleave(repeats)
        if self.input_ids_cache is not None:
            self.input_ids_cache = self.input_ids_cache.repeat_interleave(
                repeats, dim=0
            )

    def update_inputs(self, input_ids):
        assert input_ids.ndim == 2
        if self.input_ids_cache is None:
            self.input_ids_cache = input_ids.clone()
        else:
            self.input_ids_cache = torch.cat([self.input_ids_cache, input_ids], dim=-1)

    @classmethod
    def from_dynamic_cache(cls, cache):
        assert not cache.key_cache
        # basically we just create a new cache - we can do this ourselves
        raise NotImplementedError()

    @classmethod
    def from_legacy_cache(cls, cache):
        new_cache = super().from_legacy_cache(cache)
        if isinstance(cache, cls):
            new_cache.input_ids_cache = cache.input_ids_cache
        return new_cache


def log_likelihood_from_outputs(model_outputs, labels, start_ix=0, flatten=False):
    """Compute the negative log likelihood of the target sequence given the model outputs.

    Args:
        model_outputs: The model outputs from the forward pass.
        input_ids: The input sequence.

    Returns:
        The negative log likelihood of the target sequence.
    """
    logits = model_outputs.logits
    # https://github.com/huggingface/transformers/blob/4a6024921fa142f28e8d0034ae28693713b3bfd0/src/transformers/models/mistral/modeling_mistral.py#L1210

    # Shift so that tokens < n predict n
    shift_logits = logits[..., start_ix:-1, :].contiguous()  # b, L, V
    shift_labels = labels[..., start_ix + 1 :].contiguous()  # b, L
    # Ensure tensors are on the same device
    shift_labels = shift_labels.to(shift_logits.device)
    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")

    if flatten:
        # Flatten the tokens
        shift_logits = shift_logits.view(-1, shift_logits.shape[-1])
        shift_labels = shift_labels.view(-1)
        log_likelihood = -loss_fct(shift_logits, shift_labels)
    else:
        log_likelihood = -loss_fct(shift_logits.permute(0, 2, 1), shift_labels)

    return log_likelihood
