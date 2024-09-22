from typing import List, Tuple, Union

import torch
from transformers import LlamaForCausalLM
from transformers.cache_utils import Cache, StaticCache
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.utils import logging

from src.models import attention_masking
from src.models.utils import InputAwareDynamicCache

logger = logging.get_logger(__name__)


class WrappedLlamaForCausalLM(
    WrappedHFModelWithPositionEmbeddingsMixin, LlamaForCausalLM
):
    attention_mask_type: str

    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Cache,
        output_attentions: bool,
    ):
        """Compute attention mask

        N.B. update_causal_mask is a misnomer on custom model,
        as various masking patterns are supported, not just causal.

        We modify the original implementation by (i) first constructing
        a binary 4d mask reflecting the attention_mask, and (ii) then
        converting this to a bias to add to the raw attention logits.

        To get around the issue of requiring input_ids here to create input-aware
        attention masks, we have two options:
            1. as a hack, pre-update the past key values with the new input ids -
            only other thing affected is _update_model_kwargs_for_generation, which
            happens outside of forward pass anyway.
            2. compute the 4d binary attention mask in the wrapped forward pass
            and pass it to model.forward

        Going for 1 as simplest and fairly reasonable anyway."""
        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and 0.0 in attention_mask:
                return attention_mask
            return None

        # For SDPA, when possible, we will rely on its `is_causal` argument instead of its `attn_mask` argument, in
        # order to dispatch on Flash Attention 2. This feature is not compatible with static cache, as SDPA will fail
        # to infer the attention mask.
        past_seen_tokens = (
            past_key_values.get_seq_length() if past_key_values is not None else 0
        )
        using_static_cache = isinstance(past_key_values, StaticCache)

        # When output attentions is True, sdpa implementation's forward method calls the eager implementation's forward
        if (
            self.config._attn_implementation == "sdpa"
            and not using_static_cache
            and not output_attentions
        ):
            if AttentionMaskConverter._ignore_causal_mask_sdpa(
                attention_mask,
                inputs_embeds=input_tensor,
                past_key_values_length=past_seen_tokens,
                is_training=self.training,
            ):
                return None

        dtype, device = input_tensor.dtype, input_tensor.device
        sequence_length = input_tensor.shape[1]

        attention_mask = attention_masking.prepare_binary_attention_mask(
            self.attention_mask_type,
            attention_mask_2d=attention_mask,
            sequence_length=sequence_length,
            device=device,
            past_key_values=past_key_values,
            batch_size=input_tensor.shape[0],
        )

        min_dtype = torch.finfo(dtype).min
        # convert the binary attentino mask to a bias to add to raw attention logits
        causal_mask = attention_masking._prepare_4d_attention_mask_with_cache_position(
            attention_mask,
            sequence_length=sequence_length,
            target_length=attention_mask.shape[-1],
            dtype=dtype,
            device=device,
            min_dtype=min_dtype,
            cache_position=cache_position,
            batch_size=input_tensor.shape[0],
        )

        if (
            self.config._attn_implementation == "sdpa"
            and attention_mask is not None
            and attention_mask.device.type == "cuda"
            and not output_attentions
        ):
            # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
            # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
            # Details: https://github.com/pytorch/pytorch/issues/110213
            causal_mask = AttentionMaskConverter._unmask_unattended(
                causal_mask, min_dtype
            )

        return causal_mask
