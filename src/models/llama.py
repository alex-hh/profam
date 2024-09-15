from typing import Optional

import torch
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast
from transformers.cache_utils import Cache, StaticCache
from transformers.modeling_attn_mask_utils import AttentionMaskConverter

from src.models.base import BaseFamilyLitModule, BaseSingleSequenceLitModule
from src.models.wrapper import (
    WrappedHFModelWithPositionEmbeddingsMixin,
    _prepare_4d_causal_attention_mask_with_cache_position,
)


class WrappedLlamaForCausalLM(
    WrappedHFModelWithPositionEmbeddingsMixin, LlamaForCausalLM
):

    # TODO: verify that model outputs using wrapper and causal mask are same as without.

    # todo: modify update_causal_mask to accept bias or binary mask
    # bias directly specifies attention
    # binary mask gets combined with ar mask.

    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Cache,
        output_attentions: bool,
    ):
        """Just changed to use our custom prepare_4d_causal_attention_mask_with_cache_position"""
        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None:
                raise NotImplementedError()

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
            # just check if we can ignore input attention_mask
            if (
                self.attention_mask_type == "causal"
                and AttentionMaskConverter._ignore_causal_mask_sdpa(
                    attention_mask,
                    inputs_embeds=input_tensor,
                    past_key_values_length=past_seen_tokens,
                    is_training=self.training,
                )
            ):
                return None

        dtype, device = input_tensor.dtype, input_tensor.device
        min_dtype = torch.finfo(dtype).min
        sequence_length = input_tensor.shape[1]
        if using_static_cache:
            target_length = past_key_values.get_max_length()
        else:
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else past_seen_tokens + sequence_length + 1
            )

        binary_mask_4d = self.prepare_binary_attention_mask(
            attention_mask,
            sequence_length=sequence_length,
            target_length=target_length,
            cache_position=cache_position,
            batch_size=input_tensor.shape[0],
        )
        # In case the provided `attention` mask is 2D, we generate a causal mask here (4D).
        causal_mask = _prepare_4d_causal_attention_mask_with_cache_position(
            binary_mask_4d,
            sequence_length=sequence_length,
            target_length=target_length,
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
            # With torch v2.1, scaled_dot_product_attention on GPU gives nan when a sequence has all
            # large negative values (e.g torch.finfo(q.dtype).min - in order to mean no attention at
            # all places). On CPU, it won't give nan.

            # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
            # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
            # Details: https://github.com/pytorch/pytorch/issues/110213
            causal_mask = AttentionMaskConverter._unmask_unattended(
                causal_mask, min_dtype
            )

        return causal_mask


class LlamaSingleSequenceLitModule(BaseSingleSequenceLitModule):
    def __init__(
        self,
        config: LlamaConfig,
        tokenizer: PreTrainedTokenizerFast,
        lr: float = 1e-4,
        weight_decay: float = 0.1,
        scheduler_name: Optional[str] = None,
        num_warmup_steps: int = 1000,
        num_training_steps: Optional[int] = None,
        scoring_max_tokens: int = 64000,
    ) -> None:
        model = LlamaForCausalLM(config)
        super().__init__(
            model,
            tokenizer,
            lr=lr,
            weight_decay=weight_decay,
            scheduler_name=scheduler_name,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
            scoring_max_tokens=scoring_max_tokens,
        )


class LlamaLitModule(BaseFamilyLitModule):
    def __init__(
        self,
        config: LlamaConfig,
        tokenizer: PreTrainedTokenizerFast,
        lr: float = 3e-4,
        weight_decay: float = 0.1,
        scheduler_name: Optional[str] = None,
        num_warmup_steps: int = 1000,
        num_training_steps: Optional[int] = None,
        scoring_max_tokens: int = 10240,
        use_kv_cache_for_scoring: bool = True,
        embed_coords: bool = False,
        embed_sequence_index: bool = False,
        pass_constant_position_ids_for_global_index: bool = False,
        pass_sequence_position_ids_for_global_index: bool = False,
        max_sequence_index: int = 1024,
    ) -> None:
        """
        From the paper:
        We trained using the AdamW optimizer (Loshchilov and Hutter, 2017),
        with beta1=0.9,beta2=0.95,eps=10-5. We use a cosine learning rate schedule, with warmup
        of 2000 steps, and decay final learning rate down to 10% of the peak learning rate (3e-4-1.5e-4).
        We use a weight decay of 0.1 and gradient clipping of 1.0.
        """
        if (
            tokenizer.use_seq_pos or embed_coords,
        ):  # commenting out to check computation of inputs embeds is working
            model = WrappedLlamaForCausalLM(
                config,
                token_embedder="model.embed_tokens",
                embedding_dim=config.hidden_size,
                embed_coords=embed_coords,
                tokenizer=tokenizer,
                embed_sequence_index=embed_sequence_index,
                max_sequence_index=max_sequence_index,
                pass_constant_position_ids_for_global_index=pass_constant_position_ids_for_global_index,
                pass_sequence_position_ids_for_global_index=pass_sequence_position_ids_for_global_index,
            )
        else:
            model = LlamaForCausalLM(config)
        super().__init__(
            model,
            tokenizer,
            lr=lr,
            weight_decay=weight_decay,
            scheduler_name=scheduler_name,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
            scoring_max_tokens=scoring_max_tokens,
            use_kv_cache_for_scoring=use_kv_cache_for_scoring,
            embed_coords=embed_coords,
        )
