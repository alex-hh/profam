from typing import Optional

import torch
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast
from transformers.cache_utils import Cache, StaticCache
from transformers.modeling_attn_mask_utils import AttentionMaskConverter

from src.models.base import BaseFamilyLitModule, BaseSingleSequenceLitModule
from src.models.wrapper import WrappedHFModelWithPositionEmbeddingsMixin


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


def _prepare_4d_causal_attention_mask_with_cache_position(
    attention_mask: torch.Tensor,
    sequence_length: int,
    target_length: int,
    dtype: torch.dtype,
    device: torch.device,
    min_dtype: float,
    cache_position: torch.Tensor,
    batch_size: int,
):
    """
    TODO: if we want to integrate with hf proper, it would make more sense for attention mask to always be
    non-inverted.

    We assume that the attention mask is one of the following:
        - a 2D binary mask, with 1s indicating keys that can be attended to in the full sequence
        - a 4D binary mask, with 1s indicating permitted attention.
            shape should be [broadcastable to?] (batch_size, head_dim, query_length, key_value_length)
            query_length when using cache is equal to number of uncached tokens
        - a 4D bias mask, with -inf indicating disallowed attention.
            shape should be [broadcastable to?] (batch_size, head_dim, query_length, key_value_length)

    Creates a causal 4D mask of shape `(batch_size, 1, query_length, key_value_length)` from a 2D mask of shape
    `(batch_size, key_value_length)`, or if the input `attention_mask` is already 4D, do nothing.

    Args:
        attention_mask (`torch.Tensor`):
            A 2D attention mask of shape `(batch_size, key_value_length)` or a 4D attention mask of shape `(batch_size, 1, query_length, key_value_length)`.
        sequence_length (`int`):
            The sequence length being processed.
        target_length (`int`):
            The target length: when generating with static cache, the mask should be as long as the static cache, to account for the 0 padding, the part of the cache that is not filled yet.
        dtype (`torch.dtype`):
            The dtype to use for the 4D attention mask.
        device (`torch.device`):
            The device to plcae the 4D attention mask on.
        min_dtype (`float`):
            The minimum value representable with the dtype `dtype`.
        cache_position (`torch.Tensor`):
            Indices depicting the position of the input sequence tokens in the sequence.
        batch_size (`torch.Tensor`):
            Batch size.
    """
    # original code was optimised for memory - make sure this is too.
    # for example - masked fill might be better but requires inverted mask
    if attention_mask is not None:
        assert torch.is_floating_point(attention_mask) or torch.is_integral(
            attention_mask
        ), "Attention mask must be numeric"
    if attention_mask is None or attention_mask.ndim == 2:
        # N.B. the combination of binary and non-binary masks in the original code here is pretty confusing.
        # we try to first build required binary mask, then convert to a bias mask.
        causal_mask = (
            torch.arange(target_length, device=device) <= cache_position.reshape(-1, 1)
        )[None].expand(batch_size, -1, -1)
        if attention_mask is not None:
            causal_mask[:, :, : attention_mask.shape[-1]] &= attention_mask[
                :, None, :
            ].bool()
        causal_mask = causal_mask[:, None]  # add head dim
    elif attention_mask.isin([0, 1]).all() and not (attention_mask == 0).all():
        # if we pass all 0s there is ambiguity, but we assume it means a bias mask, since it would prevent any attention.
        causal_mask = attention_mask.bool()
    else:
        causal_mask = attention_mask

    assert causal_mask.ndim() == 4
    # TODO: check if attention mask is binary at this point.
    # In this case we assume that the mask comes already in inverted form and requires no inversion or slicing.
    if causal_mask.dtype == torch.bool:
        # causal mask is binary mask with 1s where attention is allowed
        # invert and use -inf to mask out disallowed attentions
        causal_mask = (~causal_mask).masked_fill(min_dtype).to(dtype)

    # otherwise bias mask already: pass on through
    return causal_mask


class WrappedLlamaForCausalLM(
    WrappedHFModelWithPositionEmbeddingsMixin, LlamaForCausalLM
):

    # todo: modify update_causal_mask to accept bias or binary mask
    # bias directly specifies attention
    # binary mask gets combined with ar mask.

    def prepare_binary_attention_mask(
        self,
        sequence_length: int,
        target_length: int,
        device: torch.device,
        cache_position: torch.Tensor,
    ):
        causal_mask = torch.zeros(
            (sequence_length, target_length), torch.int16, device=device
        )
        if sequence_length != 1:
            causal_mask = torch.triu(causal_mask, diagonal=1)
        causal_mask *= torch.arange(
            target_length, device=device
        ) > cache_position.reshape(-1, 1)
        return causal_mask[None, None]

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

        # In case the provided `attention` mask is 2D, we generate a causal mask here (4D).
        causal_mask = _prepare_4d_causal_attention_mask_with_cache_position(
            attention_mask,
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
            # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
            # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
            # Details: https://github.com/pytorch/pytorch/issues/110213
            causal_mask = AttentionMaskConverter._unmask_unattended(
                causal_mask, min_dtype
            )

        return causal_mask

    # n.b. its still fine to use rope - since its relative
    def compute_attention_mask(self, input_ids, attention_mask):
        # this needs to be combined with the causal mask
        if self.mask_between_document_attention:
            assert attention_mask is None
            assert not self.mask_between_sequence_attention
            raise NotImplementedError()
            # document_ids = torch.cumsum(
            #     (input_ids == self.bos_token_id).float(), dim=-1
            # )
        elif self.mask_between_sequence_attention:
            assert attention_mask is None
            sequence_ids = self.compute_sequence_index(input_ids)
            attention_mask = sequence_ids[:, None, :] == sequence_ids[:, :, None]
            attention_mask = torch.where(
                attention_mask, 0, -10000.0
            )  # TODO: fix dtype, device?
            return attention_mask
        else:
            return attention_mask


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
        mask_between_document_attention: bool = False,
        mask_between_sequence_attention: bool = False,
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
                use_seq_pos=tokenizer.use_seq_pos,
                max_seq_pos=tokenizer.max_seq_pos,
                embed_coords=embed_coords,
                sep_token_id=tokenizer.sep_token_id,
                embed_sequence_index=embed_sequence_index,
                max_sequence_index=max_sequence_index,
                pass_constant_position_ids_for_global_index=pass_constant_position_ids_for_global_index,
                pass_sequence_position_ids_for_global_index=pass_sequence_position_ids_for_global_index,
            )
        else:
            model = LlamaForCausalLM(config)
        self.mask_between_document_attention = mask_between_document_attention
        self.mask_between_sequence_attention = mask_between_sequence_attention
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
