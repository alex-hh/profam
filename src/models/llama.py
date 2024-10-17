from typing import Optional

import torch
from torch import nn
from transformers import (
    LlamaConfig,
    LlamaForCausalLM,
    LlamaModel,
    LlamaPreTrainedModel,
    PreTrainedTokenizerFast,
)
from transformers.cache_utils import Cache, StaticCache
from transformers.modeling_attn_mask_utils import AttentionMaskConverter

from src.models import attention_masking
from src.models.base import BaseFamilyLitModule, BaseSingleSequenceLitModule
from src.models.wrapper import WrappedHFModelWithPositionEmbeddingsMixin


class WrappedLlamaModel(LlamaModel):
    """LlamaForCausalLM doesn't have _update_causal_mask, so we also need to wrap LlamaModel."""

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
        Going for 1 as simplest and fairly reasonable anyway.
        """
        attention_mask_type = self.config.attention_mask_type or "causal"
        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask_type != "causal":
                raise ValueError("Flash attention doesn't support custom masks")
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
            and attention_mask_type == "causal"
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
            attention_mask_type,
            attention_mask_2d=attention_mask,
            new_sequence_length=sequence_length,
            device=device,
            cache_position=cache_position,
            batch_size=input_tensor.shape[0],
            past_key_values=past_key_values,
            seq_struct_sep_token_id=self.config.seq_struct_sep_token_id,
            sep_token_id=self.config.sep_token_id,
        ).int()

        print(f"{attention_mask_type} attention_mask", attention_mask)

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


class WrappedLlamaForCausalLM(LlamaForCausalLM):
    def __init__(self, config, *args, **kwargs):
        LlamaPreTrainedModel.__init__(self, config)
        # todo: maybe just wrap llamamodel (with position embedding wrapper also)
        self.model = WrappedLlamaModel(config, *args, **kwargs)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()


class DoubleWrappedLlama(
    WrappedHFModelWithPositionEmbeddingsMixin, WrappedLlamaForCausalLM
):
    pass


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
        max_seq_pos_in_doc: int = 1024,
        embed_residue_index: bool = True,
        max_res_pos_in_seq: int = 4096,
    ) -> None:
        """
        From the paper:
        We trained using the AdamW optimizer (Loshchilov and Hutter, 2017),
        with beta1=0.9,beta2=0.95,eps=10-5. We use a cosine learning rate schedule, with warmup
        of 2000 steps, and decay final learning rate down to 10% of the peak learning rate (3e-4-1.5e-4).
        We use a weight decay of 0.1 and gradient clipping of 1.0.
        """
        should_wrap = (
            tokenizer.embed_residue_index or embed_coords or embed_sequence_index
        )
        should_wrap = should_wrap or config.attention_mask_type is not None
        if should_wrap:
            # TODO: move to yaml file...
            config.seq_struct_sep_token_id = tokenizer.seq_struct_sep_token_id
            config.sep_token_id = tokenizer.sep_token_id
            model = DoubleWrappedLlama(
                config,
                token_embedder="model.embed_tokens",
                tokenizer=tokenizer,
                embedding_dim=config.hidden_size,
                embed_coords=embed_coords,
                embed_sequence_index=embed_sequence_index,
                embed_residue_index=embed_residue_index,
                max_seq_pos_in_doc=max_seq_pos_in_doc,
                pass_constant_position_ids_for_global_index=pass_constant_position_ids_for_global_index,
                pass_sequence_position_ids_for_global_index=pass_sequence_position_ids_for_global_index,
                max_res_pos_in_seq=max_res_pos_in_seq,
            )
        else:
            model = LlamaForCausalLM(config)
        # n.b. attention implementation gets set here (in from_pretrained, _from_config, __init__):
        # https://github.com/huggingface/transformers/blob/1dba608df93ffb10a9c268ef35191adf2424c5ca/src/transformers/modeling_utils.py#L1542
        # c.f. https://huggingface.co/docs/transformers/perf_infer_gpu_one#flashattention-2
        print(
            "Initialised Llama model, attention implementation: ",
            model.config._attn_implementation,
        )
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
            embed_residue_index=embed_residue_index,
        )
