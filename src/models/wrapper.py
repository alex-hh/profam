"""We don't separate position indexing and attention stuff because both benefit from input ids Cache.

TODO: I need to know - and be sure this handles - all the places where Cache gets created
and updated.

Created:
    - in model forward with use_cache True
    - in generate _prepare_cache_for_generation
"""

from typing import Any, Dict, List, Optional, Union

import torch
from torch import nn
from transformers import GenerationConfig, PreTrainedModel
from transformers.cache_utils import Cache
from transformers.utils import ModelOutput

from src.data.tokenizers import ProFamTokenizer
from src.models.utils import InputAwareDynamicCache
from src.utils.utils import nested_getattr


def assert_only_padding_after_eos(input_ids, eos_token_id, padding_token_id):
    sep_counts = (input_ids == eos_token_id).cumsum(dim=-1)
    assert sep_counts.max() <= 1
    should_pad = sep_counts.cumsum(-1) > 1
    assert (input_ids[should_pad] == padding_token_id).all()


# Question: how do we configure the cache type used by a model? can we?

# Ideally we just want to write a modified cache class that also holds input ids
# and probably also attention mask

# llama model forward. n.b. mistral model forward has same even though it could use sliding window cache.
# gpt doesn't use dynamic cache.


# TODO: try to modularise...
class WrappedHFModelWithPositionEmbeddingsMixin:
    """Wrap a pre-trained model to add sequence-relative position embeddings.

    This wrapper is PARTIALLY agnostic to the underlying model. However the underlying
    model must be able to handle a DynamicCache object as past_key_values, and must
    accept a 4d attention (bias) mask.

    have position_ids argument in .forward() method
    use modeling_attn_mask_utils.py::_prepare_4d_attention_mask() function for 4d mask generation

    (Optionally other embeddings, e.g. structure embeddings, could be added in similar way.)

    args:
        pass_sequence_position_ids_for_global_index: if True, pass sequence position ids to model
    """

    # This is a mixin for models that require seq pos input during generation
    # using the mixin allows the use of standard generation code
    def __init__(
        self,
        config,
        token_embedder: str,
        embedding_dim: int,
        tokenizer: ProFamTokenizer,
        pass_res_pos_in_doc_as_position_ids: bool = False,
    ):
        super().__init__(config)
        self.tokenizer = tokenizer
        # TODO: avoid re-tracking - does this happen automatically?
        self.token_embedder = nested_getattr(
            self, token_embedder
        )  # TODO: use self.embed_tokens or sthg
        self.tokenizer = tokenizer
        self.num_atoms = 4
        self.pass_res_pos_in_doc_as_position_ids = pass_res_pos_in_doc_as_position_ids

    def _prepare_cache_for_generation(
        self,
        generation_config: GenerationConfig,
        model_kwargs: Dict,
        assistant_model: "PreTrainedModel",
        batch_size: int,
        max_cache_length: int,
        device: torch.device,
    ) -> bool:
        cache_name = (
            "past_key_values"
            if "mamba" not in self.__class__.__name__.lower()
            else "cache_params"
        )
        assert cache_name == "past_key_values"
        assert generation_config.use_cache

        model_kwargs[cache_name] = InputAwareDynamicCache()

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        cache_position=None,
        use_cache=True,
        **kwargs,
    ):
        """Build inputs dictionary for next step in generation, given full input_ids (
        prompt + generated tokens), and other model kwargs (also full length).

        Main function is to use cache_position to slice out the unprocessed tokens,
        and corresponding inputs.

        This is a model-specific method in HF.
        Inputs are then incremented in _update_model_kwargs_for_generation.
        For some inputs, we currently have to update here (sequence_index) -
        https://github.com/huggingface/transformers/issues/33548

        n.b. we need to be aware of main steps of generation pipeline (self.generate)
        1. null cache gets created (setting past_key_values in model_kwargs) - unless creation is required from start
        https://github.com/huggingface/transformers/blob/174890280b340b89c5bfa092f6b4fb0e2dc2d7fc/src/transformers/generation/utils.py#L1854
        2. get_initial_cache_position:
        https://github.com/huggingface/transformers/blob/174890280b340b89c5bfa092f6b4fb0e2dc2d7fc/src/transformers/generation/utils.py#L2969
        n.b. cache_position is a very misleading name for cache-aware position index
        initially, cache position is just arange(len(input_ids))
        Then loop:
        3. prepare_inputs_for_generation: slice out input ids if using cache.
            in first iteration this does nothing, since cache_position shape == len(input_ids)
            in subsequent iterations, cache_position is index of newly generated token(s)
            relative to updated input_ids (i.e. prompt + generated tokens). so input_ids[cache_position]
            selects just the newly generated token(s) from the last sampling iteration to feed to the model
        4. compute logits for next position in sequence, predict a new token and update input ids
        5. update_model_kwargs_for_generation: update cache, attention_mask, cache_position.
        """
        # main place this gets called is in sample loop:
        # https://github.com/huggingface/transformers/blob/e7f4ace0929600606424efd4cd91947bd567d323/src/transformers/generation/utils.py#L2413
        # in sample loop 'input_ids' gets incremented with generated tokens

        assert input_ids.ndim == 2
        if past_key_values is not None:
            cache = InputAwareDynamicCache.from_legacy_cache(past_key_values)
        else:
            cache = None
        inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=cache,
            cache_position=cache_position,
            use_cache=use_cache,
            **kwargs,
        )  # slices out prompt and uses cache typically.

        # after first forward pass,inputs["input_ids"]
        # is last generated token - so far not passed through model:
        # this is sliced from input_ids and added to inputs dict in base class prepare_inputs_for_generation

        if inputs["input_ids"].shape[-1] == 1:
            # on first forward pass inputs["input_ids"] is the full prompt
            # after first forward pass, inputs["input_ids"] is the last generated token
            # only run this check after 1st forward pass is done and we are generating
            generated_token = input_ids[:, -1]
            if generated_token == self.tokenizer.sep_token_id or (
                generated_token == self.tokenizer.seq_struct_sep_token_id
            ):
                # sep would break incrementaion of seq pos and sequence index
                raise NotImplementedError(
                    "This code does not handle generation of sequences with separators."
                )

        # input_ids is prompt + generated tokens

        return inputs

    def _update_model_kwargs_for_generation(
        self,
        outputs: ModelOutput,
        model_kwargs: Dict[str, Any],
        is_encoder_decoder: bool = False,
        num_new_tokens: int = 1,
    ):
        """Update model kwargs for next step in generation, given model outputs and current model kwargs.

        model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
        next_tokens = ....
        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
        model_kwargs = self._update_model_kwargs_for_generation(
            outputs,
            model_kwargs,
            is_encoder_decoder=self.config.is_encoder_decoder,
        )

        if new token is sep, then new seq pos should be incremented.
        if prev token is sep, then new seq pos should be 0 and new sequence index should be incremented.
        """
        # update past_key_values using model output, and also update attention_mask, cache_position
        # TODO: handle attention mask update - maybe pop from model_kwargs and update here instead
        super()._update_model_kwargs_for_generation(
            outputs,
            model_kwargs,
            is_encoder_decoder=is_encoder_decoder,
            num_new_tokens=num_new_tokens,
        )

        assert model_kwargs["use_cache"]
        assert (
            "past_key_values" in outputs
        ), "We assume we're using cache and past_key_values is cache_name, otherwise sequence index is hard to compute"
        past_key_values = (
            outputs.past_key_values
        )  # TODO: check this is right way to access - not exactly how generate does it in super


        return model_kwargs

    def compute_sequence_index(self, input_ids, start_sequence_index=0):
        # TODO: test - if input_ids is just sep token we return start_sequence_index
        # cat means sep token gets index of PREVIOUS sequence
        sequence_indices = start_sequence_index + torch.cat(
            (
                torch.full_like(input_ids[..., :1], 0),
                torch.cumsum(
                    (input_ids == self.tokenizer.sep_token_id).float(), dim=-1
                ).long()[..., :-1],
            ),
            dim=-1,
        )
        document_start_mask = input_ids == self.tokenizer.bos_token_id
        if (document_start_mask.sum(-1) > 1).any():
            assert (
                input_ids.shape[0] == 1
            ), "Batch size must be 1 if there are multiple packed documents"
            document_starts = torch.argwhere(
                input_ids[0] == self.tokenizer.bos_token_id
            ).flatten()
            if document_starts.shape[0] > 0:
                document_start_mask = torch.zeros_like(input_ids[0], dtype=torch.bool)
                document_start_mask[document_starts] = True
                document_indices = torch.cumsum(document_start_mask, dim=-1) - 1
                if start_sequence_index != 0 and document_start_mask.sum() > 1:
                    raise NotImplementedError()  # needs to be batched
                offsets = sequence_indices[0, document_starts[document_indices]]
                return sequence_indices - offsets
        return sequence_indices

    def compute_res_pos_in_doc(self, input_ids):
        """Needs to start at 0 for compatibility with sequence packing:
        https://github.com/huggingface/transformers/blob/70b07d97cf2c5f61fff55700b65528a1b6845cd2/src/transformers/modeling_flash_attention_utils.py#L133
        """
        # FIXME: UNCOMMENT?
        # assert (
        #     input_ids.shape[0] == 1
        # ), "Since we are typically packing sequences, we assume batch size is 1"
        counter = torch.arange(input_ids.shape[1], device=input_ids.device)
        document_indices = (
            torch.cumsum(input_ids[0] == self.tokenizer.bos_token_id, 0) - 1
        )
        assert (
            document_indices >= 0
        ).all(), "Negative document indices encountered: check that bos token is first token in each document"
        doc_starts = (
            torch.argwhere(input_ids[0] == self.tokenizer.bos_token_id)
        ).flatten()
        offsets = counter[doc_starts][document_indices]
        position_ids = (counter - offsets).unsqueeze(0)
        return position_ids

    def compute_start_sequence_index(self, past_key_values):
        if past_key_values is None or past_key_values.input_ids_cache is None:
            return torch.tensor([0]).to(self.device)
        # model will automatically do compute_sequence_index on new tokens
        # so we just need to tell it the relative sequence index of the new tokens
        # suppose input_ids[:, -1] is sep token. then compute_sequence_index here will assign it
        # to previous sequence, and in forward will just pass through start_sequence_index
        full_sequence_index = self.compute_sequence_index(
            past_key_values.input_ids_cache
        )
        return torch.where(
            past_key_values.input_ids_cache[:, -1] == self.tokenizer.sep_token_id,
            full_sequence_index[:, -1] + 1,
            full_sequence_index[:, -1],
        )

    def embed_inputs(
        self,
        input_ids: Optional[torch.LongTensor],
    ):
        # we assume (which is case for e.g. gpt2 and mistral)
        # that the model will itself add its own position embeddings to inputs_embeds
        assert input_ids.ndim == 2

        # in this case model's position ids will be inferred from inputs_embeds
        inputs_embeds = self.token_embedder(input_ids)

        if hasattr(self, "dtype"):
            inputs_embeds = inputs_embeds.to(self.dtype)

        return inputs_embeds

    def get_position_ids_for_model_forward(
        self, input_ids, position_ids, past_key_values
    ):
        if input_ids.shape[0] > 1:
            assert (input_ids == self.tokenizer.bos_token_id).sum(axis=1).max() <= 1, "Sequence packing not supported with batch size > 1"
            position_ids = None
        elif past_key_values is not None:
            assert (
                input_ids == self.tokenizer.bos_token_id
            ).sum() <= 1, "Sequence packing not supported with past_key_values"
            position_ids = None

        elif self.pass_res_pos_in_doc_as_position_ids:
            position_ids = self.compute_res_pos_in_doc(input_ids)
        return position_ids

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,  # e.g. labels
    ):
        assert (
            inputs_embeds is None
        ), "Do not pass pre-computed embeddings to this class"

        if use_cache and not isinstance(past_key_values, Cache) and not self.training:
            assert (
                past_key_values is None
            ), "We need to know input ids, which can't be passed in legacy format"
            past_key_values = InputAwareDynamicCache()

        start_sequence_index = None

        inputs_embeds = self.embed_inputs(
            input_ids,
            start_sequence_index=start_sequence_index[:, None]
            if start_sequence_index is not None
            else None,  # broadcast to input ids
        )

        position_ids = self.get_position_ids_for_model_forward(
            input_ids, position_ids, past_key_values
        )

        outputs = super().forward(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )

        if past_key_values is not None and use_cache:
            assert isinstance(past_key_values, InputAwareDynamicCache)
            # TODO: only if use cache?
            past_key_values.update_inputs(input_ids=input_ids)
            assert (
                outputs.past_key_values.input_ids_cache
                == past_key_values.input_ids_cache
            ).all()  # check that we are updating the past key values in outputs
        return outputs
