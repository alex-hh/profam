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

from src.models.utils import InputAwareDynamicCache
from src.utils.tokenizers import ProFamTokenizer
from src.utils.utils import nested_getattr

# TODO: really we want to write our own cache class.
# we can just overwrite update and add relevant attributes.
# we then need to make sure that we instantiate the right type of cache
# in relevant places (e.g. in the model forward with use_cache True.)


def assert_only_padding_after_eos(input_ids, eos_token_id, padding_token_id):
    # as long as we pad after sep, it doesn't matter what seq_pos is associated with sep
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

    c.f. 4d attention mask pr: https://github.com/huggingface/transformers/pull/27539#issuecomment-1864421993
    IMPORTANT: this PR makes changes that can only used by few classes of models
    requirements to use:

    (Optionally other embeddings, e.g. structure embeddings, could be added in similar way.)

    args:
        use_seq_pos: embed position of amino acid within sequence (TODO: standardise variable naming)
        embed_sequence_index: if True, embed index of sequence within sequence of sequences (TODO: rename)
        pass_constant_position_ids_for_global_index: if True, pass constant position ids to model (for e.g. inbuilt ROPE embeddings)
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
        require_seq_pos: bool = True,
        embed_coords: bool = False,
        start_seq_pos: int = 2,
        embed_sequence_index: bool = False,
        pass_constant_position_ids_for_global_index: bool = False,
        pass_sequence_position_ids_for_global_index: bool = False,
        max_sequence_index: int = 1024,
        attention_mask_type: str = "causal",
    ):
        super().__init__(config)
        self.tokenizer = tokenizer
        self.use_seq_pos = tokenizer.use_seq_pos
        self.start_seq_pos = start_seq_pos  # TODO: double-check this is consistent
        # TODO: avoid re-tracking - does this happen automatically?
        self.token_embedder = nested_getattr(
            self, token_embedder
        )  # TODO: use self.embed_tokens or sthg
        self.attention_mask_type = attention_mask_type
        self.require_seq_pos = require_seq_pos
        self.tokenizer = tokenizer
        self.embed_coords = embed_coords
        self.start_seq_pos = start_seq_pos
        self.num_atoms = 4
        self.embed_sequence_index = embed_sequence_index
        self.max_sequence_index = max_sequence_index
        self.pass_constant_position_ids_for_global_index = (
            pass_constant_position_ids_for_global_index
        )
        self.pass_sequence_position_ids_for_global_index = (
            pass_sequence_position_ids_for_global_index
        )
        if self.embed_coords:
            self.coords_embedding = nn.Linear(
                self.num_atoms * 3, embedding_dim, bias=False
            )
        if self.tokenizer.use_seq_pos:
            self.seq_pos_embedding = nn.Embedding(
                self.tokenizer.max_seq_pos, embedding_dim
            )
        if self.embed_sequence_index:
            self.sequence_index_embedding = nn.Embedding(
                self.max_sequence_index, embedding_dim
            )

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

    # This needs to be the instantiation target if using seq pos... or wrapped hf model needs to handle properly
    def prepare_inputs_for_generation(
        self,
        input_ids,
        attention_mask=None,
        past_key_values=None,
        seq_pos=None,
        cache_position=None,
        use_cache=True,
        coords=None,
        **kwargs,
    ):
        """Build inputs dictionary for next step in generation, given full input_ids (
        prompt + generated tokens), and other model kwargs (also full length).

        Main function is to use cache_position to slice out the unprocessed tokens,
        and corresponding inputs.

        This is a model-specific method in HF.

        Q. at what point do input ids get tiled out to batch size?
        Q. can we use generation_config.cache_implementation?

        n.b. we need to be aware of main steps of generation pipeline (self.generate)
        0. kwargs passed to generate get separated into generation config and model_kwargs
        generation_config, model_kwargs = self._prepare_generation_config(generation_config, **kwargs)
        1. null cache gets created (setting past_key_values in model_kwargs) - unless creation is required from start
        https://github.com/huggingface/transformers/blob/174890280b340b89c5bfa092f6b4fb0e2dc2d7fc/src/transformers/generation/utils.py#L1854
        type of cache is determined by generation_config.cache_implementation;
        defaults to DynamicCache if supported by model.

        attention mask also gets created:
        kwargs_has_attention_mask = model_kwargs.get("attention_mask", None) is not None
        requires_attention_mask = "encoder_outputs" not in model_kwargs
        if not kwargs_has_attention_mask and requires_attention_mask and accepts_attention_mask:
            model_kwargs["attention_mask"] = self._prepare_attention_mask_for_generation(
                inputs_tensor, generation_config._pad_token_tensor, generation_config._eos_token_tensor
            )
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

            passed kwargs are passed through from generate
        4. compute logits for next position in sequence, predict a new token and update input ids
        5. update_model_kwargs_for_generation: update cache, attention_mask, cache_position.
        """
        # main place this gets called is in sample loop:
        # https://github.com/huggingface/transformers/blob/e7f4ace0929600606424efd4cd91947bd567d323/src/transformers/generation/utils.py#L2413
        # in sample loop 'input_ids' gets incremented with generated tokens

        assert input_ids.ndim == 2
        if self.use_seq_pos:
            assert seq_pos is not None

        # if None is passed to forward, default will be created.
        # we shouldn't pass a 4d mask - this is handled by forward method (_update_causal_mask)
        assert attention_mask.ndim == 2

        inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            cache_position=cache_position,
            use_cache=use_cache,
            attention_mask=attention_mask,
            **kwargs,
        )  # slices out prompt and uses cache typically.

        generated_tokens = input_ids[:, seq_pos.shape[-1] :]
        if (generated_tokens == self.tokenizer.sep_token_id).any() or (
            generated_tokens == self.tokenizer.seq_struct_sep_token_id
        ).any():
            # sep would break incrementaion of seq pos and sequence index
            raise NotImplementedError(
                "This code does not handle generation of sequences with separators."
            )

        # input_ids is prompt + generated tokens
        # kwargs["seq_pos"] is prompt only
        # inputs["input_ids"] is last generated token - so far not passed through model:
        # this is sliced from input_ids and added to inputs dict in base class prepare_inputs_for_generation
        if self.use_seq_pos:
            inputs["seq_pos"] = (
                seq_pos[:, cache_position] if past_key_values is not None else seq_pos
            )

        if self.embed_coords:
            # updated in _update_model_kwargs_for_generation
            assert input_ids.shape[-1] == kwargs["coords"].shape[1]
            inputs["coords"] = (
                coords[:, cache_position] if past_key_values is not None else coords
            )

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
        # update past_key_values using model output, token_type_ids, attention_mask, cache_position
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

        if self.tokenizer.use_seq_pos:
            assert num_new_tokens == 1
            # we assume we only increment by one, which makes things easier
            assert model_kwargs["cache_position"].shape[-1] == 1
            prev_seq_pos = model_kwargs["seq_pos"][:, -1:]
            new_seq_pos = torch.where(
                torch.isin(
                    past_key_values.input_ids_cache[:, -1:],
                    torch.tensor(
                        [
                            self.tokenizer.sep_token_id,
                            self.tokenizer.seq_struct_sep_token_id,
                        ]
                    ),
                ),
                torch.full_like(prev_seq_pos, self.start_seq_pos),
                prev_seq_pos + 1,
            )
            model_kwargs["seq_pos"] = torch.cat(
                [model_kwargs["seq_pos"], new_seq_pos], dim=-1
            )

        if "coords" in model_kwargs:
            bsz, _, n_atoms, _ = model_kwargs["coords"].shape
            model_kwargs["coords"] = torch.cat(
                [
                    model_kwargs["coords"],
                    torch.zeros(bsz, num_new_tokens, n_atoms, 3).to(
                        model_kwargs["coords"]
                    ),
                ],
                dim=1,
            )

        return model_kwargs

    def compute_sequence_index(self, input_ids, start_sequence_index=0):
        # TODO: test - if input_ids is just sep token we return start_sequence_index
        # cat means sep token gets index of PREVIOUS sequence
        return start_sequence_index + torch.cat(
            (
                torch.full_like(input_ids[..., :1], 0),
                torch.cumsum(
                    (input_ids == self.tokenizer.sep_token_id).float(), dim=-1
                ).long()[..., :-1],
            ),
            dim=-1,
        )

    def compute_start_sequence_index(self, past_key_values):
        # TODO: write test for this!
        if past_key_values is None or past_key_values.input_ids_cache is None:
            return 0
        # model will automatically do compute_sequence_index on new tokens
        # so we just need to tell it the sequence index of the new tokens
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
        seq_pos: Optional[torch.LongTensor] = None,
        coords: Optional[torch.FloatTensor] = None,
        start_sequence_index: Optional[Union[int, torch.Tensor]] = None,
    ):
        # we assume (which is case for e.g. gpt2 and mistral)
        # that the model will itself add its own position embeddings to inputs_embeds
        assert input_ids.ndim == 2

        # in this case model's position ids will be inferred from inputs_embeds
        inputs_embeds = self.token_embedder(input_ids)
        if self.tokenizer.use_seq_pos:
            if self.require_seq_pos:
                assert seq_pos is not None
            if seq_pos is not None:
                pos_embeds = self.seq_pos_embedding(seq_pos)
                inputs_embeds = inputs_embeds + pos_embeds

        # TODO: might want to embed coords mask to allow for masked coords
        if self.embed_coords:
            assert coords.ndim == 4, coords.shape  # b, l, n, 3
            coords_embeds = self.coords_embedding(coords.flatten(start_dim=-2))
            inputs_embeds += coords_embeds

        if self.embed_sequence_index:
            assert start_sequence_index is not None
            sequence_index = self.compute_sequence_index(
                input_ids, start_sequence_index=start_sequence_index
            )
            inputs_embeds += self.sequence_index_embedding(sequence_index)

        return inputs_embeds

    def get_position_ids_for_model_forward(self, input_ids, seq_pos, position_ids):
        # TODO: test these; make sure they get called during generation for example.
        if self.pass_constant_position_ids_for_global_index:
            assert position_ids is None
            position_ids = torch.full_like(input_ids, 10).long()
        if self.pass_sequence_position_ids_for_global_index:
            assert position_ids is None
            assert seq_pos is not None
            position_ids = seq_pos
        return position_ids

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        seq_pos: Optional[torch.LongTensor] = None,  # added this line for PFLM
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        coords: Optional[torch.FloatTensor] = None,
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

        if self.embed_sequence_index:
            if past_key_values is not None:
                assert isinstance(past_key_values, InputAwareDynamicCache)
                print(past_key_values.input_ids_cache)
            start_sequence_index = self.compute_start_sequence_index(past_key_values)
        else:
            start_sequence_index = None

        inputs_embeds = self.embed_inputs(
            input_ids,
            seq_pos=seq_pos,
            coords=coords,
            start_sequence_index=start_sequence_index,
        )
        position_ids = self.get_position_ids_for_model_forward(
            input_ids, seq_pos, position_ids
        )
        if past_key_values is not None and use_cache:
            # n.b. it's slight hack to do this before forward, but it
            # helps input aware attention mask construction in _update_causal_mask,
            # and nothing else in forward is affected (only _update_model_kwargs_for_generation)
            # which is outside of forward.
            # cache_position is inferred based on past_key_values.get_seq_length
            assert isinstance(past_key_values, InputAwareDynamicCache)
            past_key_values.update_inputs(input_ids=input_ids)

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
            assert (
                outputs.past_key_values.input_ids_cache
                == past_key_values.input_ids_cache
            ).all()  # check that we are updating the past key values in outputs
        return outputs
