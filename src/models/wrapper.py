"""We don't separate position indexing and attention stuff because both benefit from input ids Cache.

TODO: I need to know - and be sure this handles - all the places where Cache gets created
and updated.

Created:
    - in model forward with use_cache True
    - in generation utils?


"""

from typing import List, Optional, Union

import torch
from torch import nn
from transformers.cache_utils import Cache

from src.models import attention_masking
from src.models.utils import UpdatedDynamicCache
from src.utils.tokenizers import ProFamTokenizer
from src.utils.utils import nested_getattr

# TODO: really we want to write our own cache class.
# we can just overwrite update and add relevant attributes.
# we then need to make sure that we instantiate the right type of cache
# in relevant places (e.g. in the model forward with use_cache True.)


# Question: how do we configure the cache type used by a model? can we?

# Ideally we just want to write a modified cache class that also holds input ids
# and probably also attention mask

# llama attention forward:
# sin and cos are specific to RoPE models; cache_position needed for the static cache
# cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
# key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

# llama model forward
# return_legacy_cache = False
# this instantiates an empty cache container if the passed cache is None
# if (
#     use_cache and not isinstance(past_key_values, Cache) and not self.training
# ):  # kept for BC (non `Cache` `past_key_values` inputs)
#     return_legacy_cache = True
#     past_key_values = DynamicCache.from_legacy_cache(past_key_values)
#     logger.warning_once(
#         "We detected that you are passing `past_key_values` as a tuple and this is deprecated and will be removed in v4.43. "
#         "Please use an appropriate `Cache` class (https://huggingface.co/docs/transformers/v4.41.3/en/internal/generation_utils#transformers.Cache)"
#     )


# TODO: try to modularise...
class WrappedHFModelWithPositionEmbeddingsMixin:
    """Wrap a pre-trained model to add sequence-relative position embeddings.

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

    def prepare_binary_attention_mask(
        self,
        attention_mask_2d: Optional[torch.Tensor],
        sequence_length: int,
        target_length: int,
        device: torch.device,
        cache_position: torch.Tensor,
        batch_size: int,
        input_ids: Optional[torch.LongTensor] = None,
    ):
        if self.attention_mask_type == "causal":
            return attention_masking._prepare_causal_4d_binary_mask(
                attention_mask_2d,
                sequence_length,
                target_length,
                device,
                cache_position,
                batch_size,
            )
        elif self.attention_mask_type == "bidirectional":
            return attention_masking._prepare_bidirectional_4d_binary_mask(
                attention_mask_2d,
                sequence_length,
                target_length,
                device,
                cache_position,
                batch_size,
            )
        elif self.attention_mask_type == "sequence":
            assert input_ids is not None
            return attention_masking._prepare_intra_separator_4d_binary_mask(
                input_ids,
                attention_mask_2d,
                sequence_length,
                target_length,
                device,
                cache_position,
                batch_size,
                self.tokenizer.sep_token_id,
            )
        elif self.attention_mask_type == "document":
            raise NotImplementedError()  # would require document concatenation to be implemented
            # return _prepare_intra_separator_4d_binary_mask(
            #     attention_mask_2d,
            #     sequence_length,
            #     target_length,
            #     device,
            #     cache_position,
            #     batch_size,
            #     self.tokenizer.bos_token_id,
            # )
        elif self.attention_mask_type == "prefix-lm":
            # need a prefix indicator
            return attention_masking._prepare_prefix_lm_4d_binary_mask(
                attention_mask_2d,
                sequence_length,
                target_length,
                device,
                cache_position,
                batch_size,
                self.tokenizer.seq_struct_sep_token_id,
                self.tokenizer.sep_token_id,
            )
        else:
            raise ValueError(
                "Unsupported attention mask type", self.attention_mask_type
            )

    # This needs to be the instantiation target if using seq pos... or wrapped hf model needs to handle properly
    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        """n.b. we need to be aware of main steps of generation pipeline (self.generate)
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
        # TODO: consider putting this in update_model_kwargs_for_generation

        # main place this gets called is in sample loop:
        # https://github.com/huggingface/transformers/blob/e7f4ace0929600606424efd4cd91947bd567d323/src/transformers/generation/utils.py#L2413
        # in sample loop 'input_ids' gets incremented with generated tokens
        # # update generated ids, model inputs, and length for next step
        # input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
        # we're going to assume that the prompt ends with a separator token
        assert input_ids.ndim == 2
        assert "seq_pos" in kwargs
        # AH - to test this we first need to actually compute the cache. It's a complex test!
        inputs = super().prepare_inputs_for_generation(
            input_ids, **kwargs
        )  # slices out prompt and uses cache typically.

        # generated_tokens = input_ids[:, kwargs["seq_pos"].shape[-1] :]
        # if (generated_tokens == self.tokenizer.sep_token_id).any() or (
        #     generated_tokens == self.tokenizer.seq_struct_sep_token_id
        # ).any():
        #     # sep would break incrementaion of seq pos
        #     raise NotImplementedError(
        #         "This code does not handle generation of sequences with separators."
        #     )

        # TODO: handle sep token generation
        if input_ids.shape[-1] != kwargs["seq_pos"].shape[-1]:
            # we have incremented input ids but not seq pos
            increment = input_ids.shape[-1] - kwargs["seq_pos"].shape[-1]
            assert inputs["input_ids"].shape[-1] == 1, inputs[
                "input_ids"
            ].shape  # we have sliced out the last token and are using cache - does this need to be true?
            # https://github.com/huggingface/transformers/blob/cf32ee1753c9747b877113a309c2aa989f6d006c/src/transformers/models/llama/modeling_llama.py#L1236
            # just automatically increment the seq pos: this corresponds to never generating insertions in case of msas.
            # we need to slice out all previously considered sequence positions - at what point does this occur for input ids?
            # we need to be very careful about caching - inputs embeds is hopefully not being passed to the outer model? but if it is we
            # need to handle that. and we also need to ensure that slicing of seq pos is consistent with slicing of input ids.
            # frustratingly slicing of input ids is done on the super prepare inputs for generation
            input_final_seq_pos = kwargs["seq_pos"][:, -1:]
            input_length = kwargs["seq_pos"].shape[-1]
            if (input_final_seq_pos[:, -1] == 0).any():  # handles sep cases
                assert input_ids[0, input_length - 1].item() in [
                    self.tokenizer.sep_token_id,
                    self.tokenizer.seq_struct_sep_token_id,
                ], f"{input_ids[0, input_length-1]} {increment}"
                assert (input_final_seq_pos[:, -1] == 0).all()
                # we are starting new sequences
                seq_pos = torch.full_like(
                    input_final_seq_pos, self.start_seq_pos + increment - 1
                )
                # seq_pos corresponds to position of previously generated token in the sequence
                # when increment is 1, seq_pos is self.start_seq_pos
            else:
                if increment == 1:
                    print(
                        f"Warning: not sampling a new sequence, check inputs if this is desired behaviour ({kwargs['seq_pos']}, {input_ids})"
                    )
                seq_pos = input_final_seq_pos + increment
            inputs["seq_pos"] = seq_pos
            bsz = input_final_seq_pos.shape[0]
            if self.embed_coords:
                assert kwargs["coords"].ndim == 4  # b, l, n, 3
                inputs["coords"] = torch.full(
                    (
                        bsz,
                        1,
                    )
                    + kwargs["coords"].shape[-2:],
                    0.0,
                ).to(kwargs["coords"])
            if self.embed_sequence_index:
                assert not "start_sequence_index" in kwargs
                # n.b. input_ids is prompt + completion
                prompt_sequence_index = self.compute_sequence_index(
                    input_ids, start_sequence_index=0
                )
                # increment sequence index if prev token was sep
                start_sequence_index = torch.where(
                    input_ids[:, -1] == self.tokenizer.sep_token_id,
                    prompt_sequence_index[:, -1] + 1,
                    prompt_sequence_index[:, -1],
                )
                inputs["start_sequence_index"] = start_sequence_index
        else:
            inputs["seq_pos"] = kwargs["seq_pos"]
            if self.embed_coords:
                inputs["coords"] = kwargs["coords"]
        return inputs

    def compute_sequence_index(self, input_ids, start_sequence_index=0):
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

    def embed_inputs(
        self,
        input_ids: Optional[torch.LongTensor],
        seq_pos: Optional[torch.LongTensor] = None,
        coords: Optional[torch.FloatTensor] = None,
        start_sequence_index: int = 0,
    ):
        # n.b. we need to be careful about what happens when caching.
        # I think in that case input_ids should just be the continuation
        # and inputs_embeds should also.
        # different models will have different token embedders.
        # we assume (which is case for e.g. gpt2 and mistral)
        # that the model will itself add its own position embeddings to inputs_embeds
        assert input_ids.ndim == 2
        # gpt2 code
        # elif input_ids is not None:
        #     self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
        #     input_shape = input_ids.size()
        #     input_ids = input_ids.view(-1, input_shape[-1])
        #     batch_size = input_ids.shape[0]

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
            sequence_index = self.compute_sequence_index(
                input_ids, start_sequence_index=start_sequence_index
            )
            inputs_embeds += self.sequence_index_embedding(sequence_index)

        return inputs_embeds

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
        start_sequence_index: Optional[
            Union[torch.Tensor, int]
        ] = None,  # index of sequence within document. modify when using cache.
        **kwargs,  # e.g. labels
    ):
        if (
            use_cache and not isinstance(past_key_values, Cache) and not self.training
        ):  # kept for BC (non `Cache` `past_key_values` inputs)
            assert (
                past_key_values is None
            ), "We need to know input ids, which can't be passed in legacy format"
            past_key_values = UpdatedDynamicCache()

        if past_key_values is not None:
            assert isinstance(past_key_values, UpdatedDynamicCache)
            # TODO: only if use cahce?
            past_key_values.update_inputs(input_ids)

        assert (
            inputs_embeds is None
        ), "Do not pass pre-computed embeddings to this class"
        if self.embed_sequence_index and past_key_values is not None:
            assert (
                start_sequence_index is not None
            ), "Must pass start_sequence_index if using sequence index embeddings with cache"
        elif start_sequence_index is None:
            start_sequence_index = 0
        inputs_embeds = self.embed_inputs(
            input_ids,
            seq_pos=seq_pos,
            coords=coords,
            start_sequence_index=start_sequence_index,
        )

        # TODO: test these; make sure they get called during generation for example.
        if self.pass_constant_position_ids_for_global_index:
            assert position_ids is None
            position_ids = torch.full_like(input_ids, 10).long()
        if self.pass_sequence_position_ids_for_global_index:
            assert position_ids is None
            assert seq_pos is not None
            position_ids = seq_pos

        return super().forward(
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
