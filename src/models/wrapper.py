from typing import List, Optional, Union

import torch
from torch import nn

from src.utils.tokenizers import ProFamTokenizer
from src.utils.utils import nested_getattr

# TODO: really we want to write our own cache class.
# we can just overwrite update and add relevant attributes.
# we then need to make sure that we instantiate the right type of cache
# in relevant places (e.g. in the model forward with use_cache True.)

# Isnt cache position always just -sequence_length:?
# Test that when using this we get same as when using auto-generated mask.
def _prepare_causal_4d_binary_mask(
    attention_mask_2d: Optional[torch.Tensor],
    sequence_length: int,
    target_length: int,
    device: torch.device,
    cache_position: torch.Tensor,
    batch_size: int,
):
    assert cache_position.shape[0] == sequence_length
    full_attention_mask_2d = torch.ones(batch_size, target_length, device=device)
    if attention_mask_2d is not None:
        full_attention_mask_2d[:, : attention_mask_2d.shape[1]] = attention_mask_2d
    full_attention_mask_2d = full_attention_mask_2d[:, None, None, :]
    causal_mask = (
        torch.arange(target_length, device=device) <= cache_position.unsqueeze(1)
    )[None, None].expand(
        batch_size, -1, -1, -1
    )  # bsz, head_dim, sequence_length, target_length
    return causal_mask.int() * full_attention_mask_2d


def _prepare_bidirectional_4d_binary_mask(
    attention_mask_2d: Optional[torch.Tensor],
    sequence_length: int,
    target_length: int,
    device: torch.device,
    batch_size: int,
):
    full_attention_mask_2d = torch.ones(batch_size, target_length, device=device)
    if attention_mask_2d is not None:
        full_attention_mask_2d[:, : attention_mask_2d.shape[1]] = attention_mask_2d
    full_attention_mask_2d = full_attention_mask_2d[:, None, None, :].expand(
        -1, -1, sequence_length, -1
    )
    return full_attention_mask_2d


def _prepare_intra_separator_bidirectional_4d_binary_mask(
    input_ids: torch.Tensor,
    last_cached_seq_start: torch.Tensor,  # b, equivalent to current sequence start...
    attention_mask_2d: Optional[torch.Tensor],
    sequence_length: int,
    target_length: int,
    device: torch.device,
    cache_position: torch.Tensor,
    batch_size: int,
    sep_token_id: int,
):
    """Mask for attention within a sequence/document, but not between sequences/documents."""
    raise NotImplementedError()
    assert cache_position.shape[0] == sequence_length
    full_attention_mask_2d = torch.ones(batch_size, target_length, device=device)
    if attention_mask_2d is not None:
        full_attention_mask_2d[:, : attention_mask_2d.shape[1]] = attention_mask_2d
    full_attention_mask_2d = full_attention_mask_2d[:, None, None, :]
    sequence_index = torch.cumsum(input_ids == sep_token_id, dim=-1)
    intra_mask_4d = torch.zeros(batch_size, sequence_length, target_length)
    intra_mask_4d[:, :, cache_position] = (
        sequence_index[:, None, :] == sequence_index[:, :, None]
    ).int()
    intra_mask_4d[:, :, last_sep_position:sequence_length] = first_sequence_mask[
        None, :, None
    ]
    last_cached_seq_end = sequence_index  # b
    intra_mask_4d[
        :, :last_cached_seq_end, last_cached_seq_start : cache_position[0]
    ] = 1
    # intra_mask_4d = intra_mask_4d[:, cache_position, :]  would be right if input ids contained everything.
    return intra_mask_4d[:, None] * full_attention_mask_2d


# One test for this is that if we have a single sequence, we should observe same as standard causal mask.
def _prepare_intra_separator_causal_4d_binary_mask(
    input_ids: torch.Tensor,
    last_cached_seq_start: torch.Tensor,  # b, equivalent to current sequence start...
    attention_mask_2d: Optional[torch.Tensor],
    sequence_length: int,
    target_length: int,
    device: torch.device,
    cache_position: torch.Tensor,
    batch_size: int,
    sep_token_id: int,
):
    raise NotImplementedError("check sep ids get assigned correctly")
    bidirectional_intra_separator_mask = (
        _prepare_intra_separator_bidirectional_4d_binary_mask(
            input_ids,
            last_cached_seq_start,
            attention_mask_2d,
            sequence_length,
            target_length,
            device,
            cache_position,
            batch_size,
            sep_token_id,
        )
    )
    causal_mask = (
        torch.arange(target_length, device=device) <= cache_position.unsqueeze(1)
    )[None].expand(
        batch_size, -1, -1
    )  # sequence_length, target_length
    return causal_mask[:, None] * bidirectional_intra_separator_mask


def _prepare_sequence_causal_bidirectional_4d_binary_mask(
    input_ids: torch.LongTensor,
    last_cached_seq_start: Optional[int],
    attention_mask_2d: Optional[torch.Tensor],
    sequence_length: int,
    target_length: int,
    device: torch.device,
    cache_position: torch.Tensor,
    batch_size: int,
    sep_token_id: int,
):
    """Same-sequence attention is bidirectional, different-sequence attention is causal."""
    raise NotImplementedError("Check sep ids get assigned correctly.")
    if attention_mask_2d is not None:
        raise NotImplementedError()
    # To create a sequence-causal mask, we can just look at sequence index.
    sequence_index = (
        torch.cumsum(input_ids == sep_token_id, dim=-1) + 1
    )  # batch_size, sequence_length
    full_sequence_index = torch.zeros(batch_size, target_length, device=device)
    full_sequence_index[:, cache_position] = sequence_index
    full_sequence_index[:, last_cached_seq_start : cache_position[0]] = 1
    # this is automatically bidirectional within-sequence
    sequence_causal_mask = sequence_index[:, None, :] <= full_sequence_index[:, :, None]
    return sequence_causal_mask.int()[:, None]


def _prepare_prefix_lm_4d_binary_mask(
    input_ids: torch.LongTensor,
    last_cached_prefix_start: Optional[int],
    last_cached_seq_start: Optional[int],
    attention_mask_2d: Optional[torch.Tensor],
    sequence_length: int,
    target_length: int,
    device: torch.device,
    cache_position: torch.Tensor,
    batch_size: int,
    prefix_separator_token_id: int,
    item_separator_token_id: int,
):
    """If you're in the prefix, you can do bidirectional attention. If you're in the suffix, you can't.

    This attention mask is likely particularly useful for multimodal structure + sequence models:
    bidirectional attention within the structure and causal attention within the sequence.

    Current version allows attention between items (proteins).
    """
    input_ids
    # we can start with a standard causal mask. then 'fill in' prefix blocks with bidirectional attention.
    causal_mask = _prepare_causal_4d_binary_mask(
        attention_mask_2d,
        sequence_length,
        target_length,
        device,
        cache_position,
        batch_size,
    )
    # now we identify pairs of positions that belong to the same prefix. currently we assume
    # that no uncached position can be in a prefix.
    raise NotImplementedError("Check sep /pref ids get assigned correctly.")
    prefix_index = torch.cumsum(input_ids == prefix_separator_token_id, dim=-1) + 1
    sequence_index = torch.cumsum(input_ids == item_separator_token_id, dim=-1) + 1
    is_prefix = prefix_index == sequence_index
    prefix_index = torch.where(is_prefix, prefix_index, 0)
    full_prefix_index = torch.zeros(batch_size, target_length, device=device)
    full_prefix_index[:, cache_position] = prefix_index
    if last_prefix_end == last_seq_end:
        assert last_prefix_end is None and last_seq_end is None
        # we have a new sequence. we're therefore in a prefix which ends at the first prefix separator.
        # a prefix is somewhere where the prefix index is equal to the sequence index
        pass

    elif last_seq_end > last_prefix_end:
        # we start with a prefix
        full_prefix_index[:,last_seq_end+1:cache_position[0]] = 1

    else:
        # we start with a suffix
        pass

    same_prefix_mask = prefix_index[:, None, :] == full_prefix_index[:, :, None]
    causal_mask.masked_fill(same_prefix_mask, 1)
    return causal_mask


def is_integer(tensor: torch.Tensor, signed: bool | None = None) -> bool:
    """Determines if a PyTorch tensor has an integer dtype.

    Source:
    https://github.com/pytorch/pytorch/issues/52161
    It also can force `tensor` to be singed or unsinged.

    Parameters
    ----------
    tensor
        The tensor to check.
    signed
        Determines which dtypes are allowed for `tensor`:

        - If ``None`` both unsinged and signed integer will be allowed.

        - If ``False`` only unsigned dtypes will be allowed.

        - If ``True`` only signed dtypes will be allowed.

    Returns
    -------
    bool
        ``True`` if the input tensor satisfies the requested condition, ``False``
        otherwise.

    """
    uint_types = [torch.uint8]
    sint_types = [torch.int8, torch.int16, torch.int32, torch.int64]
    if signed is None:
        return tensor.dtype in uint_types + sint_types
    elif signed:
        return tensor.dtype in sint_types
    else:
        return tensor.dtype in uint_types


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
    assert cache_position.shape[0] == sequence_length
    # original code was optimised for memory - make sure this is too.
    # for example - masked fill might be better but requires inverted mask
    if attention_mask is not None:
        assert torch.is_floating_point(attention_mask) or is_integer(
            attention_mask
        ), "Attention mask must be numeric"
    if attention_mask is None or attention_mask.ndim == 2:
        # N.B. the combination of binary and non-binary masks in the original code here is pretty confusing.
        # we try to first build required binary mask, then convert to a bias mask.
        causal_mask = (
            torch.arange(target_length, device=device) <= cache_position.unsqueeze(1)
        )[None].expand(
            batch_size, -1, -1
        )  # sequence_length, target_length
        if attention_mask is not None:
            if attention_mask.shape[-1] < target_length:
                full_attention_mask = torch.ones(
                    batch_size, target_length, device=device
                )
                full_attention_mask[:, : attention_mask.shape[-1]] = attention_mask
            else:
                full_attention_mask = attention_mask
            causal_mask = causal_mask & full_attention_mask[:, None, :].bool()
        causal_mask = causal_mask[:, None]  # add head dim
    elif (
        torch.isin(
            attention_mask, torch.tensor([0, 1], device=attention_mask.device)
        ).all()
        and not (attention_mask == 0).all()
    ):
        # if we pass all 0s there is ambiguity, but we assume it means a bias mask, since it would prevent any attention.
        causal_mask = attention_mask.bool()
    else:
        causal_mask = attention_mask

    assert causal_mask.ndim == 4
    # TODO: check if attention mask is binary at this point.
    # In this case we assume that the mask comes already in inverted form and requires no inversion or slicing.
    if causal_mask.dtype == torch.bool:
        # causal mask is binary mask with 1s where attention is allowed
        # invert and use -inf to mask out disallowed attentions
        causal_mask = causal_mask.logical_not().to(dtype) * min_dtype

    # otherwise bias mask already: pass on through
    return causal_mask


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
            return _prepare_causal_4d_binary_mask(
                attention_mask_2d,
                sequence_length,
                target_length,
                device,
                cache_position,
                batch_size,
            )
        elif self.attention_mask_type == "bidirectional":
            return _prepare_bidirectional_4d_binary_mask(
                attention_mask_2d,
                sequence_length,
                target_length,
                device,
                cache_position,
                batch_size,
            )
        elif self.attention_mask_type == "sequence":
            assert input_ids is not None
            return _prepare_intra_separator_4d_binary_mask(
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
            return _prepare_prefix_lm_4d_binary_mask(
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
