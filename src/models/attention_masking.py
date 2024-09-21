from typing import Optional

import torch


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
        full_prefix_index[:, last_seq_end + 1 : cache_position[0]] = 1

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
