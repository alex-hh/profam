from typing import Optional

import torch
from transformers.cache_utils import StaticCache


def _prepare_4d_attention_mask_with_cache_position(
    attention_mask: torch.Tensor,
    sequence_length: int,
    target_length: int,
    dtype: torch.dtype,
    device: torch.device,
    min_dtype: float,
    cache_position: torch.Tensor,
    batch_size: int,
    attention_bias: Optional[torch.Tensor] = None,
):
    """
    We assume that the attention mask is one of the following:
        - a 2D binary mask, with 1s indicating keys that can be attended to in the full sequence
        - a 4D binary mask, with 1s indicating permitted attention.
            shape should be [broadcastable to?] (batch_size, head_dim, query_length, key_value_length)
            query_length when using cache is equal to number of uncached tokens

    Creates a causal 4D bias of shape `(batch_size, 1, query_length, key_value_length)` from a 2D mask of shape
    `(batch_size, key_value_length)`, or if the input `attention_mask` is already 4D, do nothing.
    This bias can then be added to the attention logits to achieve the desired attention masking pattern.

    TODO: check whether head dim is supported?

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
        assert is_integer(
            attention_mask
        ), "Attention mask must be integer type for binary masking."
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
    else:
        # if we pass all 0s there is ambiguity, but we assume it means a bias mask, since it would prevent any attention.
        causal_mask = attention_mask.bool()

    assert causal_mask.ndim == 4
    if causal_mask.dtype == torch.bool:
        # causal mask is binary mask with 1s where attention is allowed
        # invert and use -inf to mask out disallowed attentions
        causal_mask = causal_mask.logical_not().to(dtype) * min_dtype

    if attention_bias is not None:
        # add bias to mask
        causal_mask += attention_bias

    return causal_mask


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


def prepare_binary_attention_mask(
    attention_mask_type: str,
    attention_mask_2d: Optional[torch.Tensor],
    new_sequence_length: int,
    device: torch.device,
    cache_position: torch.Tensor,
    batch_size: int,
    past_key_values: Optional[torch.Tensor] = None,
    seq_struct_sep_token_id: Optional[int] = None,
    sep_token_id: Optional[int] = None,
):
    """Because attention mask creation is handled entirely in forward,
    e.g. in LlamaModel._update_causal_mask,
    during generation we don't in principle need to change anything.
    """
    past_seen_tokens = (
        past_key_values.get_seq_length() if past_key_values is not None else 0
    )
    using_static_cache = isinstance(past_key_values, StaticCache)

    if using_static_cache:
        target_length = past_key_values.get_max_length()
    else:
        # Q. why the +1 here? Are we handling it correctly in all cases?
        target_length = (
            attention_mask_2d.shape[-1]
            if isinstance(attention_mask_2d, torch.Tensor)
            else past_seen_tokens + new_sequence_length + 1
        )

    if attention_mask_type == "causal":
        return _prepare_causal_4d_binary_mask(
            attention_mask_2d,
            new_sequence_length,
            target_length,
            device,
            cache_position,
            batch_size,
        )
    else:
        raise ValueError("Unsupported attention mask type", attention_mask_type)
