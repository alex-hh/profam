import torch
from transformers.models.llama.modeling_llama import (
    _prepare_4d_causal_attention_mask_with_cache_position,
)

from src.models.llama import (
    _prepare_4d_causal_attention_mask_with_cache_position as mod_prepare_4d_causal_attention_mask_with_cache_position,
)


def test_prepare_4d_causal_attention_mask_with_cache_position():
    sequence_length = 5
    target_length = 10
    past_seen_length = target_length - sequence_length
    batch_size = 3
    cache_position = torch.arange(past_seen_length, past_seen_length + sequence_length)
    our_mask = mod_prepare_4d_causal_attention_mask_with_cache_position(
        attention_mask=None,
        sequence_length=sequence_length,
        target_length=target_length,
        cache_position=cache_position,
        dtype=torch.float32,
        device=torch.device("cpu"),
        batch_size=batch_size,
        min_dtype=torch.finfo(torch.float32).min,
    )

    llama_mask = _prepare_4d_causal_attention_mask_with_cache_position(
        attention_mask=None,
        sequence_length=sequence_length,
        target_length=target_length,
        cache_position=cache_position,
        dtype=torch.float32,
        device=torch.device("cpu"),
        batch_size=3,
        min_dtype=torch.finfo(torch.float32).min,
    )

    assert torch.allclose(our_mask, llama_mask)

    sequence_length = 14
    target_length = 14
    past_seen_length = target_length - sequence_length
    attention_mask = torch.ones(batch_size, target_length)
    attention_mask[:, -5:] = 0  # simulate padding
    cache_position = torch.arange(past_seen_length, past_seen_length + sequence_length)
    our_mask = mod_prepare_4d_causal_attention_mask_with_cache_position(
        attention_mask=attention_mask,
        sequence_length=sequence_length,
        target_length=target_length,
        cache_position=cache_position,
        dtype=torch.float32,
        device=torch.device("cpu"),
        batch_size=batch_size,
        min_dtype=torch.finfo(torch.float32).min,
    )

    llama_mask = _prepare_4d_causal_attention_mask_with_cache_position(
        attention_mask=attention_mask,
        sequence_length=sequence_length,
        target_length=target_length,
        cache_position=cache_position,
        dtype=torch.float32,
        device=torch.device("cpu"),
        batch_size=3,
        min_dtype=torch.finfo(torch.float32).min,
    )
    assert torch.allclose(our_mask, llama_mask)
