import numpy as np
import torch
from transformers.models.llama.modeling_llama import (
    _prepare_4d_causal_attention_mask_with_cache_position,
)

# from src.models.llama import (
#     _prepare_4d_causal_attention_mask_with_cache_position as mod_prepare_4d_causal_attention_mask_with_cache_position,
# )
from src.models.llama import LlamaLitModule
from src.models.utils import load_named_model


# TODO: rewrite this
# def test_prepare_4d_causal_attention_mask_with_cache_position():
#     # PASSING NOTHING AS INPUT ATTENTION MASK
#     sequence_length = 5
#     target_length = 10
#     past_seen_length = target_length - sequence_length
#     batch_size = 3
#     cache_position = torch.arange(past_seen_length, past_seen_length + sequence_length)
#     our_mask = mod_prepare_4d_causal_attention_mask_with_cache_position(
#         attention_mask=None,
#         sequence_length=sequence_length,
#         target_length=target_length,
#         cache_position=cache_position,
#         dtype=torch.float32,
#         device=torch.device("cpu"),
#         batch_size=batch_size,
#         min_dtype=torch.finfo(torch.float32).min,
#     )

#     llama_mask = _prepare_4d_causal_attention_mask_with_cache_position(
#         attention_mask=None,
#         sequence_length=sequence_length,
#         target_length=target_length,
#         cache_position=cache_position,
#         dtype=torch.float32,
#         device=torch.device("cpu"),
#         batch_size=3,
#         min_dtype=torch.finfo(torch.float32).min,
#     )

#     # PASSING 2D MASK
#     assert torch.allclose(our_mask, llama_mask)

#     sequence_length = 14
#     target_length = 14
#     past_seen_length = target_length - sequence_length
#     attention_mask = torch.ones(batch_size, target_length)
#     attention_mask[:, -5:] = 0  # simulate padding
#     cache_position = torch.arange(past_seen_length, past_seen_length + sequence_length)
#     our_mask = mod_prepare_4d_causal_attention_mask_with_cache_position(
#         attention_mask=attention_mask,
#         sequence_length=sequence_length,
#         target_length=target_length,
#         cache_position=cache_position,
#         dtype=torch.float32,
#         device=torch.device("cpu"),
#         batch_size=batch_size,
#         min_dtype=torch.finfo(torch.float32).min,
#     )

#     llama_mask = _prepare_4d_causal_attention_mask_with_cache_position(
#         attention_mask=attention_mask,
#         sequence_length=sequence_length,
#         target_length=target_length,
#         cache_position=cache_position,
#         dtype=torch.float32,
#         device=torch.device("cpu"),
#         batch_size=3,
#         min_dtype=torch.finfo(torch.float32).min,
#     )
#     assert torch.allclose(our_mask, llama_mask)

#     # PASSING BINARY 4D MASK
#     causal_mask = (
#         (torch.arange(target_length) <= cache_position.unsqueeze(1))[None, None]
#         .expand(batch_size, -1, -1, -1)
#         .int()
#     )  # b, h, sequence_length, target_length
#     our_mask = mod_prepare_4d_causal_attention_mask_with_cache_position(
#         attention_mask=causal_mask,
#         sequence_length=sequence_length,
#         target_length=target_length,
#         cache_position=cache_position,
#         dtype=torch.float32,
#         device=torch.device("cpu"),
#         batch_size=3,
#         min_dtype=torch.finfo(torch.float32).min,
#     )

#     llama_mask = _prepare_4d_causal_attention_mask_with_cache_position(
#         attention_mask=None,
#         sequence_length=sequence_length,
#         target_length=target_length,
#         cache_position=cache_position,
#         dtype=torch.float32,
#         device=torch.device("cpu"),
#         batch_size=3,
#         min_dtype=torch.finfo(torch.float32).min,
#     )
#     assert torch.allclose(our_mask, llama_mask)


def test_custom_attention_masking(proteingym_batch, profam_tokenizer_noseqpos):
    masked_model = load_named_model("llama_tiny", overrides=["+attention_mask_type=causal"], tokenizer=profam_tokenizer_noseqpos)
    sd = masked_model.state_dict()
    # instantiate wrapped and unwrapped models with same config and weights
    model = load_named_model("llama_tiny", tokenizer=profam_tokenizer_noseqpos)
    model.load_state_dict(sd)
    masked_scores = masked_model.score_seqs(
        input_ids=proteingym_batch["input_ids"],
        completion_ids=proteingym_batch["completion_ids"][:, :2],
        use_cache=True,
        batch_size=1,
        input_seq_pos=None,
        completion_seq_pos=None,
    )
    scores = model.score_seqs(
        input_ids=proteingym_batch["input_ids"],
        completion_ids=proteingym_batch["completion_ids"][:, :2],
        use_cache=True,
        batch_size=1,
        input_seq_pos=None,
        completion_seq_pos=None,
    )

    assert np.isclose(masked_scores, scores).all()
