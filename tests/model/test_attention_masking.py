import os

import numpy as np
import pytest
import torch
from transformers.models.llama.modeling_llama import (
    _prepare_4d_causal_attention_mask_with_cache_position,
)

# from src.models.llama import (
#     _prepare_4d_causal_attention_mask_with_cache_position as mod_prepare_4d_causal_attention_mask_with_cache_position,
# )
from src.constants import BASEDIR
from src.data import preprocessing
from src.data.datasets import ProteinDatasetConfig, load_protein_dataset
from src.data.utils import CustomDataCollator
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
    masked_model = load_named_model(
        "llama_tiny",
        overrides=["model.config.attention_mask_type=causal"],
        tokenizer=profam_tokenizer_noseqpos,
    )
    masked_model.eval()
    sd = masked_model.state_dict()
    # instantiate wrapped and unwrapped models with same config and weights
    model = load_named_model("llama_tiny", tokenizer=profam_tokenizer_noseqpos)
    model.eval()
    model.load_state_dict(sd, strict=False)  # token embedder causes mismatch

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


def test_bidirectional_attention_masking(proteingym_batch, profam_tokenizer_noseqpos):
    masked_model = load_named_model(
        "llama_tiny",
        overrides=["model.config.attention_mask_type=bidirectional"],
        tokenizer=profam_tokenizer_noseqpos,
    )
    masked_model.eval()
    outputs = masked_model.forward(
        input_ids=proteingym_batch["input_ids"],
        output_attentions=True,
    )
    # The attention mask should be zeros everywhere if not using padding

    assert not (outputs.attentions[-1] == 0).any()
    # TODO: test padding-aware attention mask


@pytest.fixture()
def foldseek_interleaved_structure_sequence_batch(
    profam_tokenizer,
):
    max_tokens = 2048
    preprocessing_cfg = preprocessing.PreprocessingConfig(
        keep_insertions=True,
        to_upper=True,
        keep_gaps=False,
        use_msa_pos=False,
    )
    parquet_3di_processor = preprocessing.ParquetStructurePreprocessor(
        config=preprocessing_cfg,
        structure_tokens_col="msta_3di",
        interleave_structure_sequence=True,
    )
    cfg = ProteinDatasetConfig(
        name="foldseek",
        preprocessor=parquet_3di_processor,
        data_path_pattern="foldseek_struct/0.parquet",
        is_parquet=True,
    )
    data = load_protein_dataset(
        cfg,
        tokenizer=profam_tokenizer,
        max_tokens=max_tokens,
        data_dir=os.path.join(BASEDIR, "data/example_data"),
        shuffle=False,
        feature_names=["input_ids", "attention_mask", "labels", "plddts", "coords"],
    )
    datapoint = next(iter(data))
    collator = CustomDataCollator(tokenizer=profam_tokenizer, mlm=False)
    return collator([datapoint])


def test_prefix_lm_attention_masking(
    foldseek_interleaved_structure_sequence_batch, profam_tokenizer_noseqpos
):
    # This is a bit tricky - prefix requires interleaved batch
    masked_model = load_named_model(
        "llama_tiny",
        overrides=["model.config.attention_mask_type=prefix-lm"],
        tokenizer=profam_tokenizer_noseqpos,
    )
    masked_model.eval()
    print(
        foldseek_interleaved_structure_sequence_batch["input_ids"].shape,
        torch.argwhere(
            foldseek_interleaved_structure_sequence_batch["input_ids"]
            == profam_tokenizer_noseqpos.seq_struct_sep_token_id
        ),
    )
    outputs = masked_model.forward(
        input_ids=foldseek_interleaved_structure_sequence_batch["input_ids"],
        output_attentions=True,
        use_cache=True,
    )
    print(outputs.attentions[-1])
    assert 1 == 0
