import os

import numpy as np
import pandas as pd
import torch

from src.constants import BASEDIR
from src.data.preprocessing import ParquetStructurePreprocessor
from src.data.transforms import interleave_structure_sequence
from src.models.inference import InterleavedInverseFoldingPromptBuilder


def test_representative_inverse_folding(profam_tokenizer):
    df = pd.read_parquet(
        os.path.join(BASEDIR, "data/example_data/foldseek_representatives/0.parquet")
    )
    example = df.iloc[0]
    preprocessor = ParquetStructurePreprocessor(
        structure_tokens_col=None,
        interleave_structure_sequence=True,
        infer_representative_from_identifier=True,
    )
    proteins = preprocessor.build_document(
        example, profam_tokenizer, max_tokens=1536, shuffle=False
    )
    rep_seq = proteins[0].sequence
    expected_coords = np.concatenate(
        (np.zeros((2, 4, 3)), proteins[0].backbone_coords, np.zeros((1, 4, 3))), axis=0
    )
    assert len(proteins) == 1
    prompt_builder = InterleavedInverseFoldingPromptBuilder(
        preprocessor=preprocessor,
        max_tokens=1536,
        representative_only=True,
    )
    example = prompt_builder(proteins, profam_tokenizer)

    expected_tokens = torch.tensor(
        [
            profam_tokenizer.convert_tokens_to_ids("[RAW]"),
            profam_tokenizer.bos_token_id,
        ]
        + [profam_tokenizer.mask_token_id] * len(rep_seq)
        + [
            profam_tokenizer.seq_struct_sep_token_id,
        ]
    )
    assert (example["input_ids"] == expected_tokens).all(), expected_tokens
    assert (example["coords"].numpy() == expected_coords).all(), expected_coords
    # TODO: test the coords etc.
    assert 1 == 0

    # TODO: actually write a test of the sliced prompt input ids (e.g. all mask token ids here.)
