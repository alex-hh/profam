import numpy as np
import pytest

from src.data.objects import ProteinDocument, check_array_lengths
from src.data.processors.transforms import (
    convert_raw_sequence_adding_positions,
    preprocess_sequences_sampling_to_max_tokens,
)
from src.data.tokenizers import ProFamTokenizer


@pytest.fixture
def protein_document():
    sequences = ["M" * 100, "A" * 150, "G" * 200]  # Sequences longer than max_tokens
    accessions = ["P12345", "Q67890", "R23456"]
    plddts = [np.random.rand(100), np.random.rand(150), np.random.rand(200)]
    backbone_coords = [
        np.random.rand(100, 4, 3),
        np.random.rand(150, 4, 3),
        np.random.rand(200, 4, 3),
    ]
    backbone_coords_masks = [
        np.ones((100, 4)),
        np.ones((150, 4)),
        np.ones((200, 4)),
    ]
    structure_tokens = ["X" * 100, "Y" * 150, "Z" * 200]

    return ProteinDocument(
        sequences=sequences,
        accessions=accessions,
        plddts=plddts,
        backbone_coords=backbone_coords,
        backbone_coords_masks=backbone_coords_masks,
        structure_tokens=structure_tokens,
    )


def test_sample_to_max_tokens_exceeds_max(protein_document, profam_tokenizer):
    max_tokens = 50  # Set max_tokens less than any sequence length
    for _ in range(10):
        # 10 times to cover random differences in algo
        sampled_proteins = preprocess_sequences_sampling_to_max_tokens(
            protein_document,
            tokenizer=profam_tokenizer,
            sequence_converter=convert_raw_sequence_adding_positions,
            max_tokens=max_tokens,
        )

        # Check that the sampled_proteins contains only one truncated sequence
        assert len(sampled_proteins) == 1
        assert (
            len(sampled_proteins.sequences[0])
            == max_tokens - profam_tokenizer.num_start_tokens - 1
        )
        sequence_lengths = check_array_lengths(
            sampled_proteins.sequences,
            sampled_proteins.modality_masks,
        )
        assert (
            sequence_lengths[0][0] == max_tokens - profam_tokenizer.num_start_tokens - 1
        )
