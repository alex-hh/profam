import numpy as np
import pytest

from src.data.objects import ProteinDocument, check_array_lengths
from src.data.transforms import sample_to_max_tokens
from src.utils.tokenizers import ProFamTokenizer


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
        np.ones((100, 4, 3)),
        np.ones((150, 4, 3)),
        np.ones((200, 4, 3)),
    ]
    structure_tokens = ["X" * 100, "Y" * 150, "Z" * 200]
    document_ids = [np.arange(100), np.arange(150), np.arange(200)]

    return ProteinDocument(
        sequences=sequences,
        accessions=accessions,
        plddts=plddts,
        backbone_coords=backbone_coords,
        backbone_coords_masks=backbone_coords_masks,
        structure_tokens=structure_tokens,
        document_ids=document_ids,
    )


def test_sample_to_max_tokens_exceeds_max(protein_document, profam_tokenizer):
    max_tokens = 50  # Set max_tokens less than any sequence length
    sampled_proteins = sample_to_max_tokens(
        protein_document, max_tokens, tokenizer=profam_tokenizer
    )

    # Check that the sampled_proteins contains only one truncated sequence
    assert len(sampled_proteins) == 1
    assert (
        len(sampled_proteins.sequences[0])
        == max_tokens - profam_tokenizer.num_start_tokens
    )
    sequence_lengths = check_array_lengths(
        sampled_proteins.sequences,
        sampled_proteins.document_ids,
        sampled_proteins.modality_masks,
    )
    assert sequence_lengths[0][0] == max_tokens - profam_tokenizer.num_start_tokens
