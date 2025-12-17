import numpy as np
import pytest

from src.data.objects import ProteinDocument
from src.data.processors.transforms import (
    preprocess_raw_sequences_sampling_to_max_tokens,
    random_crop,
)


@pytest.fixture
def protein_document():
    sequences = ["M" * 100, "A" * 150, "G" * 200]  # Sequences longer than max_tokens
    accessions = ["P12345", "Q67890", "R23456"]

    return ProteinDocument(
        sequences=sequences,
        accessions=accessions,
    )


def test_sample_to_max_tokens_exceeds_max(protein_document, profam_tokenizer):
    max_tokens = 50  # Set max_tokens less than any sequence length
    for _ in range(10):
        # 10 times to cover random differences in algo
        sampled_proteins = preprocess_raw_sequences_sampling_to_max_tokens(
            protein_document,
            tokenizer=profam_tokenizer,
            max_tokens=max_tokens,
        )

        # Check that the sampled_proteins contains only one truncated sequence
        assert len(sampled_proteins) == 1
        assert (
            len(sampled_proteins.sequences[0])
            == max_tokens - profam_tokenizer.num_start_tokens - 1
        )


def test_random_crop():
    proteins = ProteinDocument(
        sequences=["ABCDEFGHIJKLMNOPQRSTUVWXYZ"],
        accessions=["P12345"],
    )
    cropped_proteins = random_crop(proteins, min_length=3, max_length=3, crop_prob=1.0)
    assert len(cropped_proteins) == 1
    assert len(cropped_proteins.sequences[0]) == 3
