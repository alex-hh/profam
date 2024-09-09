import torch

from src.data.objects import ProteinDocument


def test_compute_sequence_index(default_model, profam_tokenizer):
    sequences = ["ARC", "MKLLL", "MKPP"]
    document = ProteinDocument(sequences=sequences)
    tokenized = profam_tokenizer.encode(document)
    sequence_indices = default_model.model.compute_sequence_index(tokenized.input_ids)
    expected_sequence_indices = torch.tensor(
        [[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2]]
    )
    assert (sequence_indices == expected_sequence_indices).all()
