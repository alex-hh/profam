import torch

from src.data.objects import ProteinDocument


def test_compute_sequence_index(default_model, profam_tokenizer):
    sequences = ["ARC", "MKLLL", "MKPP"]
    document = ProteinDocument(sequences=sequences)
    tokenized = profam_tokenizer.encode(document)
    sequence_indices = default_model.model.compute_sequence_index(
        tokenized.input_ids[None]
    )
    expected_sequence_indices = torch.tensor(
        [[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2]]
    )
    assert (sequence_indices == expected_sequence_indices).all()


def test_prepare_inputs_for_generation(default_model, profam_tokenizer):
    # TODO: also test seq_pos: e.g. currently we cannot handle sep tokens in generated sequences
    default_model.model.embed_sequence_index = True
    sequences = ["ARC", "MKLL", "MKPP"]
    # imagine we are generating a new sequence after the second prompt sequence
    # we have already generated MKPP
    tokenized = profam_tokenizer.encode(
        ProteinDocument(sequences=sequences), add_final_sep=False
    )
    sliced_seq_pos = tokenized.seq_pos[: -len(sequences[-1])]
    print(tokenized.seq_pos, tokenized.seq_pos[-len(sequences[-1])])
    assert sliced_seq_pos[-1].item() == 0
    inputs = default_model.model.prepare_inputs_for_generation(
        tokenized.input_ids[None], seq_pos=sliced_seq_pos[None], use_cache=True
    )
    assert (inputs["start_sequence_index"] == 2).all()
    # assert (inputs["input_ids"] == )

    default_model.embed_sequence_index = False
    assert 1 == 0

    # TODO: imagine we have already generated a bit; imagine we are starting from bos
