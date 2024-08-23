import numpy as np
import pandas as pd
import pytest
import torch

from src.data.pdb import get_atom_coords_residuewise, load_structure
from src.data.preprocessing import backbone_coords_from_example


@pytest.fixture
def foldseek_df():
    df = pd.read_parquet("data/example_data/foldseek_struct/0.parquet")
    return df


def test_foldseek_backbone_loading(foldseek_df):
    for _, row in foldseek_df.iterrows():
        foldseek_example = row.to_dict()
        # Q. why does this successfully load the backbone coordinates as arrays?
        backbone_coords = backbone_coords_from_example(foldseek_example)
        for seq, acc, recons_coords in zip(
            foldseek_example["sequences"],
            foldseek_example["accessions"],
            backbone_coords,
        ):
            pdbfile = (
                "data/example_data/foldseek_struct/0/AF-{}-F1-model_v4.pdb".format(
                    acc, acc
                )
            )
            structure = load_structure(pdbfile, chain="A")
            coords = get_atom_coords_residuewise(["N", "CA", "C", "O"], structure)
            assert np.allclose(coords, recons_coords)
            assert len(coords) == len(seq)


def stitch_tokens(tokenizer, struct_tokens, seq_tokens):
    assert len(struct_tokens) == len(seq_tokens)
    tensors = []
    for struct, seq in zip(struct_tokens, seq_tokens):
        tensors += [
            struct,
            torch.full((1,), tokenizer.convert_tokens_to_ids("[SEQ-STRUCT-SEP]")),
            seq,
        ]
        tensors.append(torch.full((1,), tokenizer.convert_tokens_to_ids("[SEP]")))
    return torch.cat(tensors, dim=0)


# TODO: write full manual test for coords concatenation and padding etc.
def test_foldseek_interleaved_tokenization(
    foldseek_interleaved_structure_sequence_batch,
    foldseek_interleaved_structure_sequence_datapoint,
    profam_tokenizer,
):
    num_sequences_in_batch = (
        foldseek_interleaved_structure_sequence_batch["input_ids"]
        == profam_tokenizer.sep_token_id
    ).sum()

    batch_seqs = foldseek_interleaved_structure_sequence_datapoint["sequences"][
        :num_sequences_in_batch
    ]
    batch_3dis = [
        s.replace("-", "").lower()
        for s in foldseek_interleaved_structure_sequence_datapoint["msta_3di"][
            :num_sequences_in_batch
        ]
    ]
    # TODO: make a proper test by stitching together manually encoded sequences and 3dis
    individual_seq_tokens = [
        profam_tokenizer.encode_completions(
            [s], bos_token="", eos_token=""
        ).input_ids[0]
        for s in batch_seqs
    ]
    individual_3d_tokens = [
        profam_tokenizer.encode_completions(
            [s_3d], bos_token="", eos_token=""
        ).input_ids[0]
        for s_3d in batch_3dis
    ]
    stitched_tokens = torch.tensor(
        profam_tokenizer.convert_tokens_to_ids(
            ["[RAW]", profam_tokenizer.bos_token]
        )
    )
    stitched_tokens = torch.cat(
        [
            stitched_tokens,
            stitch_tokens(
                profam_tokenizer, individual_3d_tokens, individual_seq_tokens
            ),
        ],
        dim=0,
    )

    assert (
        foldseek_interleaved_structure_sequence_batch["input_ids"][
            0, : stitched_tokens.shape[0]
        ]
        == stitched_tokens
    ).all()
