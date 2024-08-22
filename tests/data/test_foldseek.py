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


# TODO: write full manual test for coords concatenation and padding etc.
def test_foldseek_interleaved_tokenization(
    foldseek_interleaved_structure_sequence_batch,
    foldseek_interleaved_structure_sequence_datapoint,
    profam_tokenizer_seqpos,
):
    num_sequences_in_batch = (
        foldseek_interleaved_structure_sequence_batch["input_ids"]
        == profam_tokenizer_seqpos.sep_token_id
    ).sum()
    first_seq_start = torch.argwhere(
        (
            foldseek_interleaved_structure_sequence_batch["input_ids"][0]
            == profam_tokenizer_seqpos.seq_struct_sep_token_id
        )
    )[0]
    batch_seqs = foldseek_interleaved_structure_sequence_datapoint["sequences"][
        :num_sequences_in_batch
    ]
    # TODO: make a proper test by stitching together manually encoded sequences and 3dis
    print(batch_seqs, len(batch_seqs[0]), first_seq_start)
    print(
        profam_tokenizer_seqpos.encode_sequences(
            foldseek_interleaved_structure_sequence_datapoint["sequences"][
                :num_sequences_in_batch
            ]
        ).input_ids[2:52]
    )
    print(
        foldseek_interleaved_structure_sequence_batch["input_ids"][
            0, first_seq_start + 1 : first_seq_start + 51
        ]
    )
    print(foldseek_interleaved_structure_sequence_batch["input_ids"][0, :30])
    assert 0 == 1
