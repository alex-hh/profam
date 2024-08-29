import os

import numpy as np
import pandas as pd
import pytest
import torch

from src.constants import BASEDIR
from src.data.pdb import get_atom_coords_residuewise, load_structure
from src.data.preprocessing import backbone_coords_from_example
from src.data.utils import (
    CustomDataCollator,
    ProteinDatasetConfig,
    load_protein_dataset,
)


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
    batch_coords = backbone_coords_from_example(
        foldseek_interleaved_structure_sequence_datapoint
    )[:num_sequences_in_batch]
    batch_plddts = foldseek_interleaved_structure_sequence_datapoint["plddts"][
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
        profam_tokenizer.encode_completions([s], bos_token="", eos_token="").input_ids[
            0
        ]
        for s in batch_seqs
    ]
    individual_3d_tokens = [
        profam_tokenizer.encode_completions(
            [s_3d], bos_token="", eos_token=""
        ).input_ids[0]
        for s_3d in batch_3dis
    ]
    stitched_tokens = torch.tensor(
        profam_tokenizer.convert_tokens_to_ids(["[RAW]", profam_tokenizer.bos_token])
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

    sep_locations = torch.argwhere(
        foldseek_interleaved_structure_sequence_batch["input_ids"][0]
        == profam_tokenizer.sep_token_id
    ).flatten()
    struct_sep_locations = torch.argwhere(
        foldseek_interleaved_structure_sequence_batch["input_ids"][0]
        == profam_tokenizer.convert_tokens_to_ids("[SEQ-STRUCT-SEP]")
    ).flatten()
    assert (
        sep_locations.shape[0]
        == struct_sep_locations.shape[0]
        == num_sequences_in_batch
    )
    struct_start_index = profam_tokenizer.num_start_tokens

    for i in range(num_sequences_in_batch):
        struct_end_index = struct_sep_locations[i]
        seq_start_index = struct_end_index + 1
        seq_end_index = sep_locations[i]
        assert (
            foldseek_interleaved_structure_sequence_batch["coords"][
                0, struct_start_index:struct_end_index
            ]
            == torch.from_numpy(batch_coords[i])
        ).all()
        assert (
            foldseek_interleaved_structure_sequence_batch["plddts"][
                0, seq_start_index:seq_end_index
            ]
            == torch.tensor(batch_plddts[i])
        ).all()
        assert (
            foldseek_interleaved_structure_sequence_batch["coords"][
                0, seq_start_index:seq_end_index
            ]
            == torch.from_numpy(batch_coords[i])
        ).all()
        assert (
            foldseek_interleaved_structure_sequence_batch["plddts"][
                0, seq_start_index:seq_end_index
            ]
            == torch.tensor(batch_plddts[i])
        ).all()
        struct_start_index = sep_locations[i] + 1


def test_foldseek_plddt_masking(profam_tokenizer, parquet_3di_processor):
    profam_tokenizer.mask_below_plddt = 90
    cfg = ProteinDatasetConfig(
        name="foldseek",
        preprocessor=parquet_3di_processor,
        data_path_pattern="foldseek_struct/0.parquet",
        is_parquet=True,
    )
    data = load_protein_dataset(
        cfg,
        tokenizer=profam_tokenizer,
        max_tokens=2048,
        data_dir=os.path.join(BASEDIR, "data/example_data"),
        shuffle=False,
    )
    datapoint = next(iter(data))
    collator = CustomDataCollator(tokenizer=profam_tokenizer, mlm=False)
    batch = collator([datapoint])

    assert (
        torch.where(batch["plddt_mask"], batch["plddts"], torch.tensor(-1e6)).max() < 90
    )
    assert (
        torch.where(batch["plddt_mask"], batch["labels"], torch.tensor(-100)) == -100
    ).all()
    assert (
        torch.where(
            batch["plddt_mask"], batch["input_ids"], profam_tokenizer.mask_token_id
        )
        == profam_tokenizer.mask_token_id
    ).all()
    assert not (
        batch["input_ids"][0][batch["aa_mask"][0]] == profam_tokenizer.mask_token_id
    ).any()
    assert not batch["plddts"].isnan().any()
