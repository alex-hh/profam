import functools
import os

import numpy as np
import pandas as pd
import pytest
import torch

from src.constants import BASEDIR
from src.data import preprocessing, transforms
from src.data.datasets import ProteinDatasetConfig, load_protein_dataset
from src.data.pdb import get_atom_coords_residuewise, load_structure
from src.data.utils import CustomDataCollator


@pytest.fixture
def foldseek_df():
    df = pd.read_parquet("data/example_data/foldseek_struct/0.parquet")
    return df


def test_foldseek_backbone_loading(foldseek_df):
    for _, row in foldseek_df.iterrows():
        foldseek_example = row.to_dict()
        # Q. why does this successfully load the backbone coordinates as arrays?
        backbone_coords = preprocessing.backbone_coords_from_example(foldseek_example)
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
            torch.full((1,), tokenizer.seq_struct_sep_token_id),
            seq,
        ]
        tensors.append(torch.full((1,), tokenizer.convert_tokens_to_ids("[SEP]")))
    return torch.cat(tensors, dim=0)


# def test_foldseek_plddt_masking(profam_tokenizer, parquet_3di_processor):
#     profam_tokenizer.mask_below_plddt = 90
#     cfg = ProteinDatasetConfig(
#         name="foldseek",
#         preprocessor=parquet_3di_processor,
#         data_path_pattern="foldseek_struct/0.parquet",
#         is_parquet=True,
#     )
#     data = load_protein_dataset(
#         cfg,
#         tokenizer=profam_tokenizer,
#         max_tokens=2048,
#         data_dir=os.path.join(BASEDIR, "data/example_data"),
#         shuffle=False,
#         feature_names=["input_ids", "attention_mask", "labels", "plddts", "coords"],
#     )
#     datapoint = next(iter(data))
#     collator = CustomDataCollator(tokenizer=profam_tokenizer, mlm=False)
#     batch = collator([datapoint])

#     assert (
#         torch.where(batch["plddt_mask"], batch["plddts"], torch.tensor(-1e6)).max() < 90
#     )
#     assert (
#         torch.where(batch["plddt_mask"], batch["labels"], torch.tensor(-100)) == -100
#     ).all()
#     assert (
#         torch.where(
#             batch["plddt_mask"], batch["input_ids"], profam_tokenizer.mask_token_id
#         )
#         == profam_tokenizer.mask_token_id
#     ).all()
#     assert not (
#         batch["input_ids"][0][batch["aa_mask"][0]] == profam_tokenizer.mask_token_id
#     ).any()
#     assert not batch["plddts"].isnan().any()
#     profam_tokenizer.mask_below_plddt = None


def test_foldseek_representative_concatenation(profam_tokenizer):
    # verify that building representatives into a single document is successful
    max_tokens = 2048
    preprocessing_cfg = preprocessing.PreprocessingConfig(
        keep_insertions=True,
        to_upper=True,
        keep_gaps=False,
        use_msa_pos=False,
        batched_map=True,
        map_batch_size=30,
    )
    parquet_3di_processor = preprocessing.ParquetStructurePreprocessor(
        config=preprocessing_cfg,
        structure_tokens_col=None,
        interleave_proteins=False,  # n.b. interleaving transform automatically computes max_tokens
    )
    cfg = ProteinDatasetConfig(
        name="foldseek",
        preprocessor=parquet_3di_processor,
        data_path_pattern="foldseek_representatives/0.parquet",
        is_parquet=True,
        shuffle=False,
    )
    dataset = load_protein_dataset(
        cfg,
        profam_tokenizer,
        data_dir=os.path.join(BASEDIR, "data/example_data"),
        max_tokens=max_tokens,
        shuffle=False,
    )
    example = next(iter(dataset))
    assert (example["input_ids"] == profam_tokenizer.sep_token_id).sum() > 1
