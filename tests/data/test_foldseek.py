import functools
import os

import numpy as np
import pandas as pd
import pytest
import torch

from src.constants import ALL_FEATURE_NAMES, BASEDIR
from src.data.builders import HFProteinDatasetConfig, ParquetStructureDataset
from src.data.collators import CustomDataCollator
from src.data.preprocessing import backbone_coords_from_example
from src.data.processors import backbone_coords_from_example, preprocessing, transforms
from src.data.utils import CustomDataCollator
from src.structure.pdb import get_atom_coords_residuewise, load_structure


@pytest.fixture
def foldseek_df():
    df = pd.read_parquet("data/example_data/foldseek_struct/0.parquet")
    return df


# TODO: Update pdb files and uncomment
def test_foldseek_backbone_loading(foldseek_df):
    for _, row in foldseek_df.head(3).iterrows():
        foldseek_example = row.to_dict()
        # Q. why does this successfully load the backbone coordinates as arrays?
        backbone_coords, _ = backbone_coords_from_example(foldseek_example)
        for seq, acc, recons_coords in zip(
            foldseek_example["sequences"],
            foldseek_example["accessions"],
            backbone_coords,
        ):
            pdbfile = (
                "data/example_data/foldseek_struct/0-0/AF-{}-F1-model_v4.pdb".format(
                    acc, acc
                )
            )
            structure = load_structure(pdbfile, chain="A")
            # This is loaded in full precision, whereas coords are saved in float16
            coords = get_atom_coords_residuewise(["N", "CA", "C", "O"], structure)
            assert np.allclose(coords, recons_coords.astype(np.float32), atol=2e-2)
            assert len(coords) == len(seq)


def stitch_tokens(tokenizer, struct_tokens, seq_tokens):
    assert len(struct_tokens) == len(seq_tokens)
    arrays = []
    for struct, seq in zip(struct_tokens, seq_tokens):
        arrays += [
            struct,
            np.full((1,), tokenizer.seq_struct_sep_token_id),
            seq,
        ]
        arrays.append(np.full((1,), tokenizer.convert_tokens_to_ids("[SEP]")))
    return np.concatenate(arrays, axis=0)


@pytest.fixture()
def foldseek_interleaved_structure_sequence_batch(
    profam_tokenizer,
):
    max_tokens = 2048
    preprocessing_cfg = preprocessing.PreprocessingConfig(
        keep_insertions=True,
        to_upper=True,
        keep_gaps=False,
        use_msa_pos=False,
    )
    parquet_3di_processor = preprocessing.ProteinDocumentPreprocessor(
        config=preprocessing_cfg,
        interleave_structure_sequence=True,
    )
    cfg = HFProteinDatasetConfig(
        data_path_pattern="foldseek_struct/0.parquet",
        file_type="parquet",
    )
    builder = ParquetStructureDataset(
        name="foldseek_example",
        cfg=cfg,
        preprocessor=parquet_3di_processor,
        infer_representative_from_identifier=True,
    )
    data = builder.load(data_dir=os.path.join(BASEDIR, "data/example_data"))
    data = builder.process(
        data,
        tokenizer=profam_tokenizer,
        max_tokens_per_example=max_tokens,
        shuffle_proteins_in_document=False,
        feature_names=ALL_FEATURE_NAMES,
    )
    datapoint = next(iter(data))
    collator = CustomDataCollator(tokenizer=profam_tokenizer, mlm=False)
    return collator([datapoint])


@pytest.fixture()
def foldseek_datapoint(profam_tokenizer):
    cfg = HFProteinDatasetConfig(
        data_path_pattern="foldseek_struct/0.parquet",
        file_type="parquet",
        structure_tokens_col="msta_3di",
    )
    builder = ParquetStructureDataset(
        name="foldseek_example",
        cfg=cfg,
        preprocessor=None,
        infer_representative_from_identifier=True,
    )
    data = builder.load(data_dir=os.path.join(BASEDIR, "data/example_data"))
    data = builder.process(
        data,
        tokenizer=profam_tokenizer,
        max_tokens_per_example=2048,
        shuffle_proteins_in_document=False,
        feature_names=ALL_FEATURE_NAMES,
    )
    # bc preprocessor is none we have to filter out the datapoint manually
    data = data.filter(lambda x: x["msta_3di"] is not None)
    ds_iter = iter(data)
    return next(ds_iter)


# TODO: write full manual test for coords concatenation and padding etc.
def test_foldseek_interleaved_tokenization(
    foldseek_interleaved_structure_sequence_batch,
    foldseek_datapoint,
    profam_tokenizer,
):
    num_sequences_in_batch = (
        foldseek_interleaved_structure_sequence_batch["input_ids"]
        == profam_tokenizer.sep_token_id
    ).sum()

    batch_seqs = foldseek_datapoint["sequences"][:num_sequences_in_batch]
    batch_coords, _ = backbone_coords_from_example(foldseek_datapoint)[
        :num_sequences_in_batch
    ]
    batch_plddts = foldseek_datapoint["plddts"][:num_sequences_in_batch]
    batch_3dis = [
        s.replace("-", "").lower()
        for s in foldseek_datapoint["msta_3di"][:num_sequences_in_batch]
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
    stitched_tokens = np.array(
        profam_tokenizer.convert_tokens_to_ids(["[RAW]", profam_tokenizer.bos_token])
    )
    stitched_tokens = np.concatenate(
        [
            stitched_tokens,
            stitch_tokens(
                profam_tokenizer, individual_3d_tokens, individual_seq_tokens
            ),
        ],
        axis=0,
    )

    assert (
        foldseek_interleaved_structure_sequence_batch["input_ids"][
            0, : stitched_tokens.shape[0]
        ]
        == torch.from_numpy(stitched_tokens)
    ).all()

    sep_locations = torch.argwhere(
        foldseek_interleaved_structure_sequence_batch["input_ids"][0]
        == profam_tokenizer.sep_token_id
    ).flatten()
    struct_sep_locations = torch.argwhere(
        foldseek_interleaved_structure_sequence_batch["input_ids"][0]
        == profam_tokenizer.seq_struct_sep_token_id
    ).flatten()
    assert (
        sep_locations.shape[0]
        == struct_sep_locations.shape[0]
        == num_sequences_in_batch
    )
    # TODO: assertion on seq pos at seps
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
            == torch.zeros_like(torch.tensor(batch_coords[i]))
        ).all()
        assert (
            foldseek_interleaved_structure_sequence_batch["plddts"][
                0, seq_start_index:seq_end_index
            ]
            == torch.tensor(batch_plddts[i])
        ).all()
        struct_start_index = sep_locations[i] + 1


def test_foldseek_plddt_masking(profam_tokenizer):
    preprocessing_cfg = preprocessing.PreprocessingConfig(
        keep_insertions=True,
        to_upper=True,
        keep_gaps=False,
        use_msa_pos=False,
    )
    plddt_cutoff = 80.0
    preprocessor = preprocessing.ProteinDocumentPreprocessor(
        config=preprocessing_cfg,
        transform_fns=[
            functools.partial(
                transforms.apply_plddt_mask, threshold=plddt_cutoff, mask_plddts=True
            ),
            transforms.interleave_structure_sequence,
        ],
    )
    cfg = HFProteinDatasetConfig(
        data_path_pattern="foldseek_struct/0.parquet",
        file_type="parquet",
        structure_tokens_col="msta_3di",
    )
    builder = ParquetStructureDataset(
        name="foldseek_example",
        cfg=cfg,
        preprocessor=preprocessor,
    )
    data = builder.load(data_dir=os.path.join(BASEDIR, "data/example_data"))
    data = builder.process(
        data,
        tokenizer=profam_tokenizer,
        max_tokens_per_example=2048,
        shuffle_proteins_in_document=False,
        feature_names=ALL_FEATURE_NAMES,
    )

    datapoint = next(iter(data))
    collator = CustomDataCollator(tokenizer=profam_tokenizer, mlm=False)
    batch = collator([datapoint])

    plddt_mask = (batch["plddts"] == 0.0) & batch["structure_mask"]

    assert (
        torch.where(plddt_mask, batch["plddts"], torch.tensor(-1e6)).max()
        < plddt_cutoff
    )
    assert (
        torch.where(
            ~plddt_mask & batch["structure_mask"], batch["plddts"], torch.tensor(100)
        ).min()
        >= plddt_cutoff
    )

    # N.B. this only applies to structure tokens due to intersection with structure mask
    assert (
        torch.where(plddt_mask, batch["input_ids"], profam_tokenizer.mask_token_id)
        == profam_tokenizer.mask_token_id
    ).all()
    assert not (
        batch["input_ids"][0][batch["aa_mask"][0]] == profam_tokenizer.mask_token_id
    ).any()
    assert not batch["plddts"].isnan().any()


def test_foldseek_representative_concatenation(profam_tokenizer):
    max_tokens = 2048
    # verify that building representatives into a single document is successful
    preprocessing_cfg = preprocessing.PreprocessingConfig(
        keep_insertions=True,
        to_upper=True,
        keep_gaps=False,
        use_msa_pos=False,
    )
    parquet_3di_processor = preprocessing.ProteinDocumentPreprocessor(
        config=preprocessing_cfg,
        interleave_structure_sequence=False,  # n.b. interleaving transform automatically computes max_tokens
    )
    cfg = HFProteinDatasetConfig(
        data_path_pattern="foldseek_representatives/0.parquet",
        file_type="parquet",
        shuffle=False,
        structure_tokens_col=None,
    )
    builder = ParquetStructureDataset(
        name="foldseek_example",
        cfg=cfg,
        preprocessor=parquet_3di_processor,
    )
    data = builder.load(data_dir=os.path.join(BASEDIR, "data/example_data"))
    data = builder.process(
        data,
        tokenizer=profam_tokenizer,
        max_tokens_per_example=2048,
        shuffle_proteins_in_document=False,
        feature_names=["input_ids", "attention_mask", "labels", "plddts", "coords"],
    )
    example = next(iter(data))
    assert (example["input_ids"] == profam_tokenizer.sep_token_id).sum() > 1
