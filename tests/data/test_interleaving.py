import functools
import os

import pytest
import torch

from src.constants import BASEDIR
from src.data import preprocessing, transforms
from src.data.datasets import ProteinDatasetConfig, load_protein_dataset
from src.data.utils import CustomDataCollator


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
    parquet_3di_processor = preprocessing.ParquetStructurePreprocessor(
        config=preprocessing_cfg,
        structure_tokens_col="msta_3di",
        interleave_proteins=True,
        interleaved_transform_fns=[
            functools.partial(
                transforms.mask_interleave_structure_sequence, use_structure_tokens=True
            )
        ],
    )
    cfg = ProteinDatasetConfig(
        name="foldseek",
        preprocessor=parquet_3di_processor,
        data_path_pattern="foldseek_struct/0.parquet",
        is_parquet=True,
    )
    data = load_protein_dataset(
        cfg,
        tokenizer=profam_tokenizer,
        max_tokens=max_tokens,
        data_dir=os.path.join(BASEDIR, "data/example_data"),
        shuffle=False,
        feature_names=["input_ids", "attention_mask", "labels", "plddts", "coords"],
    )
    datapoint = next(iter(data))
    collator = CustomDataCollator(tokenizer=profam_tokenizer, mlm=False)
    return collator([datapoint])


@pytest.fixture()
def foldseek_datapoint(profam_tokenizer):
    cfg = ProteinDatasetConfig(
        name="foldseek",
        data_path_pattern="foldseek_struct/0.parquet",
        is_parquet=True,
    )
    data = load_protein_dataset(
        cfg,
        tokenizer=profam_tokenizer,
        max_tokens=2048,
        data_dir=os.path.join(BASEDIR, "data/example_data"),
        shuffle=False,
        feature_names=["input_ids", "attention_mask", "labels", "plddts", "coords"],
    )
    return next(iter(data))


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
    batch_coords = preprocessing.backbone_coords_from_example(foldseek_datapoint)[
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
        assert torch.allclose(
            foldseek_interleaved_structure_sequence_batch["backbone_coords"][
                0, struct_start_index:struct_end_index
            ],
            torch.from_numpy(batch_coords[i]).float(),
        )
        assert torch.allclose(
            foldseek_interleaved_structure_sequence_batch["plddts"][
                0, seq_start_index:seq_end_index
            ],
            torch.tensor(batch_plddts[i]).float(),
        )
        assert torch.isnan(
            foldseek_interleaved_structure_sequence_batch["backbone_coords"][
                0, seq_start_index:seq_end_index
            ]
        ).all()
        assert torch.allclose(
            foldseek_interleaved_structure_sequence_batch["plddts"][
                0, seq_start_index:seq_end_index
            ],
            torch.tensor(batch_plddts[i]).float(),
        )
        struct_start_index = sep_locations[i] + 1
