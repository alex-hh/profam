import math
import os

import pytest

from src.constants import BASEDIR
from src.data import preprocessing
from src.data.custom_datasets import ProteinDatasetConfig, load_protein_dataset
from src.data.utils import examples_list_to_dict


@pytest.fixture()
def foldseek_datapoint(profam_tokenizer):
    cfg = ProteinDatasetConfig(
        data_path_pattern="foldseek_struct/0.parquet",
        is_parquet=True,
    )
    data = load_protein_dataset(
        cfg,
        tokenizer=profam_tokenizer,
        dataset_name="foldseek",
        max_tokens_per_example=2048,
        data_dir=os.path.join(BASEDIR, "data/example_data"),
        shuffle=False,
        feature_names=["input_ids", "attention_mask", "labels", "plddts", "coords"],
    )
    data = data.filter(lambda x: x["msta_3di"] is not None)
    ds_iter = iter(data)
    return next(ds_iter)


# TODO: add tests for standard preprocessing.


def test_build_combined_documents(foldseek_datapoint, profam_tokenizer):
    examples = [foldseek_datapoint, foldseek_datapoint]
    examples = examples_list_to_dict(examples)

    config = preprocessing.PreprocessingConfig(
        keep_gaps=False,
        to_upper=False,
        keep_insertions=False,
        use_msa_pos=False,
    )
    preprocessor = preprocessing.ParquetStructurePreprocessor(
        config=config,
        structure_tokens_col="msta_3di",
    )
    proteins_list = preprocessor.build_documents(
        examples, profam_tokenizer, max_tokens=None, shuffle=False
    )
    assert len(proteins_list) == 1  # we expect documents to be combined
    assert proteins_list[0].sequences == 2 * foldseek_datapoint["sequences"]


# for inverse folding, we want a single document with all sequences concatenated
def test_concat_representatives_into_single_document(profam_tokenizer):
    cfg = ProteinDatasetConfig(
        data_path_pattern="foldseek_representatives/0.parquet",
        is_parquet=True,
    )
    data = load_protein_dataset(
        cfg,
        tokenizer=profam_tokenizer,
        dataset_name="foldseek",
        max_tokens_per_example=None,
        data_dir=os.path.join(BASEDIR, "data/example_data"),
        shuffle=False,
        feature_names=["input_ids", "attention_mask", "labels", "plddts", "coords"],
    )
    example = next(iter(data))
    assert len(example["sequences"]) == 1
    protein_len = (
        len(example["sequences"][0]) + profam_tokenizer.num_start_tokens + 1
    )  # +1 for the end token
    examples = [example] * 20
    examples = examples_list_to_dict(examples)

    config = preprocessing.PreprocessingConfig(
        keep_gaps=False,
        to_upper=False,
        keep_insertions=False,
        use_msa_pos=False,
    )
    preprocessor = preprocessing.ParquetStructurePreprocessor(
        config=config,
        structure_tokens_col=None,
    )
    proteins_list = preprocessor.build_documents(
        examples,
        profam_tokenizer,
        max_tokens=protein_len * 4,
        shuffle=False,
    )
    print(len(proteins_list))
    expected_num_documents = math.ceil(20 / 4)
    # we expect documents to be combined up to max_tokens
    assert (
        len(proteins_list) == expected_num_documents
    ), f"Expected {expected_num_documents} documents go {len(proteins_list)}"
