import functools
import math
import os

import pytest

from src.constants import BASEDIR
from src.data.builders import ProteinDatasetConfig, StructureDocumentIterableDataset
from src.data.tokenizers import examples_list_to_dict


@pytest.fixture()
def foldseek_datapoint(profam_tokenizer):
    cfg = ProteinDatasetConfig(
        data_path_pattern="foldseek_struct/0.parquet",
        file_type="parquet",
    )
    builder = StructureDocumentIterableDataset(
        name="foldseek_example",
        cfg=cfg,
        preprocessor=None,
    )
    data = builder.load(data_dir=os.path.join(BASEDIR, "data/example_data"))
    data = builder.process(
        data,
        tokenizer=profam_tokenizer,
        max_tokens_per_example=2048,
        shuffle_proteins_in_document=False,
        feature_names=["input_ids", "attention_mask", "labels", "plddts", "coords"],
    )
    data = data.filter(lambda x: x["msta_3di"] is not None)
    ds_iter = iter(data)
    return next(ds_iter)


# TODO: add tests for standard preprocessing.


# def test_build_combined_documents(foldseek_datapoint, profam_tokenizer):
#     examples = [foldseek_datapoint, foldseek_datapoint]
#     examples = examples_list_to_dict(examples)

#     document_builder = functools.partial(
#         StructureDocumentIterableDataset.build_document,
#         structure_tokens_col="msta_3di",
#     )
#     proteins_list = [document_builder._build_document(example) for example in examples]
#     assert len(proteins_list) == 1  # we expect documents to be combined
#     assert proteins_list[0].sequences == 2 * foldseek_datapoint["sequences"]


# for inverse folding, we want a single document with all sequences concatenated
# def test_concat_representatives_into_single_document(profam_tokenizer):
#     cfg = ProteinDatasetConfig(
#         data_path_pattern="foldseek_representatives/0.parquet",
#         file_type="parquet",
#     )
#     builder = StructureDocumentIterableDataset(
#         name="foldseek_example",
#         cfg=cfg,
#         preprocessor=None,
#     )
#     data = builder.load(data_dir=os.path.join(BASEDIR, "data/example_data"))
#     data = builder.process(
#         data,
#         tokenizer=profam_tokenizer,
#         max_tokens_per_example=None,
#         shuffle_proteins_in_document=False,
#         feature_names=["input_ids", "attention_mask", "labels", "plddts", "coords"],
#     )

#     example = next(iter(data))
#     assert len(example["sequences"]) == 1
#     protein_len = (
#         len(example["sequences"][0]) + profam_tokenizer.num_start_tokens + 1
#     )  # +1 for the end token
#     examples = [example] * 20
#     examples = examples_list_to_dict(examples)

#     proteins_list = build_documents_helper(
#         examples,
#         StructureDocumentIterableDataset.build_document,
#         max_tokens=protein_len * 4,
#         shuffle=False,
#     )
#     expected_num_documents = math.ceil(20 / 4)
#     # we expect documents to be combined up to max_tokens
#     assert (
#         len(proteins_list) == expected_num_documents
#     ), f"Expected {expected_num_documents} documents go {len(proteins_list)}"
