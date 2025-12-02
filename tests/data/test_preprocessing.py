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
        feature_names=["input_ids", "attention_mask", "labels"],
    )
    data = data.filter(lambda x: x["msta_3di"] is not None)
    ds_iter = iter(data)
    return next(ds_iter)
