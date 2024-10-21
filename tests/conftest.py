"""This file prepares config fixtures for other tests."""
import os
from pathlib import Path

import hydra
import pandas as pd
import pytest
import rootutils
from hydra import compose, initialize, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, open_dict

from src.constants import BASEDIR
from src.data import preprocessing, transforms
from src.data.datasets import ProteinDatasetConfig, load_protein_dataset
from src.data.proteingym import load_gym_dataset
from src.data.utils import DocumentBatchCollator
from src.utils.tokenizers import ProFamTokenizer


@pytest.fixture(scope="package")
def profam_tokenizer():
    tokenizer = ProFamTokenizer(
        tokenizer_file=os.path.join(
            BASEDIR, "src/data/components/profam_tokenizer.json"
        ),
        unk_token="[UNK]",
        pad_token="[PAD]",
        bos_token="[start-of-document]",
        sep_token="[SEP]",
        mask_token="?",
        seq_struct_sep_token="|",
        embed_residue_index=True,
        max_res_pos_in_seq=2048,
        max_tokens=2048,
        mask_below_plddt=None,
    )
    return tokenizer


@pytest.fixture(scope="package")
def profam_tokenizer_noseqpos():
    tokenizer = ProFamTokenizer(
        tokenizer_file=os.path.join(
            BASEDIR, "src/data/components/profam_tokenizer.json"
        ),
        unk_token="[UNK]",
        pad_token="[PAD]",
        bos_token="[start-of-document]",
        sep_token="[SEP]",
        mask_token="?",
        seq_struct_sep_token="|",
        embed_residue_index=False,
        max_res_pos_in_seq=2048,
        max_tokens=2048,
        mask_below_plddt=None,
    )
    return tokenizer


@pytest.fixture(scope="package")
def test_model_noseqpos(profam_tokenizer_noseqpos):
    # otherwise could do this via overrides...
    with initialize(config_path="../configs", version_base="1.3"):
        cfg = compose(
            config_name="train.yaml",
            return_hydra_config=True,
            overrides=[
                "model=llama_test",
                "model.embed_sequence_index=False",
            ],
        )
    return hydra.utils.instantiate(cfg.model, tokenizer=profam_tokenizer_noseqpos)


@pytest.fixture(scope="package")
def test_model(profam_tokenizer):
    with initialize(config_path="../configs", version_base="1.3"):
        cfg = compose(
            config_name="train.yaml",
            return_hydra_config=True,
            overrides=[
                "model=llama_test",
                "model.embed_sequence_index=True",
            ],
        )
    return hydra.utils.instantiate(cfg.model, tokenizer=profam_tokenizer)


@pytest.fixture(scope="package")
def model_seq_index(profam_tokenizer):
    with initialize_config_dir(os.path.join(BASEDIR, "configs"), version_base="1.3"):
        cfg = compose(
            config_name="train.yaml",
            return_hydra_config=True,
            overrides=[
                "model.embed_sequence_index=True",
                "model.config.attn_implementation=null",
            ],
        )
    return hydra.utils.instantiate(cfg.model, tokenizer=profam_tokenizer)


@pytest.fixture(scope="package")
def parquet_raw_sequence_processor():
    preprocessing_cfg = preprocessing.PreprocessingConfig(
        keep_insertions=True,
        to_upper=True,
        keep_gaps=False,
        use_msa_pos=False,
    )
    return preprocessing.ParquetSequencePreprocessor(
        config=preprocessing_cfg,
    )


@pytest.fixture(scope="package")
def parquet_3di_processor():
    preprocessing_cfg = preprocessing.PreprocessingConfig(
        keep_insertions=True,
        to_upper=True,
        keep_gaps=False,
        use_msa_pos=False,
    )
    return preprocessing.ParquetStructurePreprocessor(
        config=preprocessing_cfg,
        structure_tokens_col="msta_3di",
        transform_fns=[transforms.interleave_structure_sequence],
    )


@pytest.fixture(scope="package")
def proteingym_batch(profam_tokenizer):
    # TODO: use filtered msa - processing the full msa very slow (why?)
    data = load_gym_dataset(
        dms_ids=["BLAT_ECOLX_Jacquier_2013"],
        tokenizer=profam_tokenizer,
        gym_data_dir="data/example_data/ProteinGym",
        max_tokens=2048,
        keep_gaps=False,
        num_proc=None,
        use_filtered_msa=True,
    )
    datapoint = next(iter(data))
    collator = DocumentBatchCollator(tokenizer=profam_tokenizer)
    return collator([datapoint])


@pytest.fixture()
def pfam_batch(profam_tokenizer):
    cfg = ProteinDatasetConfig(
        keep_gaps=False,
        data_path_pattern="pfam/Domain_60429258_61033370.parquet",
        keep_insertions=True,  # TODO unexpected argument
        to_upper=True,  # TODO unexpected argument
        is_parquet=True,
    )
    data = load_protein_dataset(
        cfg,
        tokenizer=profam_tokenizer,
        dataset_name="pfam",
        max_tokens_per_example=2048,
        data_dir=os.path.join(BASEDIR, "data/example_data"),
        shuffle=False,
    )
    datapoint = next(iter(data))
    collator = DocumentBatchCollator(tokenizer=profam_tokenizer)
    return collator([datapoint])


@pytest.fixture()
def foldseek_batch(profam_tokenizer):
    cfg = ProteinDatasetConfig(
        keep_gaps=False,
        data_path_pattern="foldseek_struct/3.parquet",
        keep_insertions=True,  # TODO unexpected argument
        to_upper=True,  # TODO unexpected argument
        is_parquet=True,
    )
    data = load_protein_dataset(
        cfg,
        tokenizer=profam_tokenizer,
        dataset_name="foldseek",
        max_tokens_per_example=2048,
        data_dir=os.path.join(BASEDIR, "data/example_data"),
        shuffle=False,
    )
    datapoint = next(iter(data))
    collator = DocumentBatchCollator(tokenizer=profam_tokenizer)
    return collator([datapoint])


@pytest.fixture
def pfam_fasta_text():
    return pd.read_parquet(
        "data/example_data/pfam/Domain_60429258_61033370.parquet"
    ).iloc[0]["text"]
