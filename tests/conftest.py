"""This file prepares config fixtures for other tests."""
import os

import hydra
import pandas as pd
import pytest
from hydra import compose, initialize, initialize_config_dir

from src.constants import BASEDIR
from src.data.builders import ProteinGymDataset
from src.data.collators import DocumentBatchCollator
from src.data.processors import preprocessing, transforms
from src.data.tokenizers import ProFamTokenizer


@pytest.fixture(scope="package")
def profam_tokenizer():
    tokenizer = ProFamTokenizer(
        tokenizer_file=os.path.join(BASEDIR, "data/profam_tokenizer.json"),
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
        tokenizer_file=os.path.join(BASEDIR, "data/profam_tokenizer.json"),
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
    preprocessing_cfg = preprocessing.PreprocessingConfig()
    return preprocessing.ParquetSequencePreprocessor(
        config=preprocessing_cfg,
    )


@pytest.fixture
def pfam_fasta_text():
    return pd.read_parquet(
        "data/example_data/pfam/Domain_60429258_61033370.parquet"
    ).iloc[0]["text"]


@pytest.fixture(scope="package")
def parquet_3di_processor():
    preprocessing_cfg = preprocessing.PreprocessingConfig()
    return preprocessing.ProteinDocumentPreprocessor(
        config=preprocessing_cfg,
        structure_tokens_col="msta_3di",
        transform_fns=[transforms.interleave_structure_sequence],
    )


@pytest.fixture(scope="package")
def proteingym_batch(profam_tokenizer):
    builder = ProteinGymDataset(
        name="pfam",
        dms_ids=["BLAT_ECOLX_Jacquier_2013"],
        keep_gaps=False,
        use_filtered_msa=True,
        seed=42,
        max_tokens_per_example=2048,
        num_proc=None,
    )
    data = builder.load(
        data_dir=os.path.join(BASEDIR, "data/example_data"),
    )
    data = builder.process(
        data,
        tokenizer=profam_tokenizer,
    )
    datapoint = next(iter(data))
    collator = DocumentBatchCollator(tokenizer=profam_tokenizer)
    return collator([datapoint])


@pytest.fixture
def pfam_fasta_text():
    return pd.read_parquet(
        "data/example_data/pfam/Domain_60429258_61033370.parquet"
    ).iloc[0]["text"]
