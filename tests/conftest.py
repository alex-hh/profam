"""This file prepares config fixtures for other tests."""

import os
from pathlib import Path

import hydra
import pandas as pd
import pytest
import rootutils
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, open_dict

from src.constants import BASEDIR
from src.data.proteingym import load_gym_dataset
from src.data.utils import (
    CustomDataCollator,
    ProteinDatasetConfig,
    load_protein_dataset,
)
from src.utils.tokenizers import ProFamTokenizer


@pytest.fixture()
def profam_tokenizer_seqpos():
    tokenizer = ProFamTokenizer(
        tokenizer_file=os.path.join(
            BASEDIR, "src/data/components/profam_tokenizer.json"
        ),
        unk_token="[UNK]",
        pad_token="[PAD]",
        bos_token="[start-of-document]",
        sep_token="[SEP]",
        mask_token="[MASK]",
        add_special_tokens=True,
        use_seq_pos=True,
        max_seq_pos=2048,
        max_tokens=2048,
    )
    return tokenizer


@pytest.fixture()
def profam_tokenizer_noseqpos():
    tokenizer = ProFamTokenizer(
        tokenizer_file=os.path.join(
            BASEDIR, "src/data/components/profam_tokenizer.json"
        ),
        unk_token="[UNK]",
        pad_token="[PAD]",
        bos_token="[start-of-document]",
        sep_token="[SEP]",
        mask_token="[MASK]",
        add_special_tokens=True,
        use_seq_pos=False,
        max_seq_pos=2048,
        max_tokens=2048,
    )
    return tokenizer


@pytest.fixture()
def default_model_noseqpos(profam_tokenizer_noseqpos):
    # otherwise could do this via overrides...
    with initialize(config_path="../configs", version_base="1.3"):
        cfg = compose(
            config_name="train.yaml",
            return_hydra_config=True,
        )
    return hydra.utils.instantiate(cfg.model, tokenizer=profam_tokenizer_noseqpos)


@pytest.fixture()
def default_model_seqpos(profam_tokenizer_seqpos):
    with initialize(config_path="../configs", version_base="1.3"):
        cfg = compose(
            config_name="train.yaml",
            return_hydra_config=True,
        )
    return hydra.utils.instantiate(cfg.model, tokenizer=profam_tokenizer_seqpos)


@pytest.fixture()
def proteingym_batch(profam_tokenizer_seqpos):
    data = load_gym_dataset(
        dms_ids=["BLAT_ECOLX_Jacquier_2013"],
        tokenizer=profam_tokenizer_seqpos,
        gym_data_dir="data/example_data/proteingym",
        max_tokens=profam_tokenizer_seqpos.max_tokens,
        keep_gaps=False,
        num_proc=None,
    )
    datapoint = next(iter(data))
    collator = CustomDataCollator(tokenizer=profam_tokenizer_seqpos, mlm=False)
    return collator([datapoint])


@pytest.fixture()
def pfam_batch(profam_tokenizer):
    cfg = ProteinDatasetConfig(
        name="pfam",
        keep_gaps=False,
        data_path_pattern="pfam/Domain_60429258_61033370.parquet",
        keep_insertions=True,
        to_upper=True,
        is_parquet=True,
    )
    data = load_protein_dataset(
        cfg,
        tokenizer=profam_tokenizer,
        max_tokens=2048,
        data_dir=os.path.join(BASEDIR, "data/example_data"),
        use_seq_pos=True,
        max_seq_pos=2048,
        shuffle=False,
    )
    datapoint = next(iter(data))
    collator = CustomDataCollator(tokenizer=profam_tokenizer, mlm=False)
    return collator([datapoint])


@pytest.fixture()
def foldseek_batch(profam_tokenizer):
    cfg = ProteinDatasetConfig(
        name="foldseek",
        keep_gaps=False,
        data_path_pattern="foldseek_struct/3.parquet",
        keep_insertions=True,
        to_upper=True,
        is_parquet=True,
    )
    data = load_protein_dataset(
        cfg,
        tokenizer=profam_tokenizer,
        max_tokens=2048,
        data_dir=os.path.join(BASEDIR, "data/example_data"),
        use_seq_pos=True,
        max_seq_pos=2048,
        shuffle=False,
    )
    datapoint = next(iter(data))
    collator = CustomDataCollator(tokenizer=profam_tokenizer, mlm=False)
    return collator([datapoint])


@pytest.fixture
def pfam_fasta_text():
    return pd.read_parquet(
        "data/example_data/pfam/Domain_60429258_61033370.parquet"
    ).iloc[0]["text"]


@pytest.fixture(scope="package")
def cfg_train_global() -> DictConfig:
    """A pytest fixture for setting up a default Hydra DictConfig for training.

    :return: A DictConfig object containing a default Hydra configuration for training.
    """
    with initialize(version_base="1.3", config_path="../configs"):
        cfg = compose(config_name="train.yaml", return_hydra_config=True, overrides=[])

        # set defaults for all tests
        with open_dict(cfg):
            cfg.paths.root_dir = str(rootutils.find_root(indicator=".project-root"))
            cfg.trainer.max_epochs = 1
            cfg.trainer.limit_train_batches = 0.01
            cfg.trainer.limit_val_batches = 0.1
            cfg.trainer.limit_test_batches = 0.1
            cfg.trainer.accelerator = "cpu"
            cfg.trainer.devices = 1
            cfg.data.num_workers = 0
            cfg.data.pin_memory = False
            cfg.extras.print_config = False
            cfg.extras.enforce_tags = False
            cfg.logger = None

    return cfg


@pytest.fixture(scope="package")
def cfg_eval_global() -> DictConfig:
    """A pytest fixture for setting up a default Hydra DictConfig for evaluation.

    :return: A DictConfig containing a default Hydra configuration for evaluation.
    """
    with initialize(version_base="1.3", config_path="../configs"):
        cfg = compose(
            config_name="eval.yaml", return_hydra_config=True, overrides=["ckpt_path=."]
        )

        # set defaults for all tests
        with open_dict(cfg):
            cfg.paths.root_dir = str(rootutils.find_root(indicator=".project-root"))
            cfg.trainer.max_epochs = 1
            cfg.trainer.limit_test_batches = 0.1
            cfg.trainer.accelerator = "cpu"
            cfg.trainer.devices = 1
            cfg.data.num_workers = 0
            cfg.data.pin_memory = False
            cfg.extras.print_config = False
            cfg.extras.enforce_tags = False
            cfg.logger = None

    return cfg


@pytest.fixture(scope="function")
def cfg_train(cfg_train_global: DictConfig, tmp_path: Path) -> DictConfig:
    """A pytest fixture built on top of the `cfg_train_global()` fixture, which accepts a temporary
    logging path `tmp_path` for generating a temporary logging path.

    This is called by each test which uses the `cfg_train` arg. Each test generates its own temporary logging path.

    :param cfg_train_global: The input DictConfig object to be modified.
    :param tmp_path: The temporary logging path.

    :return: A DictConfig with updated output and log directories corresponding to `tmp_path`.
    """
    cfg = cfg_train_global.copy()

    with open_dict(cfg):
        cfg.paths.output_dir = str(tmp_path)
        cfg.paths.log_dir = str(tmp_path)

    yield cfg

    GlobalHydra.instance().clear()


@pytest.fixture(scope="function")
def cfg_eval(cfg_eval_global: DictConfig, tmp_path: Path) -> DictConfig:
    """A pytest fixture built on top of the `cfg_eval_global()` fixture, which accepts a temporary
    logging path `tmp_path` for generating a temporary logging path.

    This is called by each test which uses the `cfg_eval` arg. Each test generates its own temporary logging path.

    :param cfg_train_global: The input DictConfig object to be modified.
    :param tmp_path: The temporary logging path.

    :return: A DictConfig with updated output and log directories corresponding to `tmp_path`.
    """
    cfg = cfg_eval_global.copy()

    with open_dict(cfg):
        cfg.paths.output_dir = str(tmp_path)
        cfg.paths.log_dir = str(tmp_path)

    yield cfg

    GlobalHydra.instance().clear()
