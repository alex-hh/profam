import re

from hydra import compose, initialize_config_dir

import profam
from profam.cli import generate_sequences, score_sequences
from profam.constants import CONFIGS_DIR, TOKENIZER_FILE


def test_package_version_is_exposed():
    assert re.match(r"^\d+\.\d+\.\d+", profam.__version__)


def test_packaged_runtime_assets_exist():
    assert CONFIGS_DIR.is_dir()
    assert (CONFIGS_DIR / "train.yaml").is_file()
    assert TOKENIZER_FILE.is_file()


def test_train_config_loads_from_packaged_config_dir():
    with initialize_config_dir(str(CONFIGS_DIR), version_base="1.3"):
        cfg = compose(
            config_name="train.yaml",
            return_hydra_config=True,
            overrides=["experiment=train_profam_example"],
        )

    assert cfg.tokenizer.tokenizer_file == str(TOKENIZER_FILE)
    assert "root_dir" in cfg.paths


def test_cli_modules_are_exposed():
    assert callable(generate_sequences.main)
    assert callable(score_sequences.main)
