"""This file prepares config fixtures for other tests."""
import os

import hydra
import pandas as pd
import pytest
import torch
from hydra import compose, initialize, initialize_config_dir

from src.constants import BASEDIR
from src.data.collators import DocumentBatchCollator
from src.data.objects import ProteinDocument
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
                "experiment=train_profam_example",
                # lightweight test model (previously `model=llama_test`)
                "model.scheduler_name=inverse_sqrt",
                "model.lr=1e-3",
                "model.pass_res_pos_in_doc_as_position_ids=false",
                "model.config.hidden_size=64",
                "model.config.intermediate_size=128",
                "model.config.num_attention_heads=4",
                "model.config.num_hidden_layers=1",
                "model.config.num_key_value_heads=4",
                "model.config.max_position_embeddings=2048",
                "model.config.scoring_max_tokens=10240",
                "model.config.attn_implementation=null",
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
                "experiment=train_profam_example",
                # lightweight test model (previously `model=llama_test`)
                "model.scheduler_name=inverse_sqrt",
                "model.lr=1e-3",
                "model.pass_res_pos_in_doc_as_position_ids=false",
                "model.config.hidden_size=64",
                "model.config.intermediate_size=128",
                "model.config.num_attention_heads=4",
                "model.config.num_hidden_layers=1",
                "model.config.num_key_value_heads=4",
                "model.config.max_position_embeddings=2048",
                "model.config.scoring_max_tokens=10240",
                "model.config.attn_implementation=null",
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
                "experiment=train_profam_example",
                # lightweight test model (previously `model=llama_test`)
                "model.scheduler_name=inverse_sqrt",
                "model.lr=1e-3",
                "model.pass_res_pos_in_doc_as_position_ids=false",
                "model.config.hidden_size=64",
                "model.config.intermediate_size=128",
                "model.config.num_attention_heads=4",
                "model.config.num_hidden_layers=1",
                "model.config.num_key_value_heads=4",
                "model.config.max_position_embeddings=2048",
                "model.config.scoring_max_tokens=10240",
                "model.config.attn_implementation=null",
                "model.pass_res_pos_in_doc_as_position_ids=False",
            ],
        )
    return hydra.utils.instantiate(cfg.model, tokenizer=profam_tokenizer)


@pytest.fixture(scope="package")
def proteingym_batch(profam_tokenizer):
    # This test suite only needs a prompt + multiple "completion" variants to
    # validate KV-cache scoring is consistent with non-cached scoring.
    #
    # Avoid depending on external ProteinGym data (not vendored in this repo) and
    # avoid the deprecated ProteinGymDataset.load()/process() API.

    # Prompt: a single protein sequence, with NO trailing [SEP] (the completion
    # begins with [SEP] to match how ProteinGym is tokenized).
    prompt_doc = ProteinDocument(sequences=["ACDEFGHIKLMNPQRSTVWY"])
    prompt_tok = profam_tokenizer.encode(
        prompt_doc, document_token="[RAW]", add_final_sep=False
    )
    input_ids = torch.as_tensor(prompt_tok["input_ids"], dtype=torch.long).unsqueeze(0)

    # Completions: include leading and trailing [SEP] so the boundary token exists.
    completion_seqs = [
        "ACDEFGHIKLMNPQRSTVWY",  # "WT"
        "ACDEFGHIKLMNPQKSTVWY",  # one substitution
        "ACDEFGHIKLMNPQRSTVWV",  # one substitution
        "ACDEFGHIKLMNPQRSTVWYAC",  # slightly longer
    ]
    comp_tok = profam_tokenizer.encode_completions(completion_seqs)
    completion_ids = torch.as_tensor(comp_tok["input_ids"], dtype=torch.long).unsqueeze(
        0
    )

    # Collator expects a list of datapoints; we mimic the minimal fields used by tests.
    collator = DocumentBatchCollator(tokenizer=profam_tokenizer)
    datapoint = {
        "input_ids": input_ids.squeeze(0).numpy(),
        "completion_ids": completion_ids.squeeze(0).numpy(),
        "DMS_scores": torch.zeros(len(completion_seqs), dtype=torch.float32).numpy(),
        "ds_name": "gym",
        "DMS_id": "SYNTHETIC",
    }
    return collator([datapoint])
