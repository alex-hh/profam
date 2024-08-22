import glob
import os
import random
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from datasets import Dataset, load_dataset
from omegaconf.listconfig import ListConfig
from transformers import DataCollatorForLanguageModeling

from src.data.preprocessing import preprocess_protein_data
from src.utils.tokenizers import ProFamTokenizer


# TODO: in future we might actually want standalone dataset class for
# more flexible customisation (e.g. mapping uniprot ids via db)
@dataclass
class ProteinDatasetConfig:
    name: str
    keep_gaps: bool = False
    data_path_pattern: Optional[str] = None
    holdout_data_files: Optional[str] = None
    holdout_identifiers: Optional[List[str]] = None
    identifier_col: Optional[str] = None
    data_path_file: Optional[str] = None
    keep_insertions: bool = False
    to_upper: bool = False
    file_repeats: int = 1
    is_parquet: bool = False
    minimum_sequences: Optional[int] = None
    document_tag: str = "[RAW]"
    truncate_after_n_sequences: Optional[int] = None
    use_msa_pos: bool = True  # for msa sequences, if true, position index will be relative to alignment cols
    interleave_structure_tokens: bool = False
    # global arguments that will get overridden in load_protein_dataset
    max_tokens: Optional[int] = 5000
    shuffle: bool = (True,)
    include_doc_hashes: bool = False

    def set_global_args(
        self,
        max_tokens: int,
        shuffle: bool,
        include_doc_hashes: bool,
    ):
        self.max_tokens = max_tokens
        self.shuffle = shuffle
        self.include_doc_hashes = include_doc_hashes


class StringObject:
    """
    Custom class to allow for
    non-tensor elements in batch
    """

    text: List[str]

    def to(self, device):
        return self


class CustomDataCollator:
    """
    Wraps DataCollatorForLanguageModeling
    allows us to include elements which are not
    tensors with seq_len dimension, eg. dataset names
    """

    def __init__(self, tokenizer, mlm=False, ignore_gaps: bool = False):
        self.tokenizer = tokenizer
        self.base_collator = DataCollatorForLanguageModeling(tokenizer, mlm=mlm)
        self.ignore_gaps = ignore_gaps

    def __call__(self, examples):
        non_string_data = [
            {k: v for k, v in e.items() if not isinstance(v, str)} for e in examples
        ]
        string_data = [
            {k: v for k, v in e.items() if isinstance(v, str)} for e in examples
        ]
        string_data_keys = set(k for obs in string_data for k in obs.keys())
        batch = self.base_collator(non_string_data)
        if self.ignore_gaps:
            batch["labels"] = batch["labels"][
                batch["labels"] == self.tokenizer.convert_tokens_to_ids("-")
            ] = -100
        for str_key in string_data_keys:
            str_vals = [obs.get(str_key, "") for obs in string_data]
            str_obj = StringObject()
            str_obj.text = str_vals
            batch[str_key] = str_obj
        return batch


def subsample_fasta_lines(lines, n_lines, shuffle=True):
    start_ix = np.array([i for i, l in enumerate(lines) if l[0] == ">"])
    end_ix = start_ix[1:]
    end_ix = np.append(end_ix, len(lines))
    lines_per_seq = len(lines) // len(start_ix)
    n_samples = min(n_lines // lines_per_seq, len(start_ix))
    if shuffle:
        sample_indices = np.random.choice(len(start_ix), n_samples, replace=False)
    else:
        sample_indices = np.arange(n_samples)
    starts = start_ix[sample_indices]
    ends = end_ix[sample_indices]
    sampled_lines = []
    for start, end in zip(starts, ends):
        assert lines[end - 1][0] != ">"
        sampled_lines.extend(lines[start:end])
    return sampled_lines


def load_protein_dataset(
    cfg: ProteinDatasetConfig,
    tokenizer: ProFamTokenizer,
    max_tokens: Optional[int] = 5000,
    data_dir="../data",
    split="train",
    include_doc_hashes: bool = False,
    shuffle: bool = True,
) -> Dataset:
    if cfg.data_path_pattern is not None:
        # replace hf path resolution with manual glob, to allow repetition
        # https://github.com/huggingface/datasets/blob/98fdc9e78e6d057ca66e58a37f49d6618aab8130/src/datasets/data_files.py#L323
        data_files = glob.glob(os.path.join(data_dir, cfg.data_path_pattern))
    else:
        assert cfg.data_path_file is not None
        with open(os.path.join(data_dir, cfg.data_path_file), "r") as f:
            data_files = [
                os.path.join(data_dir, data_file) for data_file in f.read().splitlines()
            ]

    cfg.set_global_args(
        max_tokens=max_tokens,
        shuffle=shuffle,
        include_doc_hashes=include_doc_hashes,
    )

    if cfg.holdout_data_files is not None:
        assert isinstance(cfg.holdout_data_files, list) or isinstance(
            cfg.holdout_data_files, ListConfig
        ), f"holdout files is {type(cfg.holdout_data_files)} not list"
        all_files = len(data_files)
        data_files = [f for f in data_files if f not in cfg.holdout_data_files]
        print("Excluding", all_files - len(data_files), "holdout files")

    assert isinstance(data_files, list)
    data_files = data_files * cfg.file_repeats
    random.shuffle(data_files)  # TODO: seed explicitly?
    print(
        f"Loading {cfg.name} dataset from {len(data_files)} files, "
        f"({cfg.file_repeats} repeats), "
        f"{os.path.join(data_dir, cfg.data_path_pattern)}"
    )
    if cfg.is_parquet:
        dataset = load_dataset(
            path="parquet",
            data_files=data_files,
            split=split,
            streaming=True,
            verification_mode="no_checks",
        )
    else:
        # THIS STEP IS SLOW FOR GYM MSAS (V LARGE FILES) --- BUT WHY - WHAT HAPPENS?
        dataset = load_dataset(
            "text",
            data_files=data_files,
            split=split,
            streaming=True,
            sample_by="document",
        )
    print("Dataset n shards", dataset.n_shards)
    print("Verifying dataset content:")
    for i, item in enumerate(dataset.take(3)):
        print(f"  Item {i + 1}:")
        for key, value in item.items():
            print(f"    {key}: {value[:100] if isinstance(value, str) else value}")
        print()

    if cfg.holdout_identifiers:
        assert (
            cfg.identifier_col is not None
        ), "Need identifier column for identifier holdout"

    def filter_example(example):
        filter_num_seqs = example["total_num_sequences"] >= (cfg.minimum_sequences or 1)
        # TODO: we need to be very careful with this!
        filter_identifier = (
            cfg.holdout_data_files is None
            or example["identifier"] not in cfg.holdout_identifiers
        )
        return filter_num_seqs and filter_identifier

    dataset = dataset.map(
        preprocess_protein_data,
        batched=False,
        remove_columns=["text"],
        fn_kwargs={"cfg": cfg, "tokenizer": tokenizer},
    ).filter(filter_example)

    return dataset
