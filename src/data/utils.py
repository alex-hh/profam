import glob
import os
import random
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from datasets import Dataset, load_dataset
from omegaconf.listconfig import ListConfig
from transformers import DataCollatorForLanguageModeling

from src.data.objects import StringObject
from src.data.preprocessing import BasePreprocessor
from src.utils.tokenizers import ProFamTokenizer

# TODO: add things like sequence col, structure col, etc.
# TODO: be careful around loading coords if using alignment - how can we test for this?


def filter_on_length(example, cfg, max_tokens, tokenizer):
    if cfg.length_filter is None:
        return True
    elif cfg.length_filter == "max_seq_pos":
        return any([len(s) <= tokenizer.max_seq_pos - 1 for s in example["sequences"]])
    elif cfg.length_filter == "max_tokens":
        if max_tokens is None:
            return True
        elif cfg.preprocessor.interleave_structure_sequence:
            return (
                max([len(s) for s in example["sequences"]])
                <= (max_tokens // 2) - tokenizer.num_start_tokens - 2
            )
        else:
            return (
                max([len(s) for s in example["sequences"]])
                <= max_tokens - tokenizer.num_start_tokens - 1
            )
    else:
        raise ValueError(f"Unknown length filter {cfg.length_filter}")


class CustomDataCollator:
    """
    Wraps DataCollatorForLanguageModeling
    allows us to include elements which are not
    tensors with seq_len dimension, eg. dataset names
    """

    def __init__(
        self,
        tokenizer,
        mlm=False,
        ignore_gaps: bool = False,
        feature_names: Optional[List[str]] = None,
    ):
        self.tokenizer = tokenizer
        self.base_collator = DataCollatorForLanguageModeling(tokenizer, mlm=mlm)
        self.ignore_gaps = ignore_gaps
        self.feature_names = feature_names

    def __call__(self, examples):
        # TODO: maybe I have an issue with blending data with different keys?
        # need to handle either in collator or by standardising in tokenizer.
        def keep_feature(feature_name):
            return self.feature_names is None or feature_name in self.feature_names

        non_string_data = [
            {k: v for k, v in e.items() if (not isinstance(v, str)) and keep_feature(k)}
            for e in examples
        ]
        string_data = [
            {k: v for k, v in e.items() if isinstance(v, str) and keep_feature(k)}
            for e in examples
        ]
        string_data_keys = set(k for obs in string_data for k in obs.keys())
        try:
            batch = self.base_collator(non_string_data)
        except Exception as e:
            print("Error in collator")
            print(string_data)
            # print(non_string_data)
            raise e
        if self.ignore_gaps:
            batch["labels"][
                batch["labels"] == self.tokenizer.convert_tokens_to_ids("-")
            ] = -100
        # dont predict mask tokens.
        batch["labels"][batch["labels"] == self.tokenizer.mask_token_id] = -100
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


@dataclass
class ProteinDatasetConfig:
    name: str
    preprocessor: Optional[BasePreprocessor] = None
    data_path_pattern: Optional[str] = None
    holdout_data_files: Optional[str] = None
    holdout_identifiers: Optional[List[str]] = None
    identifier_col: Optional[str] = None
    data_path_file: Optional[str] = None
    file_repeats: int = 1
    minimum_sequences: Optional[int] = None
    is_parquet: bool = False
    shuffle: bool = True
    length_filter: Optional[str] = None  # max_tokens, max_seq_pos
    minimum_mean_plddt: Optional[float] = None
    stream: bool = True


def load_protein_dataset(
    cfg: ProteinDatasetConfig,
    tokenizer: ProFamTokenizer,
    data_dir="data",
    split="train",
    max_tokens: Optional[int] = None,
    shuffle: bool = True,
    feature_names: Optional[List[str]] = None,
) -> Dataset:
    if cfg.data_path_pattern is not None:
        # replace hf path resolution with manual glob, to allow repetition
        # https://github.com/huggingface/datasets/blob/98fdc9e78e6d057ca66e58a37f49d6618aab8130/src/datasets/data_files.py#L323
        data_files = glob.glob(os.path.join(data_dir, cfg.data_path_pattern))
        assert (
            len(data_files) > 0
        ), f"No files found for pattern {cfg.data_path_pattern} in {data_dir}"
    else:
        assert cfg.data_path_file is not None
        with open(os.path.join(data_dir, cfg.data_path_file), "r") as f:
            data_files = [
                os.path.join(data_dir, data_file) for data_file in f.read().splitlines()
            ]
            assert all([os.path.exists(f) for f in data_files])

    if cfg.holdout_data_files is not None:
        if isinstance(cfg.holdout_data_files, str):
            holdout_files = [cfg.holdout_data_files]
        else:
            assert isinstance(cfg.holdout_data_files, list) or isinstance(
                cfg.holdout_data_files, ListConfig
            ), f"holdout files is {type(cfg.holdout_data_files)} not list"
            holdout_files = cfg.holdout_data_files

        holdout_files = [os.path.join(data_dir, f) for f in holdout_files]
        assert all(
            [f in data_files for f in holdout_files]
        ), f"Not all holdout files {holdout_files} found in data files"

        all_files = len(data_files)
        data_files = [f for f in data_files if f not in holdout_files]
        print("Excluding", all_files - len(data_files), "holdout files")
        assert len(data_files) > 0, "No files left after holdout"

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
            streaming=cfg.stream,
            verification_mode="no_checks",
        )
    else:
        # THIS STEP IS SLOW FOR GYM MSAS (V LARGE FILES) --- BUT WHY - WHAT HAPPENS?
        # TODO: load identifier?
        assert (
            cfg.holdout_identifiers is None
        ), "Holdout identifiers not supported for fasta"
        dataset = load_dataset(
            "text",
            data_files=data_files,
            split=split,
            streaming=cfg.stream,
            sample_by="document",
        )
    print("Dataset n shards", dataset.n_shards)
    print("Verifying dataset content:")
    for i, item in enumerate(dataset.take(3)):
        print(f"  Item {i + 1}:")
        for key, value in item.items():
            if isinstance(value, str):
                value_to_print = value[:100]
            elif isinstance(value, list):
                # TODO: if its a list of lists we want to print only first few elements
                if isinstance(value[0], list):
                    value_to_print = f"[{value[0][:10]},...]"
                else:
                    value_to_print = f"{value[:3]}..." if len(value) > 3 else value
            else:
                value_to_print = value
            print(f"    {key}: {value_to_print}")
        print()

    if cfg.holdout_identifiers:
        assert (
            cfg.identifier_col is not None
        ), "Need identifier column for identifier holdout"

    def prefilter_example(example):
        # TODO: base this on max_seq_pos
        filter_num_seqs = len(example["sequences"]) >= (cfg.minimum_sequences or 1)
        # TODO: we need to be very careful with this!
        filter_identifier = (
            cfg.holdout_identifiers is None
            or example[cfg.identifier_col] not in cfg.holdout_identifiers
        )
        length_filter = filter_on_length(
            example, cfg=cfg, max_tokens=max_tokens, tokenizer=tokenizer
        )
        if cfg.preprocessor.required_keys is not None:
            for k in cfg.preprocessor.required_keys:
                if k not in example or not example[k]:
                    return False

        if cfg.minimum_mean_plddt is not None:
            if "plddts" in example:
                mean_plddt = np.mean(example["plddts"])
                filter_plddt = mean_plddt >= (cfg.minimum_mean_plddt or 0.0)
            else:
                filter_plddt = True
            return (
                length_filter and filter_plddt and filter_identifier and filter_num_seqs
            )
        else:
            return length_filter and filter_identifier and filter_num_seqs

    def wrapped_preprocess(example):
        """Function to be mapped.

        a map is an instruction for converting an example to a new example.
        it should return a datapoint dict.

        a batched map is an instruction for converting a set of examples to a
        new set of examples (not necessarily of the same size). it should return a dict of lists,
        where the length of the lists determines the size of the new set of examples.
        """
        example = cfg.preprocessor.preprocess_protein_data(
            example,
            tokenizer=tokenizer,
            max_tokens=max_tokens,
            shuffle=shuffle,
        )
        if "coords" in example:
            # https://discuss.huggingface.co/t/dataset-map-return-only-list-instead-torch-tensors/15767
            example["coords"] = example["coords"].tolist()
            example["coords_mask"] = example["coords_mask"].tolist()

        if cfg.preprocessor.cfg.batched_map:
            # Q: should we tolist all tensors?
            assert example["input_ids"].ndim == 2
            batch_size = example["input_ids"].shape[0]
            example["ds_name"] = [cfg.name] * batch_size
            if cfg.identifier_col is not None:
                example["identifier"] = [
                    cfg.name + "/" + example[cfg.identifier_col]
                ] * batch_size
        else:
            example["ds_name"] = cfg.name
            # TODO: get identifier for fasta files...
            if cfg.identifier_col is not None:
                example["identifier"] = cfg.name + "/" + example[cfg.identifier_col]

        return example

    if cfg.preprocessor is not None:
        # Q. how does batched map interact with interleave datasets?
        if dataset.column_names is not None:
            # Q: what causes None? maybe loading text rather than parquet
            remove_columns = [
                c for c in dataset.column_names if c not in (feature_names or [])
            ]  # shouldnt be necessary but is for plddts - bug?
        else:
            remove_columns = None
        # TODO: write a separate batched preprocess entrypoint
        dataset = dataset.filter(prefilter_example).map(
            wrapped_preprocess,
            batched=cfg.preprocessor.cfg.batched_map,
            batch_size=cfg.preprocessor.cfg.map_batch_size,
            remove_columns=remove_columns,
        )
        # n.b. coords is returned as a list...

    return dataset
