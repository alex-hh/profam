import bisect
import glob
import hashlib
import itertools
import os
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from datasets import Dataset, load_dataset
from omegaconf.listconfig import ListConfig
from torch import stack
from transformers import DataCollatorForLanguageModeling, PreTrainedTokenizerFast

from src.data.fasta import convert_sequence_with_positions, read_fasta_sequences


# TODO: in future we might actually want standalone dataset class for
# more flexible customisation (e.g. mapping uniprot ids via db)
@dataclass
class ProteinDatasetConfig:
    name: str
    keep_gaps: bool = False
    data_path_pattern: Optional[str] = None
    holdout_data_files: Optional[str] = None
    data_path_file: Optional[str] = None
    keep_insertions: bool = False
    to_upper: bool = False
    file_repeats: int = 1
    is_parquet: bool = False
    minimum_sequences: Optional[int] = None
    document_tag: str = "[RAW]"
    truncate_after_n_sequences: Optional[int] = None
    use_msa_pos: bool = True  # for msa sequences, if true, position index will be relative to alignment cols


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


def get_flat_seq_pos_from_positions(
    positions,
    max_seq_pos: int = 1024,
    prepend_index=0,
    append_index=0,
    sep_index=0,
    num_start_tokens=1,
    num_end_tokens=1,
):
    # TODO: maybe raise exception if max_seq_pos exceeded rather than duplicating...
    if len(positions) > 0:
        flat_positions = [prepend_index] * num_start_tokens
        for sequence_positions in positions[:-1]:
            # add 1 so that sep doesnt have same position index
            flat_positions += [min(p + 1, max_seq_pos - 1) for p in sequence_positions]
            flat_positions.append(sep_index)
        flat_positions += [min(p + 1, max_seq_pos - 1) for p in positions[-1]]
        flat_positions += [append_index] * num_end_tokens
        return flat_positions
    else:
        return []


def get_seq_pos_from_positions(
    input_ids,
    positions,
    pad_token_id,
    max_seq_pos: int = 1024,
    num_start_tokens=1,
    num_end_tokens=1,
):
    assert input_ids.ndim == 1
    seq_pos = torch.zeros_like(input_ids)
    flat_pos = get_flat_seq_pos_from_positions(
        positions,
        max_seq_pos=max_seq_pos,
        prepend_index=0,
        append_index=0,
        sep_index=0,
        num_start_tokens=num_start_tokens,  # TODO: handle better
        num_end_tokens=num_end_tokens,
    )
    pad_any = torch.argwhere(input_ids == pad_token_id)
    if pad_any.any():
        pad_start = pad_any.min()
    else:
        pad_start = input_ids.shape[0]
    seq_pos[:pad_start] = torch.tensor(flat_pos)
    return seq_pos


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
    tokenizer: PreTrainedTokenizerFast,
    max_tokens: Optional[int] = 5000,
    data_dir="../data",
    split="train",
    include_doc_hashes: bool = False,
    use_seq_pos: bool = False,
    max_seq_pos: int = 1024,
    shuffle: bool = True,
) -> Dataset:
    def preprocess_fasta(example: Dict[str, Any]) -> Dict[str, Any]:
        # N.B. for stockholm format we need to check that sequences aren't split over
        # multiple lines
        if "sequences" in example:
            sequence_iterator = example["sequences"]
            max_sequences_to_preprocess = max_tokens // 10
            if len(sequence_iterator) > max_sequences_to_preprocess:
                selected_indices = np.random.choice(
                    len(sequence_iterator), max_sequences_to_preprocess, replace=False
                )
                sequence_iterator = [sequence_iterator[i] for i in selected_indices]
        else:
            lines = example["text"].split("\n")
            if not len(lines[-1]):
                lines = lines[:-1]
            # min 2 lines per seq, assume at least 10 tks per line
            max_fasta_lines_to_preprocess = (
                max_tokens // 5
            )  # upper bound on lines to proc.
            if len(lines) > max_fasta_lines_to_preprocess:
                lines = subsample_fasta_lines(
                    lines,
                    max_fasta_lines_to_preprocess,
                    shuffle=shuffle,
                )
            sequence_iterator = read_fasta_sequences(
                lines,
                # preserve original sequences before getting positions
                keep_gaps=True if use_seq_pos else cfg.keep_gaps,
                keep_insertions=True if use_seq_pos else cfg.keep_insertions,
                to_upper=False if use_seq_pos else cfg.to_upper,
            )
        if use_seq_pos:
            sequences = []
            positions = []
            for seq in itertools.islice(
                sequence_iterator, cfg.truncate_after_n_sequences
            ):
                seq, pos, _ = convert_sequence_with_positions(
                    seq,
                    keep_gaps=cfg.keep_gaps,
                    keep_insertions=cfg.keep_insertions,
                    to_upper=cfg.to_upper,
                    use_msa_pos=cfg.use_msa_pos,
                )
                sequences.append(seq)
                positions.append(pos)

            # TODO: seed explicitly?
            if shuffle:
                perm = np.random.permutation(len(sequences))
                sequences = [sequences[i] for i in perm]
                positions = [positions[i] for i in perm]
        else:
            sequences = [
                seq
                for seq in itertools.islice(
                    sequence_iterator, cfg.truncate_after_n_sequences
                )
            ]  # necessary for fasta iterator...
            if shuffle:
                perm = np.random.permutation(len(sequences))
                sequences = [sequences[i] for i in perm]

        if max_tokens is not None:
            cumulative_lengths = list(
                itertools.accumulate([len(s) + 1 for s in sequences])
            )  # +1 for separator
            insertion_point = bisect.bisect_left(
                cumulative_lengths,
                max_tokens - 2,
            )  # -2 for doc start and end tokens
        else:
            insertion_point = len(sequences)
        concatenated_seqs = (
            cfg.document_tag
            + tokenizer.bos_token
            + tokenizer.sep_token.join(sequences[:insertion_point])
            + tokenizer.sep_token
        )
        tokenized = tokenizer(
            concatenated_seqs,
            truncation=False,  # shouldnt be necessary: bisection should handle
            max_length=max_tokens,
            return_tensors="pt",
            # padding="longest",
            padding="max_length",
            add_special_tokens=False,
        )
        if max_tokens is not None:
            assert tokenized.input_ids.shape[1] <= max_tokens, (
                tokenized.input_ids.shape[1],
                max_tokens,
            )

        tokenized.data = {k: v.squeeze() for k, v in tokenized.data.items()}
        # tokenized.input_ids is flat now
        tokenized.data["ds_name"] = cfg.name
        tokenized.data["total_num_sequences"] = len(sequences)  # below length threshold
        if include_doc_hashes:
            # identify documents by a hash of the first 512 characters
            tokenized.data["doc_hash"] = hashlib.md5(
                example["text"][:512].encode()
            ).hexdigest()

        if use_seq_pos:
            seq_pos = get_seq_pos_from_positions(
                tokenized.input_ids,
                positions[:insertion_point],
                pad_token_id=tokenizer.pad_token_id,
                max_seq_pos=max_seq_pos,
                num_start_tokens=2,
            )
            tokenized.data["seq_pos"] = seq_pos

        return tokenized

    def batched_preprocess_and_filter(batch):
        batch_dict = {}
        for example_text in batch["text"]:
            example = {"text": example_text}
            processed = preprocess_fasta(example).data
            if (
                cfg.minimum_sequences is None
                or processed["total_num_sequences"] >= cfg.minimum_sequences
            ):
                for k, v in processed.items():
                    if k not in batch_dict:
                        batch_dict[k] = []
                    batch_dict[k].append(v)
        print(
            len(batch["text"]),
            len(batch_dict["ds_name"]) if "ds_name" in batch_dict else 0,
        )
        return batch_dict

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
    # with batched map there is a massive delay before training actually starts - why?
    # dataset = dataset.map(
    #     batched_preprocess_and_filter,
    #     batched=True,
    #     remove_columns=["text"],
    #     batch_size=2,
    # )
    # filter after map also seems to slow things down...
    dataset = dataset.map(
        preprocess_fasta, batched=False, remove_columns=["text"]
    ).filter(lambda x: x["total_num_sequences"] >= (cfg.minimum_sequences or 1))

    return dataset


def sample_to_max_tokens(
    sequences,
    seed: int = None,
    keep_first: bool = False,
    drop_first: bool = False,
    max_tokens: int = 5000,
):
    rng = np.random.default_rng(seed)
    if keep_first:
        # TODO: might want to allow it to be shuffled
        assert not drop_first
        sampled_sequences = [sequences[0]]
        token_count = len(sampled_sequences[0]) + 2  # bos and eos
    else:
        sampled_sequences = []
        token_count = 2

    shuffled_sequences = sequences[1:] if drop_first or keep_first else sequences
    rng.shuffle(shuffled_sequences)
    for seq in shuffled_sequences:
        if token_count + len(seq) + 1 > max_tokens:
            break
        sampled_sequences.append(seq)
        token_count += len(seq) + 1
    return sampled_sequences


def get_token_from_name(name: str, tokenizer: PreTrainedTokenizerFast):
    if name == "bos":
        return tokenizer.bos_token
    elif name == "sep":
        return tokenizer.sep_token
    else:
        pass


def tokenize_msa(
    sample,
    tokenizer: PreTrainedTokenizerFast,
    document_tag: Optional[str] = "[RAW]",
    use_seq_pos: bool = False,
    max_seq_pos: int = 1024,
):
    # TODO: fix tokenization. copying hf loader for now
    concatenated_seqs = (
        document_tag + tokenizer.bos_token + tokenizer.sep_token.join(sample["MSA"])
    )  # No EOS token here because the target seq will be added
    tokenized = tokenizer(
        concatenated_seqs, return_tensors="pt", add_special_tokens=False
    )
    sample["input_ids"] = tokenized.input_ids[0]  # no extra dim
    if use_seq_pos:
        if any([any(c.islower() for c in s) for s in sample["MSA"]]):
            raise NotImplementedError("insertions not supported in seq pos calculation")
        positions = [list(range(1, len(s) + 1)) for s in sample["MSA"]]
        sample["seq_pos"] = get_seq_pos_from_positions(
            sample["input_ids"],
            positions,
            pad_token_id=tokenizer.pad_token_id,
            max_seq_pos=max_seq_pos,
            num_start_tokens=2,
            num_end_tokens=0,
        )
    return sample


def tokenize_completions(
    sample,
    tokenizer: PreTrainedTokenizerFast,
    bos_token="sep",
    use_seq_pos: bool = False,
    max_seq_pos: int = 1024,
):
    max_length = max(len(seq) for seq in sample["completion_seqs"])
    completion_seqs = [
        get_token_from_name(bos_token, tokenizer) + seq + tokenizer.sep_token
        for seq in sample["completion_seqs"]
    ]
    tokenized = tokenizer(
        completion_seqs,
        return_tensors="pt",
        padding="max_length",  # todo handle the padding in the validation step
        truncation=False,  # should be handled elsewhere
        max_length=max_length + 2,  # bos_token and sep_token
        add_special_tokens=False,
    )
    sample["completion_ids"] = tokenized.input_ids
    if use_seq_pos and "completion_seq_pos" not in sample:
        # +1 to match convert_sequence_with_positions
        # get_seq_pos_from_positions adds another offset
        completion_seq_pos = stack(
            [
                get_seq_pos_from_positions(
                    sample["completion_ids"][i],
                    [list(range(1, len(seq) + 1))],
                    pad_token_id=tokenizer.pad_token_id,
                    max_seq_pos=max_seq_pos,
                    num_start_tokens=1,
                )
                for i, seq in enumerate(sample["completion_seqs"])
            ]
        )
        sample["completion_seq_pos"] = completion_seq_pos
    return sample


def tokenize(
    sample,
    tokenizer: PreTrainedTokenizerFast,
    mutant_bos_token="sep",
    use_seq_pos: bool = False,
    max_seq_pos: int = 1024,
    document_tag="[RAW]",
):
    sample = tokenize_msa(
        sample,
        tokenizer,
        document_tag=document_tag,
        use_seq_pos=use_seq_pos,
        max_seq_pos=max_seq_pos,
    )
    if "completion_ids" not in sample:
        # pfam family classification datasets add pre-computed completions_ids
        sample = tokenize_completions(
            sample,
            tokenizer,
            bos_token=mutant_bos_token,
            use_seq_pos=use_seq_pos,
            max_seq_pos=max_seq_pos,
        )
    return sample
