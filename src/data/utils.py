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
from transformers import DataCollatorForLanguageModeling, PreTrainedTokenizerFast

from src.data.fasta import _read_fasta_lines


# TOOD: in future we might actually want standalone dataset class for
# more flexible customisation (e.g. mapping uniprot ids via db)
@dataclass
class ProteinDatasetConfig:
    name: str
    keep_gaps: bool = False
    data_path_pattern: Optional[str] = None
    data_path_file: Optional[str] = None
    keep_insertions: bool = False
    to_upper: bool = False
    file_repeats: int = 1
    is_parquet: bool = False
    use_seq_pos: bool = False
    max_seq_pos: Optional[int] = None


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

    def __init__(self, tokenizer, mlm=False):
        self.base_collator = DataCollatorForLanguageModeling(tokenizer, mlm=mlm)

    def __call__(self, examples):
        has_ds_name = "ds_name" in examples[0]
        has_doc_hash = "doc_hash" in examples[0]
        has_pfam_acc = "pfam_acc" in examples[0]
        has_msa_id = "msa_id" in examples[0]
        if has_ds_name or has_doc_hash or has_pfam_acc or has_msa_id:
            if has_ds_name:
                ds_names = [example.pop("ds_name") for example in examples]
            if has_doc_hash:
                doc_hashes = [example.pop("doc_hash") for example in examples]
            if has_pfam_acc:
                pfam_accs = [example.pop("pfam_acc") for example in examples]
            if has_msa_id:
                msa_ids = [example.pop("msa_id") for example in examples]
            try:
                batch = self.base_collator(examples)
            except Exception as e:
                bp=1
            if has_ds_name:
                ds_names_obj = StringObject()
                ds_names_obj.text = ds_names
                batch["ds_name"] = ds_names_obj
            if has_doc_hash:
                doc_hash_obj = StringObject()
                doc_hash_obj.text = doc_hashes
                batch["doc_hash"] = doc_hash_obj
            if has_pfam_acc:
                pfam_acc_obj = StringObject()
                pfam_acc_obj.text = pfam_accs
                batch["pfam_acc"] = pfam_acc_obj
            if has_msa_id:
                msa_id_obj = StringObject()
                msa_id_obj.text = msa_ids
                batch["msa_id"] = msa_id_obj
        else:
            batch = self.base_collator(examples)
        return batch


def get_seq_pos(
    input_ids,
    sep_token_id,
    max_seq_pos: int = 1024,
    last_aa_token_id=24,
):
    """
    returns a tensor representing the sequence position
    of each token relative to the last SEP token
    0-indexed
    assumes that the zeroth token is not a residue
    Note that PAD tokens also get assigned seq positions
    after reaching the max_seq_pos it will just repeat
    the last position index
    """
    seq_pos_ids = torch.zeros_like(input_ids)
    current_position = 0
    for i, token_id in enumerate(input_ids):
        if token_id == sep_token_id or i == 0:
            current_position = 0
            assert (
                token_id <= last_aa_token_id,
                "First token should not represent a residue",
            )
        else:
            seq_pos_ids[i] = current_position
            if current_position < max_seq_pos - 1:
                # don't add position indices higher than max_seq_pos
                current_position += 1
    return seq_pos_ids


def load_protein_dataset(
    cfg: ProteinDatasetConfig,
    tokenizer: PreTrainedTokenizerFast,
    max_tokens: int = 5000,
    data_dir="../data",
    split="train",
    include_doc_hashes: bool = False,
    use_seq_pos: bool = False,
    max_seq_pos: int = None,
) -> Dataset:
    def preprocess_fasta(example: Dict[str, Any]) -> Dict[str, Any]:
        sequences = [
            seq
            for _, seq in _read_fasta_lines(
                example["text"].split("\n"),
                keep_gaps=cfg.keep_gaps,
                keep_insertions=cfg.keep_insertions,
                to_upper=cfg.to_upper,
            )
        ]
        random.shuffle(sequences)
        cumulative_lengths = list(
            itertools.accumulate([len(s) + 1 for s in sequences])
        )  # +1 for separator
        insertion_point = bisect.bisect_left(
            cumulative_lengths,
            max_tokens - 2,
        )  # -2 for doc start and end tokens
        concatenated_seqs = (
            tokenizer.bos_token
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
        assert tokenized.input_ids.shape[1] <= max_tokens, (
            tokenized.input_ids.shape[1],
            max_tokens,
        )

        tokenized.data = {k: v.squeeze() for k, v in tokenized.data.items()}
        tokenized.data["ds_name"] = cfg.name
        if include_doc_hashes:
            # identify documents by a hash of the first 512 characters
            tokenized.data["doc_hash"] = hashlib.md5(
                example["text"][:512].encode()
            ).hexdigest()

        if use_seq_pos:
            tokenized.data["seq_pos"] = get_seq_pos(
                tokenized.input_ids, tokenizer.sep_token_id, max_seq_pos=max_seq_pos
            )

        return tokenized

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
        dataset = load_dataset(
            "text",
            data_files=data_files,
            split=split,
            streaming=True,
            sample_by="document",
        )
    print("Dataset n shards", dataset.n_shards)
    # Verify the dataset by looking at the first few items
    print("Verifying dataset content:")
    for i, item in enumerate(dataset.take(3)):
        print(f"  Item {i + 1}:")
        for key, value in item.items():
            print(f"    {key}: {value[:100] if isinstance(value, str) else value}")
        print()
    # TODO: possibly we could speed this up by batching...
    dataset = dataset.map(preprocess_fasta, batched=False, remove_columns=["text"])

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
