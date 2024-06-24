import bisect
import itertools
import os
import random
from typing import Any, Dict, Optional

import numpy as np
from datasets import Dataset, load_dataset
from transformers import PreTrainedTokenizerFast

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


def load_protein_dataset(
    cfg: ProteinDatasetConfig,
    tokenizer: PreTrainedTokenizerFast,
    max_tokens: int = 5000,
    data_dir="../data",
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
            truncation=False,  # shouldn't be necessary - bisection should handle.
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
        return tokenized

    if cfg.data_path_pattern is not None:
        data_files = os.path.join(data_dir, cfg.data_path_pattern)
    else:
        assert cfg.data_path_file is not None
        with open(os.path.join(data_dir, cfg.data_path_file), "r") as f:
            data_files = [
                os.path.join(data_dir, data_file) for data_file in f.read().splitlines()
            ]

    print(f"Loading {cfg.name} dataset from {len(data_files)} files")
    dataset = load_dataset(
        "text",
        data_files=data_files,
        split="train",
        streaming=True,
        sample_by="document",
    )
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
