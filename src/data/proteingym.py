import functools
import os
from typing import Optional

import pandas as pd
from datasets import Dataset
from transformers import PreTrainedTokenizerFast

from src.data import fasta
from src.data import utils as data_utils

GYM_DATA_DIR = os.environ.get(
    "GYM_DATA_DIR", "data/tiny_data/ProteinGym"
)  # TODO update this
print(f"Loading ProteinGym data from {GYM_DATA_DIR}")


def tokenize_msa(sample, tokenizer: PreTrainedTokenizerFast, max_tokens=5000):
    # TODO: fix tokenization. copying hf loader for now
    concatenated_seqs = tokenizer.bos_token + tokenizer.sep_token.join(
        sample["MSA"]
    )  # No EOS token here because the mutated seq will be added
    tokenized = tokenizer(
        concatenated_seqs, return_tensors="pt", add_special_tokens=False
    )
    sample["input_ids"] = tokenized.input_ids[0]  # no extra dim
    return sample


def tokenize_mutants(sample, tokenizer: PreTrainedTokenizerFast):
    sample["mutated_sequences"] = [
        tokenizer.sep_token + seq + tokenizer.sep_token
        for seq in sample["mutated_sequences"]
    ]
    tokenized = tokenizer(
        sample["mutated_sequences"],
        return_tensors="pt",
        padding="max_length",
        add_special_tokens=True,
    )
    sample["completion_ids"] = tokenized.input_ids
    return sample


def tokenize(sample, tokenizer: PreTrainedTokenizerFast, **kwargs):
    sample = tokenize_msa(sample, tokenizer, **kwargs)
    sample = tokenize_mutants(sample, tokenizer)
    return sample


def load_msa_for_row(
    row, seed, max_tokens, keep_wt=False, drop_wt=True, keep_gaps=False
):
    labels, seqs = fasta.read_fasta(
        os.path.join(GYM_DATA_DIR, "DMS_msa_files", row["MSA_filename"]),
        keep_insertions=True,
        to_upper=True,
        keep_gaps=keep_gaps,
    )
    sampled_seqs = data_utils.sample_to_max_tokens(
        seqs, seed=seed, keep_first=keep_wt, drop_first=drop_wt, max_tokens=max_tokens
    )
    row["MSA"] = sampled_seqs
    return row


def load_dms_scores_for_row(row, seed, max_mutated_sequences):
    dms_df = pd.read_csv(
        os.path.join(GYM_DATA_DIR, "DMS_ProteinGym_substitutions", row["DMS_filename"])
    )
    if max_mutated_sequences is not None and max_mutated_sequences < len(dms_df):
        dms_df = dms_df.sample(n=max_mutated_sequences, random_state=seed)
    row["DMS_scores"] = dms_df["DMS_score"].tolist()
    row["mutated_sequences"] = dms_df["mutated_sequence"].tolist()
    return row


def build_gym_df(
    dms_ids,
    seed: Optional[int] = None,
    max_mutated_sequences: Optional[int] = None,
    max_tokens: int = 5000,
    keep_gaps: bool = False,
):
    """We pre-load and pre-sample MSAs, ensuring they are same at each validation step."""
    df = pd.read_csv(os.path.join(GYM_DATA_DIR, "DMS_substitutions.csv"))
    df = df[df["DMS_id"].isin(dms_ids)].sort_values("DMS_id")
    df = df.apply(
        load_msa_for_row, axis=1, seed=seed, max_tokens=max_tokens, keep_gaps=keep_gaps
    )
    df = df.apply(
        load_dms_scores_for_row,
        axis=1,
        seed=seed,
        max_mutated_sequences=max_mutated_sequences,
    )
    return df[["DMS_id", "MSA", "DMS_scores", "mutated_sequences"]]


def load_gym_dataset(
    dms_ids,
    tokenizer,
    seed: Optional[int] = None,
    max_mutated_sequences: Optional[int] = None,
    max_tokens: int = 5000,
):
    df = build_gym_df(
        dms_ids,
        seed=seed,
        max_mutated_sequences=max_mutated_sequences,
        max_tokens=max_tokens,
    )
    dataset = Dataset.from_pandas(df, preserve_index=False)
    dataset = dataset.map(
        functools.partial(tokenize, tokenizer=tokenizer, max_tokens=max_tokens),
        batched=False,
        remove_columns=["DMS_id", "MSA", "mutated_sequences"],
    )
    # https://discuss.huggingface.co/t/dataset-map-return-only-list-instead-torch-tensors/15767
    dataset.set_format(
        type="torch", columns=["input_ids", "completion_ids", "DMS_scores"]
    )
    return dataset
