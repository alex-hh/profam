import functools
import os
from typing import Optional

import pandas as pd
from datasets import Dataset
from transformers import PreTrainedTokenizerFast

from src.data import fasta
from src.data import utils as data_utils


def tokenize_msa(sample, tokenizer: PreTrainedTokenizerFast, max_tokens=5000):
    # TODO: fix tokenization. copying hf loader for now
    concatenated_seqs = tokenizer.bos_token + tokenizer.sep_token.join(
        sample["MSA"]
    )  # No EOS token here because the target seq will be added
    tokenized = tokenizer(
        concatenated_seqs, return_tensors="pt", add_special_tokens=False
    )
    sample["input_ids"] = tokenized.input_ids[0]  # no extra dim
    return sample


def tokenize_completions(example, tokenizer: PreTrainedTokenizerFast):
    example["completion_seqs"] = [
        tokenizer.sep_token + seq + tokenizer.sep_token
        for seq in example["completion_seqs"]
    ]
    max_length = max(len(seq) for seq in example["completion_seqs"])
    tokenized = tokenizer(
        example["completion_seqs"],
        return_tensors="pt",
        padding="max_length",  # todo handle the padding in the validation step
        truncation=True,
        max_length=max_length,
        add_special_tokens=False,
    )
    example["completion_ids"] = tokenized.input_ids
    return example


def tokenize(sample, tokenizer: PreTrainedTokenizerFast, **kwargs):
    sample = tokenize_msa(sample, tokenizer, **kwargs)
    sample = tokenize_completions(sample, tokenizer)
    return sample


def load_msa_for_row(
    row, seed, max_tokens, gym_data_dir, keep_wt=False, drop_wt=True, keep_gaps=False
):
    labels, seqs = fasta.read_fasta(
        os.path.join(gym_data_dir, "DMS_msa_files", row["MSA_filename"]),
        keep_insertions=True,
        to_upper=True,
        keep_gaps=keep_gaps,
    )
    sampled_seqs = data_utils.sample_to_max_tokens(
        seqs, seed=seed, keep_first=keep_wt, drop_first=drop_wt, max_tokens=max_tokens
    )
    row["MSA"] = sampled_seqs
    return row


def load_dms_scores_for_row(row, seed, max_mutated_sequences, gym_data_dir):
    dms_df = pd.read_csv(
        os.path.join(gym_data_dir, "DMS_ProteinGym_substitutions", row["DMS_filename"])
    )
    if max_mutated_sequences is not None and max_mutated_sequences < len(dms_df):
        dms_df = dms_df.sample(n=max_mutated_sequences, random_state=seed)
    row["DMS_scores"] = dms_df["DMS_score"].tolist()
    row["completion_seqs"] = dms_df["mutated_sequence"].tolist()
    return row


def build_gym_df(
    dms_ids,
    gym_data_dir: str,
    seed: Optional[int] = None,
    max_mutated_sequences: Optional[int] = None,
    max_tokens: int = 5000,
    keep_gaps: bool = False,
):
    """We pre-load and pre-sample MSAs, ensuring they are same at each validation step."""
    df = pd.read_csv(os.path.join(gym_data_dir, "DMS_substitutions.csv"))
    df = df[df["DMS_id"].isin(dms_ids)].sort_values("DMS_id")
    df = df.apply(
        load_msa_for_row,
        axis=1,
        seed=seed,
        gym_data_dir=gym_data_dir,
        max_tokens=max_tokens,
        keep_gaps=keep_gaps,
    )
    df = df.apply(
        load_dms_scores_for_row,
        axis=1,
        seed=seed,
        max_mutated_sequences=max_mutated_sequences,
        gym_data_dir=gym_data_dir,
    )
    return df[["DMS_id", "MSA", "DMS_scores", "completion_seqs"]]


def load_gym_dataset(
    dms_ids,
    tokenizer,
    seed: Optional[int] = None,
    max_mutated_sequences: Optional[int] = None,
    max_tokens: int = 5000,
    gym_data_dir: str = "data/example_data/ProteinGym",
):
    df = build_gym_df(
        dms_ids,
        gym_data_dir=gym_data_dir,
        seed=seed,
        max_mutated_sequences=max_mutated_sequences,
        max_tokens=max_tokens,
    )
    dataset = Dataset.from_pandas(df, preserve_index=False)
    dataset = dataset.map(
        functools.partial(tokenize, tokenizer=tokenizer, max_tokens=max_tokens),
        batched=False,
        remove_columns=["DMS_id", "MSA", "completion_seqs"],
    )
    # https://discuss.huggingface.co/t/dataset-map-return-only-list-instead-torch-tensors/15767
    dataset.set_format(
        type="torch",
        columns=["input_ids", "completion_ids", "DMS_scores"],
    )
    return dataset
