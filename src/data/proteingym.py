import functools
import os
from typing import List, Optional

import pandas as pd
from datasets import Dataset
from lightning import LightningDataModule
from torch.utils.data import DataLoader
from transformers import DataCollatorForLanguageModeling, PreTrainedTokenizerFast

from src.data import fasta
from src.data import utils as data_utils


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
    row["mutated_sequences"] = dms_df["mutated_sequence"].tolist()
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
    return df[["DMS_id", "MSA", "DMS_scores", "mutated_sequences"]]


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
        remove_columns=["DMS_id", "MSA", "mutated_sequences"],
    )
    # https://discuss.huggingface.co/t/dataset-map-return-only-list-instead-torch-tensors/15767
    dataset.set_format(
        type="torch", columns=["input_ids", "completion_ids", "DMS_scores"]
    )
    return dataset


def load_gym_msa_dataset(
    dms_id,
    tokenizer,
    gym_data_dir: str = "data/example_data/ProteinGym",
    keep_gaps: bool = True,
):
    df = pd.read_csv(os.path.join(gym_data_dir, "DMS_substitutions.csv"))
    row = df[df["DMS_id"] == dms_id].iloc[0]
    labels, seqs = fasta.read_fasta(
        os.path.join(gym_data_dir, "DMS_msa_files", row["MSA_filename"]),
        keep_insertions=True,
        to_upper=True,
        keep_gaps=keep_gaps,
    )
    dataset = Dataset.from_dict({"sequence": seqs})

    def tokenize_sequence(example):
        return tokenizer(example["sequence"], return_tensors="pt")

    dataset = dataset.map(
        tokenize_sequence,
        batched=True,
        remove_columns=["sequence"],
    )
    return dataset


class GymMSADataModule(LightningDataModule):
    """For training on a single protein gym MSA."""

    def __init__(
        self,
        tokenizer_path: str,
        gym_dms_id: str,
        gym_data_dir: str,
        batch_size: int,
        max_gym_sequences: Optional[int] = None,
        num_workers: int = 0,
    ):
        super().__init__()
        self.gym_data_dir = gym_data_dir
        self.batch_size = batch_size
        self.max_gym_sequences = max_gym_sequences
        self.gym_dms_id = gym_dms_id
        self.num_workers = num_workers
        self.tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=tokenizer_path,
            unk_token="[UNK]",
            pad_token="[PAD]",
            bos_token="[start-of-document]",
            sep_token="[SEP]",
            mask_token="[MASK]",
            add_special_tokens=True,
        )
        self.collator = DataCollatorForLanguageModeling(self.tokenizer, mlm=False)
        # TODO: fix to avoid hardcoding
        assert self.gym_dms_id is not None
        assert self.gym_data_dir is not None
        self.gym_dataset = load_gym_dataset(
            dms_ids=[gym_dms_id],
            tokenizer=self.tokenizer,
            max_mutated_sequences=self.max_gym_sequences,
            gym_data_dir=self.gym_data_dir,
        )
        self.msa_dataset = load_gym_msa_dataset(
            dms_id=gym_dms_id,
            tokenizer=self.tokenizer,
            gym_data_dir=self.gym_data_dir,
        )
        ddict = self.msa_dataset.train_test_split(test_size=0.01, seed=42)
        self.train_dataset = ddict["train"]
        self.val_dataset = ddict["test"]

    def setup(self, stage: Optional[str] = None) -> None:
        pass

    def train_dataloader(self) -> list[DataLoader]:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            collate_fn=self.collator,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self) -> list[DataLoader]:
        loaders = [
            DataLoader(
                self.val_dataset,
                batch_size=self.batch_size,
                collate_fn=self.collator,
                shuffle=False,
                num_workers=self.num_workers,
            )
        ]
        loaders.append(
            [
                DataLoader(
                    self.gym_dataset,
                    batch_size=1,  # gym needs batch size 1
                    shuffle=False,
                )  # n.b. in this case we do standard collation
            ]
        )
        return loaders

    def test_dataloader(self) -> list[DataLoader]:
        loaders = [
            DataLoader(
                self.test_dataset,
                batch_size=self.batch_size,
                collate_fn=self.collator,
                shuffle=False,
                num_workers=self.num_workers,
            )
        ]
        loaders.append(
            [
                DataLoader(
                    self.gym_dataset,
                    batch_size=1,  # gym needs batch size 1
                    shuffle=False,
                )  # n.b. in this case we do standard collation
            ]
        )
        return loaders
