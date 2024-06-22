import bisect
import glob
import itertools
import os
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from datasets import Dataset, interleave_datasets, load_dataset
from lightning import LightningDataModule
from torch.utils.data import DataLoader
from transformers import DataCollatorForLanguageModeling, PreTrainedTokenizerFast

from src.data.fasta import _read_fasta_lines
from src.data.proteingym import load_gym_dataset


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
    split="train",
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
        split=split,
        streaming=True,
        sample_by="document",
    )
    dataset = dataset.map(preprocess_fasta, batched=False, remove_columns=["text"])

    return dataset


class ProteinDataModule(LightningDataModule):
    def __init__(
        self,
        dataset_cfgs: Dict[str, ProteinDatasetConfig],
        data_weights: Dict[str, float],
        tokenizer_path: str,
        data_dir: str,
        val_dataset_name: str,
        batch_size: int = 8,
        max_tokens: int = 5000,
        evaluate_gym: bool = False,
        gym_data_dir: Optional[str] = None,
        max_gym_sequences: Optional[int] = None,
        gym_dms_ids: Optional[List[str]] = None,
        num_workers: Optional[int] = None,
        num_shards: Optional[int] = 8,
    ):
        super().__init__()
        self.dataset_cfgs = dataset_cfgs
        self.data_weights = data_weights
        self.batch_size = batch_size
        self.max_tokens = max_tokens
        self.data_dir = data_dir
        self.val_dataset_name = val_dataset_name
        self.num_shards = num_shards
        if num_workers is None:
            num_workers = min(os.cpu_count(), self.num_shards) or 1
        self.num_workers = num_workers
        self.evaluate_gym = evaluate_gym
        self.gym_data_dir = os.path.join(self.data_dir, gym_data_dir)
        self.max_gym_sequences = max_gym_sequences
        self.gym_dms_ids = gym_dms_ids
        self.tokenizer_path = tokenizer_path
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

        if self.evaluate_gym:
            # TODO: fix to avoid hardcoding
            assert self.gym_dms_ids is not None
            assert self.gym_data_dir is not None
            self.gym_dataset = load_gym_dataset(
                dms_ids=self.gym_dms_ids,
                tokenizer=self.tokenizer,
                max_mutated_sequences=self.max_gym_sequences,
                gym_data_dir=self.gym_data_dir,
            )

    def setup(self, stage: Optional[str] = None) -> None:
        if self.num_workers > 0:
            os.environ["TOKENIZERS_PARALLELISM"] = "true"
            print(f"Using {self.num_workers} workers for data loading")

        train_datasets = []
        train_data_weights = []
        for data_key, dataset_config in self.dataset_cfgs.items():
            if data_key != self.val_dataset_name:
                dataset = load_protein_dataset(
                    dataset_config,
                    self.tokenizer,
                    self.max_tokens,
                    split="train",
                    data_dir=self.data_dir,
                )
                # unclear how to get a sharded dataset for use with num workers?
                # https://huggingface.co/docs/datasets/about_mapstyle_vs_iterable
                # https://huggingface.co/docs/datasets/v2.20.0/en/package_reference/main_classes#datasets.Dataset.to_iterable_dataset
                # https://github.com/huggingface/datasets/pull/5735
                train_datasets.append(dataset)
                train_data_weights.append(self.data_weights[data_key])
        self.train_dataset = interleave_datasets(
            train_datasets,
            probabilities=train_data_weights,
            stopping_strategy="all_exhausted",
            split="train",
            seed=42,
        )
        self.val_dataset = load_protein_dataset(
            self.dataset_cfgs[self.val_dataset_name],
            self.tokenizer,
            self.max_tokens,
            data_dir=self.data_dir,
        )
        self.test_dataset = load_protein_dataset(
            self.dataset_cfgs[self.val_dataset_name],
            self.tokenizer,
            self.max_tokens,
            data_dir=self.data_dir,
        )
        if self.evaluate_gym:
            assert self.gym_dms_ids is not None
            self.gym_dataset = load_gym_dataset(
                dms_ids=self.gym_dms_ids,
                tokenizer=self.tokenizer,
                max_mutated_sequences=self.max_gym_sequences,
                gym_data_dir=self.gym_data_dir,
            )

    def train_dataloader(self) -> list[DataLoader]:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            collate_fn=self.collator,
            num_workers=self.num_workers,
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
        if self.evaluate_gym:
            loaders.append(
                [
                    DataLoader(
                        self.gym_dataset,
                        batch_size=1,  # gym needs batch size 1
                        shuffle=False,
                    )  # n.b. in this case we do standard collation
                ]
            )
        print("Val loaders", loaders)
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
        if self.evaluate_gym:
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
