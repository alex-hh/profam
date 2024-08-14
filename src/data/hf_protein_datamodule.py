import glob
import os
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from datasets import interleave_datasets, load_dataset
from lightning import LightningDataModule
from omegaconf.listconfig import ListConfig
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerFast

from src.data.family_classification import load_classifier_dataset
from src.data.preprocessing import preprocess_protein_data
from src.data.proteingym import load_gym_dataset
from src.data.utils import CustomDataCollator


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
    # global arguments that will get overridden in load_protein_dataset
    max_tokens: Optional[int] = 5000
    use_seq_pos: bool = False
    max_seq_pos: int = 1024
    shuffle: bool = (True,)
    include_doc_hashes: bool = False

    def set_global_args(
        self,
        max_tokens: int,
        use_seq_pos: bool,
        max_seq_pos: int,
        shuffle: bool,
        include_doc_hashes: bool,
    ):
        self.max_tokens = max_tokens
        self.use_seq_pos = use_seq_pos
        self.max_seq_pos = max_seq_pos
        self.shuffle = shuffle
        self.include_doc_hashes = include_doc_hashes


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
        use_seq_pos=use_seq_pos,
        max_seq_pos=max_seq_pos,
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

    dataset = dataset.map(
        preprocess_protein_data,
        batched=False,
        remove_columns=["text"],
        fn_kwargs={"cfg": cfg, "tokenizer": tokenizer},
    ).filter(lambda x: x["total_num_sequences"] >= (cfg.minimum_sequences or 1))

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
        keep_gym_gaps: bool = False,
        gym_data_dir: Optional[str] = None,
        max_gym_sequences: Optional[int] = None,
        gym_dms_ids: Optional[List[str]] = None,
        num_workers: Optional[int] = None,
        evaluate_ec_class: bool = True,
        count_doc_hashes: bool = True,
        use_seq_pos: bool = False,
        max_seq_pos: int = 1024,
    ):
        super().__init__()
        self.dataset_cfgs = dataset_cfgs
        self.data_weights = data_weights
        self.batch_size = batch_size
        self.max_tokens = max_tokens
        self.data_dir = data_dir
        self.val_dataset_name = val_dataset_name
        self.num_workers = num_workers
        self.evaluate_gym = evaluate_gym
        self.keep_gym_gaps = keep_gym_gaps
        if self.evaluate_gym:
            self.gym_data_dir = os.path.join(self.data_dir, gym_data_dir)
        self.evaluate_ec_class = evaluate_ec_class
        self.max_gym_sequences = max_gym_sequences
        self.gym_dms_ids = gym_dms_ids
        self.use_seq_pos = use_seq_pos
        self.max_seq_pos = max_seq_pos  # max embed index for relative position
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
        self.collator = CustomDataCollator(self.tokenizer, mlm=False)
        self.count_doc_hashes = count_doc_hashes
        self._is_setup = False

    def setup(self, stage: Optional[str] = None) -> None:
        # happens on every gpu
        if not self._is_setup:
            train_datasets = []
            train_data_weights = []
            for data_key, dataset_config in self.dataset_cfgs.items():
                if data_key != self.val_dataset_name:
                    dataset = load_protein_dataset(
                        dataset_config,
                        self.tokenizer,
                        self.max_tokens,
                        data_dir=self.data_dir,
                        include_doc_hashes=self.count_doc_hashes,
                        use_seq_pos=self.use_seq_pos,
                        max_seq_pos=self.max_seq_pos,
                    )
                    # unclear how to get a sharded dataset for use with num workers?
                    # actually when using data_files n_shards is equal to n_files
                    # https://huggingface.co/docs/datasets/about_mapstyle_vs_iterable
                    # https://huggingface.co/docs/datasets/v2.20.0/en/package_reference/main_classes#datasets.Dataset.to_iterable_dataset
                    # https://github.com/huggingface/datasets/pull/5735
                    train_datasets.append(dataset)
                    train_data_weights.append(self.data_weights[data_key])
            train_data_weights = [
                w / sum(train_data_weights) for w in train_data_weights
            ]
            self.train_dataset = interleave_datasets(
                train_datasets,
                probabilities=train_data_weights,
                stopping_strategy="all_exhausted",
                split="train",
                seed=42,
            )
            print("Num shards", self.train_dataset.n_shards)
            if self.num_workers is None:
                self.num_workers = min(os.cpu_count(), self.train_dataset.n_shards)
            print(f"Using {self.num_workers} workers for data loading")
            # will shuffle the shards order and use a shuffle buffer when you start iterating
            # n.b. set_epoch is required in order for shuffling to be correctly randomised
            # - this is handled by ShuffleCallback
            self.train_dataset = self.train_dataset.shuffle(buffer_size=1000, seed=42)
            self.val_dataset = load_protein_dataset(
                self.dataset_cfgs[self.val_dataset_name],
                self.tokenizer,
                self.max_tokens,
                data_dir=self.data_dir,
                use_seq_pos=self.use_seq_pos,
                max_seq_pos=self.max_seq_pos,
            )
            self.test_dataset = load_protein_dataset(
                self.dataset_cfgs[self.val_dataset_name],
                self.tokenizer,
                self.max_tokens,
                data_dir=self.data_dir,
                use_seq_pos=self.use_seq_pos,
                max_seq_pos=self.max_seq_pos,
            )
            if self.evaluate_gym:
                assert self.gym_dms_ids is not None
                print("Loading gym dataset", self.gym_dms_ids)
                self.gym_dataset = load_gym_dataset(
                    dms_ids=self.gym_dms_ids,
                    tokenizer=self.tokenizer,
                    max_mutated_sequences=self.max_gym_sequences,
                    gym_data_dir=self.gym_data_dir,
                    max_tokens=self.max_tokens,
                    use_seq_pos=self.use_seq_pos,
                    max_seq_pos=self.max_seq_pos,
                    keep_gaps=self.keep_gym_gaps,
                    num_proc=self.num_workers,
                )
            if self.evaluate_ec_class:
                # TODO: add other classifier dataset kwargs to config
                self.ec_class_dataset = load_classifier_dataset(
                    "data/example_data/expasy_ec/*.fasta",
                    self.tokenizer,
                    max_tokens=self.max_tokens,
                    use_seq_pos=self.use_seq_pos,
                    max_seq_pos=self.max_seq_pos,
                )

            self._is_setup = True

    def train_dataloader(self) -> List[DataLoader]:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            collate_fn=self.collator,
            num_workers=self.num_workers,
        )

    def gym_dataloader(self) -> DataLoader:
        assert self.evaluate_gym, "Gym validation not enabled"
        return DataLoader(
            self.gym_dataset,
            batch_size=1,  # gym needs batch size 1
            shuffle=False,
        )

    def ec_dataloader(self) -> DataLoader:
        assert self.evaluate_ec_class, "EC class validation not enabled"
        return DataLoader(
            self.ec_class_dataset,
            batch_size=1,
            collate_fn=self.collator,
            shuffle=False,
        )

    def val_dataloader(self) -> List[DataLoader]:
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
                    ),  # n.b. in this case we do standard collation
                ]
            )
        if self.evaluate_ec_class:
            loaders.append(
                DataLoader(
                    self.ec_class_dataset,
                    batch_size=1,
                    collate_fn=self.collator,
                    shuffle=False,
                )
            )
        return loaders

    def test_dataloader(self) -> List[DataLoader]:
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
        if self.evaluate_ec_class:
            loaders.append(
                DataLoader(
                    self.ec_class_dataset,
                    batch_size=1,
                    collate_fn=self.collator,
                    shuffle=False,
                )
            )
        return loaders
