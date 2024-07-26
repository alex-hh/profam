import os
from typing import Any, Dict, List, Optional

from datasets import interleave_datasets
from lightning import LightningDataModule
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerFast

from src.data.family_classification import load_classifier_dataset
from src.data.proteingym import load_gym_dataset
from src.data.utils import (
    CustomDataCollator,
    ProteinDatasetConfig,
    load_protein_dataset,
)


class ProteinDataModule(LightningDataModule):
    def __init__(
        self,
        dataset_cfgs: Dict[str, ProteinDatasetConfig],
        data_weights: Dict[str, float],
        tokenizer_path: str,
        data_dir: str,
        val_dataset_cfgs: Optional[Dict] = None,
        test_dataset_cfgs: Optional[Dict] = None,
        batch_size: int = 8,
        max_tokens: int = 5000,
        num_workers: Optional[int] = None,
        count_doc_hashes: bool = True,
        use_seq_pos: bool = False,
        max_seq_pos: int = 1024,
    ):
        super().__init__()
        self.dataset_cfgs = dataset_cfgs
        self.val_dataset_cfgs = val_dataset_cfgs
        self.test_dataset_cfgs = test_dataset_cfgs
        self.data_weights = data_weights
        self.batch_size = batch_size
        self.max_tokens = max_tokens
        self.data_dir = data_dir
        self.num_workers = num_workers
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

    def setup_train_datasets(self):
        train_datasets = []
        train_data_weights = []
        for data_key, dataset_config in self.dataset_cfgs.items():
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

    def setup_val_datasets(self):
        self.val_datasets = []
        if self.val_dataset_cfgs is None:
            return
        for v_data_key, config in self.val_dataset_cfgs.items():
            if v_data_key == "gym":
                print("Loading gym dataset", config.gym_dms_ids)
                self.val_datasets.append(
                    load_gym_dataset(
                        dms_ids=config.gym_dms_ids,
                        tokenizer=self.tokenizer,
                        max_eval_seqs=config.max_eval_seqs,
                        gym_data_dir=os.path.join(self.data_dir, config.data_path),
                        max_tokens=self.max_tokens,
                        use_seq_pos=self.use_seq_pos,
                        max_seq_pos=self.max_seq_pos,
                        keep_gaps=config.keep_gaps,
                        num_proc=self.num_workers,
                    )
                )
            elif v_data_key == "ec":
                print("Loading EC class dataset")
                self.val_datasets.append(
                    load_classifier_dataset(
                        fasta_file_pattern=config.data_path,
                        tokenizer=self.tokenizer,
                        max_tokens=self.max_tokens,
                        use_seq_pos=self.use_seq_pos,
                        max_seq_pos=self.max_seq_pos,
                        keep_gaps=config.keep_gaps,
                        to_upper=config.to_upper,
                        keep_insertions=config.keep_insertions,
                        max_eval_seqs=config.max_eval_seqs,
                        num_decoys_per_target=config.num_decoys_per_target,
                    )
                )
            else:
                self.val_datasets.append(
                    load_protein_dataset(
                        self.dataset_cfgs[v_data_key],
                        self.tokenizer,
                        self.max_tokens,
                        data_dir=self.data_dir,
                        use_seq_pos=self.use_seq_pos,
                        max_seq_pos=self.max_seq_pos,
                    )
                )


    def setup_test_datasets(self):
        self.test_datasets = []


    def setup(self, stage: Optional[str] = None) -> None:
        self.setup_train_datasets()
        self.setup_val_datasets()
        self.setup_test_datasets()

    def train_dataloader(self) -> list[DataLoader]:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            collate_fn=self.collator,
            num_workers=self.num_workers,
        )

    def val_dataloader(self) -> list[DataLoader]:
        loaders = []
        for dataset in self.val_datasets:
            loaders.append(
                DataLoader(
                    dataset,
                    batch_size=self.batch_size,
                    collate_fn=self.collator,
                    shuffle=False,
                    num_workers=self.num_workers,
                )
            )
        return loaders

    def test_dataloader(self) -> list[DataLoader]:
        loaders = []
        for dataset in self.test_datasets:
            loaders.append(
                DataLoader(
                    dataset,
                    batch_size=self.batch_size,
                    collate_fn=self.collator,
                    shuffle=False,
                    num_workers=self.num_workers,
                )
            )
        return loaders
