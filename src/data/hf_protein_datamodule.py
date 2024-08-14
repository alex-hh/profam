import os
from typing import Any, Dict, List, Optional

from datasets import interleave_datasets
from lightning import LightningDataModule
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerFast

from src.data.family_classification import (
    load_classifier_dataset,
    load_ec_cluster_classifier_dataset,
)
from src.data.pfam_classification import load_pfam_classification_dataset
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
        val_dataset_names: List[str],
        batch_size: int = 8,
        max_tokens: int = 5000,
        evaluate_gym: bool = False,
        keep_gym_gaps: bool = False,
        gym_data_dir: Optional[str] = None,
        max_gym_sequences: Optional[int] = None,
        gym_dms_ids: Optional[List[str]] = None,
        num_workers: Optional[int] = None,
        evaluate_ec_class: bool = True,
        evaluate_ec_cluster_class: bool = True,
        evaluate_pfam_class: bool = False,
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
        self.val_dataset_names = val_dataset_names
        self.num_workers = num_workers
        self.evaluate_gym = evaluate_gym
        self.keep_gym_gaps = keep_gym_gaps
        if self.evaluate_gym:
            self.gym_data_dir = os.path.join(self.data_dir, gym_data_dir)
        self.evaluate_ec_class = evaluate_ec_class
        self.evaluate_ec_cluster_class = evaluate_ec_cluster_class
        self.evaluate_pfam_class = evaluate_pfam_class
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
                if data_key not in self.val_dataset_names:
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
            self.val_datasets = [
                load_protein_dataset(
                    self.dataset_cfgs[v_ds_name],
                    self.tokenizer,
                    self.max_tokens,
                    data_dir=self.data_dir,
                    use_seq_pos=self.use_seq_pos,
                    max_seq_pos=self.max_seq_pos,
                )
                for v_ds_name in self.val_dataset_names
            ]
            self.test_dataset = load_protein_dataset(
                self.dataset_cfgs[self.val_dataset_names[0]],
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
            if self.evaluate_ec_cluster_class:
                self.ec_cluster_class_dataset = load_ec_cluster_classifier_dataset(
                    tokenizer=self.tokenizer,
                    fasta_dir="../data/ec/ec_fastas",
                    val_df_path="data/val/ec_val_clustered_seqs_w_different_ec_nums.csv",
                    max_tokens=self.max_tokens,
                    use_seq_pos=self.use_seq_pos,
                    max_seq_pos=self.max_seq_pos,
                )
            if self.evaluate_pfam_class:
                self.pfam_class_dataset = load_pfam_classification_dataset(
                    tokenizer=self.tokenizer,
                    keep_insertions=True,  # TODO: should be val config
                    to_upper=True,  # TODO: should be val config
                    keep_gaps=True,  # TODO: should be val config
                    pfam_dir="../data/pfam/pfam_eval_splits/clustered_split_fastas",
                    max_tokens=self.max_tokens,
                    use_seq_pos=self.use_seq_pos,
                    max_seq_pos=self.max_seq_pos,
                    num_workers=self.num_workers,
                    max_eval_per_fam=4,
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
                val_ds,
                batch_size=self.batch_size,
                collate_fn=self.collator,
                shuffle=False,
                num_workers=self.num_workers,
            )
            for val_ds in self.val_datasets
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

        if self.evaluate_ec_cluster_class:
            loaders.append(
                DataLoader(
                    self.ec_cluster_class_dataset,
                    batch_size=1,
                    collate_fn=self.collator,
                    shuffle=False,
                )
            )

        if self.evaluate_pfam_class:
            loaders.append(
                DataLoader(
                    self.pfam_class_dataset,
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
