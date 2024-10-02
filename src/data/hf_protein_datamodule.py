import os
from typing import Dict, List, Optional

from datasets import interleave_datasets
from datasets.distributed import split_dataset_by_node
from datasets.iterable_dataset import IterableDataset
from lightning import LightningDataModule
from torch.utils.data import DataLoader

from src.data.datasets import ProteinDatasetConfig
from src.data.utils import CustomDataCollator
from src.utils.tokenizers import ProFamTokenizer

# N.B. feature names only needs to be applied for training
# i.e. to standardise features across interleaved datasets
DEFAULT_FEATURE_NAMES = [
    "input_ids",
    "attention_mask",
    "labels",
    "ds_name",
    "original_size",
    "seq_pos",
    "identifier",
    "coords",
    "coords_mask",
    "interleaved_coords_mask",
    "aa_mask",
    "plddts",
    "structure_mask",
]


class ProteinDataModule(LightningDataModule):
    """Data module for single dataset training."""

    pass


class ProteinDataMixture(LightningDataModule):
    """Data module for training on mixture of datasets."""

    def __init__(
        self,
        dataset_builders: Dict[str, ProteinDatasetConfig],
        data_weights: Dict[str, float],
        tokenizer: ProFamTokenizer,
        data_dir: str,
        val_dataset_batch_sizes: Dict[str, int],
        batch_size: int = 8,
        max_tokens: int = 8192,
        num_workers: Optional[int] = None,
        shuffle: bool = True,
        ignore_gaps: bool = False,
        feature_names: Optional[List[str]] = None,
    ):
        super().__init__()
        self.dataset_builders = dataset_builders
        self.data_weights = data_weights
        self.batch_size = batch_size
        self.data_dir = data_dir
        self.val_dataset_batch_sizes = val_dataset_batch_sizes
        print("Val dataset batch sizes", self.val_dataset_batch_sizes)
        self.num_workers = num_workers
        self.max_tokens = max_tokens
        self.shuffle = shuffle
        self.tokenizer = tokenizer
        self.feature_names = feature_names or DEFAULT_FEATURE_NAMES
        self.train_collator = CustomDataCollator(
            self.tokenizer,
            mlm=False,
            ignore_gaps=ignore_gaps,
            feature_names=self.feature_names,
        )
        self.val_collator = CustomDataCollator(
            self.tokenizer,
            mlm=False,
            ignore_gaps=ignore_gaps,
            feature_names=None,
        )
        self._is_setup = False

    def setup(self, stage: Optional[str] = None) -> None:
        # happens on every gpu
        if not self._is_setup:
            train_datasets = []
            train_data_weights = []
            train_dataset_names = []
            world_size = self.trainer.world_size
            print("World size", world_size)
            for data_key, dataset_builder in self.dataset_builders.items():
                assert (
                    dataset_builder.name == data_key
                ), f"Dataset builder name {dataset_builder.name} must match data key {data_key}"
                if data_key not in self.val_dataset_batch_sizes:
                    dataset = dataset_builder.load(
                        data_dir=self.data_dir,
                        world_size=world_size,
                        verbose=False,
                    )
                    dataset = dataset_builder.process(
                        dataset,
                        tokenizer=self.tokenizer,
                        max_tokens_per_example=self.max_tokens,
                        shuffle_proteins_in_document=self.shuffle,
                        feature_names=self.feature_names,
                    )
                    # unclear how to get a sharded dataset for use with num workers?
                    # actually when using data_files n_shards is equal to n_files
                    # https://huggingface.co/docs/datasets/about_mapstyle_vs_iterable
                    # https://huggingface.co/docs/datasets/v2.20.0/en/package_reference/main_classes#datasets.Dataset.to_iterable_dataset
                    # https://github.com/huggingface/datasets/pull/5735
                    print(
                        f"Dataset {data_key} keys in example batch",
                        list(next(iter(dataset)).keys()),
                    )
                    train_datasets.append(dataset)
                    # TODO: we could also shuffle individual datasets here - is there a reason we might want to?
                    # https://github.com/huggingface/datasets/issues/6623#issuecomment-2367769573 c.f. currently wont aßffect interleave anyways
                    train_data_weights.append(self.data_weights[data_key])
                    train_dataset_names.append(data_key)
            train_data_weights = [
                w / sum(train_data_weights) for w in train_data_weights
            ]

            assert len(train_datasets) > 0
            if len(train_datasets) > 1:
                self.train_dataset = interleave_datasets(
                    train_datasets,
                    probabilities=train_data_weights,
                    stopping_strategy="all_exhausted",
                    split="train",
                    seed=42,
                )
            else:
                self.train_dataset = train_datasets[0]

            if isinstance(self.train_dataset, IterableDataset):
                # c.f. iterable dataset examples...
                # will shuffle the shards order and use a shuffle buffer when you start iterating
                # n.b. set_epoch is required in order for shuffling to be correctly randomised
                # - this is handled by ShuffleCallback
                # TODO: configure seed - although non-null seed prob important for ddp?
                # or does split_dataset_by_node synchronise the state of the data?
                # no - seeding is required. in face an error will be raised if not.
                # split dataset by node sets distributed config.
                # https://github.com/huggingface/datasets/blob/2eb4edb97e1a6af2ea62738ec58afbd3812fc66e/src/datasets/iterable_dataset.py#L1707
                self.train_dataset = self.train_dataset.shuffle(
                    buffer_size=1000, seed=42
                )
                print("Num shards", self.train_dataset.n_shards)
                if self.num_workers is None:
                    self.num_workers = min(os.cpu_count(), self.train_dataset.n_shards)
                    print(f"Using {self.num_workers} workers for data loading")
                # TODO: verify that non-iterable datasets are split automatically (e.g. by lightning...)
                if world_size > 1:
                    assert (
                        self.train_dataset.n_shards % world_size == 0
                    )  # handled in load_protein_dataset
                    # If the dataset has a number of shards that is a factor of world_size (i.e. if
                    # dataset.n_shards % world_size == 0), then the shards are evenly assigned across
                    # the nodes, which is the most optimized. Otherwise, each node keeps 1 example out of
                    # world_size, skipping the other examples.
                    # https://huggingface.co/docs/datasets/en/package_reference/main_classes#datasets.distributed.split_dataset_by_node
                    self.train_dataset = split_dataset_by_node(
                        self.train_dataset,
                        rank=self.trainer.global_rank,
                        world_size=world_size,
                    )
            else:
                self.train_dataset = self.train_dataset.shuffle(seed=42)
            self.val_datasets = []
            self.val_dataset_names = []
            for v_ds_name, val_batch_size in self.val_dataset_batch_sizes.items():
                if int(val_batch_size) > 1:
                    print(
                        "Warning: val_batch_size > 1 will not work for scoring validations (fine for standard val datasets)"
                    )
                dataset_builder = self.dataset_builders[v_ds_name]
                assert (
                    dataset_builder.name == v_ds_name
                ), f"Dataset builder name {dataset_builder.name} must match data key {v_ds_name}"
                dataset = dataset_builder.load(
                    data_dir=self.data_dir,
                    world_size=world_size,
                    verbose=False,
                )
                dataset = dataset_builder.process(
                    dataset,
                    tokenizer=self.tokenizer,
                    max_tokens_per_example=self.max_tokens,
                    shuffle_proteins_in_document=self.shuffle,
                    feature_names=self.feature_names,
                )
                if world_size > 1:
                    # https://github.com/huggingface/datasets/issues/6623
                    assert 8 % world_size == 0, (
                        f"HACK: To ensure consistent val we are currently hard-coding world_size=8 for val in"
                        f"load_protein_dataset to ensure n_shards % 8 == 0"
                        f"This will produce consistent val datasets for world sizes that are factors of 8."
                    )
                    assert (
                        dataset.n_shards % world_size == 0 and dataset.n_shards % 8 == 0
                    )
                    dataset = split_dataset_by_node(
                        dataset,
                        rank=self.trainer.global_rank,
                        world_size=world_size,
                    )
                self.val_datasets.append(dataset)
                self.val_dataset_names.append(v_ds_name)

            self._is_setup = True

    def train_dataloader(self) -> List[DataLoader]:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            collate_fn=self.train_collator,
            num_workers=self.num_workers,
        )

    def val_dataloader(self) -> List[DataLoader]:
        loaders = [
            DataLoader(
                val_ds,
                batch_size=int(self.val_dataset_batch_sizes[val_ds_name]),
                collate_fn=self.val_collator,
                shuffle=False,
                num_workers=self.num_workers,
            )
            for val_ds, val_ds_name in zip(self.val_datasets, self.val_dataset_names)
        ]
        return loaders

    def test_dataloader(self) -> List[DataLoader]:
        loaders = [
            DataLoader(
                self.test_dataset,
                batch_size=self.batch_size,
                collate_fn=self.val_collator,
                shuffle=False,
                num_workers=self.num_workers,
            )
        ]
        return loaders
