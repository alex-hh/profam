import os
from typing import Dict, List, Optional

from datasets import interleave_datasets
from datasets.distributed import split_dataset_by_node
from datasets.iterable_dataset import IterableDataset
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from src.constants import SEQUENCE_FEATURE_NAMES
from src.data.builders import (
    BaseProteinDataset,
    IterableHFProteinDataset,
    MemoryMappedHFProteinDataset,
    ProteinFamilyMemmapDataset,
    ProteinFamilyMemmapDatasetBuilder,
    ProteinGymDataset,
)
from src.data.collators import DocumentBatchCollator
from src.data.online_sample_mapping import (
    OffsetOnlineDataset,
    WeightedConcatOnlineDataset,
    OnlineSampleMappingDataset,
)
from src.data.tokenizers import ProFamTokenizer


class ProteinDataMixture(LightningDataModule):
    """Data module for training on mixture of datasets.

    total_num_train_samples: estimate of total number of samples across all datasets
        (because of on-the-fly filtering, may not be exact). used to ensure the
        same number of samples are seen on each device when using distributed
        training. If the dataset on a given device has fewer than total_num_train_samples
        samples, it will be repeated to ensure the same number of samples are seen
        on each device. However total_num_train_samples must be no greater than twice
        the number of samples on any single device. TODO: figure out some way of
        raising an error if this is exceeded.
    """

    def __init__(
        self,
        dataset_builders: Dict[str, BaseProteinDataset],
        data_weights: Dict[str, float],
        tokenizer: ProFamTokenizer,
        data_dir: str,
        val_dataset_batch_sizes: Dict[str, int],
        batch_size: int = 8,
        num_workers: Optional[int] = None,
        shuffle: bool = True,
        interleaved: bool = True,
        ignore_gaps: bool = False,
        total_num_train_samples: Optional[int] = None,
        feature_names: Optional[List[str]] = None,
        pack_to_max_tokens: Optional[int] = None,
        prefetch_factor: Optional[int] = None,
        train_datasets: Optional[List[Dataset]] = [],
        # TODO: add data_return_format (needs to be same for all datasets I guess...)
    ):
        super().__init__()
        self.dataset_builders = dataset_builders
        self.data_weights = data_weights
        self.batch_size = batch_size
        self.data_dir = data_dir
        self.val_dataset_batch_sizes = val_dataset_batch_sizes
        print("Val dataset batch sizes", self.val_dataset_batch_sizes)
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.interleaved = interleaved
        self.tokenizer = tokenizer
        self.pack_to_max_tokens = pack_to_max_tokens
        # N.B. feature names only needs to be applied for training
        # i.e. to standardise features across interleaved datasets
        self.feature_names = feature_names or SEQUENCE_FEATURE_NAMES
        self.prefetch_factor = prefetch_factor if num_workers is not None else None
        self.train_collator = DocumentBatchCollator(
            self.tokenizer,
            ignore_gaps=ignore_gaps,
            feature_names=self.feature_names,
            pack_to_max_tokens=self.pack_to_max_tokens,
        )
        self.val_collator = DocumentBatchCollator(
            self.tokenizer,
            ignore_gaps=ignore_gaps,
            feature_names=None,
            pack_to_max_tokens=self.pack_to_max_tokens,
        )
        self._is_setup = False
        self.total_num_train_samples = total_num_train_samples
        self.train_datasets = train_datasets

    def setup(self, stage: Optional[str] = None) -> None:
        # happens on every gpu
        if not self._is_setup:
            train_datasets = []
            train_data_weights = []
            train_dataset_names = []
            world_size = self.trainer.world_size if self.trainer is not None else 1
            print("World size", world_size)

            for data_key, dataset_builder in self.dataset_builders.items():
                assert (
                    dataset_builder.name == data_key
                ), f"Dataset builder name {dataset_builder.name} must match data key {data_key}"
                if data_key not in self.val_dataset_batch_sizes:
                    dataset_weight = self.data_weights.get(data_key, 0)
                    if dataset_weight <= 0:
                        print(
                            f"Skipping dataset {data_key} with weight {dataset_weight}"
                        )
                        continue

                    dataset = dataset_builder.load(
                        data_dir=self.data_dir,
                        world_size=world_size,
                        verbose=False,
                    )
                    dataset = dataset_builder.process(
                        dataset,
                        tokenizer=self.tokenizer,
                        feature_names=self.feature_names,
                        pack_to_max_tokens=self.pack_to_max_tokens,
                    )
                    # unclear how to get a sharded dataset for use with num workers?
                    # actually when using data_files n_shards is equal to n_files
                    # https://huggingface.co/docs/datasets/about_mapstyle_vs_iterable
                    # https://huggingface.co/docs/datasets/v2.20.0/en/package_reference/main_classes#datasets.Dataset.to_iterable_dataset
                    # https://github.com/huggingface/datasets/pull/5735
                    print(
                        f"Dataset {data_key} example batch types",
                        {k: type(v) for k, v in next(iter(dataset)).items()},
                    )
                    train_datasets.append(dataset)
                    # TODO: we could also shuffle individual datasets here - is there a reason we might want to?
                    # https://github.com/huggingface/datasets/issues/6623#issuecomment-2367769573 c.f. currently wont aßffect interleave anyways
                    train_data_weights.append(dataset_weight)
                    train_dataset_names.append(data_key)
            train_data_weights = [
                w / sum(train_data_weights) for w in train_data_weights
            ]

            assert len(train_datasets) > 0
            if len(train_datasets) > 1:
                assert (
                    len(set([type(ds) for ds in train_datasets])) == 1
                ), "All train datasets must be same type"
                if isinstance(train_datasets[0], IterableDataset):
                    self.train_dataset = interleave_datasets(
                        train_datasets,
                        probabilities=train_data_weights,
                        stopping_strategy="all_exhausted",
                        split="train",
                        seed=42,
                    )
                else:
                    # FIXME: pass here number of samples seen and wrap train_dataset in OffsetOnlineDataset
                    print(f"Using interleaved train dataset with {len(train_datasets)} datasets, shuffle = {self.shuffle}, interleaved = {self.interleaved}")
                    print(f"num_samples = {self.total_num_train_samples}")
                    print(f"train_dataset_names = {train_dataset_names}")
                    print(f"train_data_weights = {train_data_weights}")
                    self.train_dataset = WeightedConcatOnlineDataset(
                        datasets=train_datasets,
                        num_samples=self.total_num_train_samples,
                        weights=train_data_weights,
                        seed=42,
                        shuffle=self.shuffle,
                        interleaved= self.interleaved,
                    )
                print(
                    "Interleaved train dataset example types",
                    {k: type(v) for k, v in next(iter(self.train_dataset)).items()},
                )
            else:
                print("Using single dataset", flush=True)
                print(f"Using sampled mapped train dataset, shuffle = {self.shuffle}")
                print(f"num_samples = {self.total_num_train_samples}")
                print(f"train_dataset_names = {train_dataset_names}")
                self.train_dataset = OnlineSampleMappingDataset(
                    dataset=train_datasets[0],
                    num_samples=self.total_num_train_samples,
                    seed=42,
                    shuffle=self.shuffle,
                )

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
                    # number of workers should be less than available GPUs to prevent a computation bottleneck
                    self.num_workers = min(
                        os.cpu_count() * 3 // 4, self.train_dataset.n_shards
                    )
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
                    self.train_dataset = self.train_dataset.with_format(
                        "numpy"
                    )  # otherwise they gen converted to lists
                    assert (
                        self.total_num_train_samples is not None
                    ), "total_num_train_samples must be set for distributed iterable datasets"
                    print(
                        f"Using {self.total_num_train_samples//world_size} samples for training on each device"
                    )
                elif self.total_num_train_samples is None:
                    print(
                        "Warning: total_num_train_samples not needed for world size 1 and will be ignored"
                    )
            else:
                if self.num_workers is None:
                    self.num_workers = int(os.cpu_count() *3 // 4)
                # unnecessary and could slow down in memory datasets
                # self.train_dataset = self.train_dataset.shuffle(seed=42)
                # if self.total_num_train_samples is not None:
                #     print(
                #         "Warning: total_num_train_samples not needed for non iterable datasets and will be ignored"
                #     )

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
                # n.b. this is still going to produce val metrics that are somewhat world-size dependent
                # because of repeating samples to ensure even number of samples per device
                # TODO: ProteinGymDataset should inherit from MemoryMappedHFProteinDataset
                assert isinstance(
                    dataset_builder,
                    (
                        MemoryMappedHFProteinDataset,
                        ProteinGymDataset,
                        ProteinFamilyMemmapDataset,
                        ProteinFamilyMemmapDatasetBuilder,
                    ),
                ), f"Only (MemoryMappedHFProteinDataset, ProteinGymDataset, ProteinFamilyMemmapDataset, ProteinFamilyMemmapDatasetBuilder) supported for val: {v_ds_name} {type(dataset_builder)}"
                dataset = dataset_builder.load(
                    data_dir=self.data_dir,
                    world_size=world_size,
                    verbose=False,
                )
                # N.B. processing (map) will happen once up front for val datasets, not on the fly
                dataset = dataset_builder.process(
                    dataset,
                    tokenizer=self.tokenizer,
                    feature_names=self.feature_names,  # Actually only needed for train bc of interleaving
                    pack_to_max_tokens=self.pack_to_max_tokens,
                )
                if world_size > 1:
                    if isinstance(dataset, IterableHFProteinDataset):
                        # https://github.com/huggingface/datasets/issues/6623
                        assert (
                            dataset.n_shards % world_size == 0
                            and dataset.n_shards % 8 == 0
                        )
                        dataset = split_dataset_by_node(
                            dataset,
                            rank=self.trainer.global_rank,
                            world_size=world_size,
                        )
                        dataset = dataset.with_format("numpy")
                self.val_datasets.append(dataset)
                self.val_dataset_names.append(v_ds_name)
                print(
                    f"{v_ds_name} val dataset example types",
                    {k: type(v) for k, v in next(iter(dataset)).items()},
                )

            self._is_setup = True

    def train_dataloader(self) -> DataLoader:
        # Get samples_seen from trainer if available
        samples_seen = (
            getattr(self.trainer, "samples_seen", 0) if self.trainer is not None else 0
        )
        if samples_seen > 0:
            print(
                f"Checkpoint state has {samples_seen} samples seen: RESUMING NOT YET IMPLEMENTED"
            )

        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            collate_fn=self.train_collator,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers
            > 0,  # https://lightning.ai/docs/pytorch/stable/advanced/speed.html
            prefetch_factor=self.prefetch_factor,
        )

    def val_dataloader(self) -> List[DataLoader]:
        loaders = [
            DataLoader(
                val_ds,
                batch_size=int(self.val_dataset_batch_sizes[val_ds_name]),
                collate_fn=self.val_collator,
                shuffle=False,
                num_workers=self.num_workers // 2,
                persistent_workers=self.num_workers > 1,
                prefetch_factor=self.prefetch_factor,
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
                num_workers=self.num_workers // 2,
                persistent_workers=self.num_workers > 1,
                prefetch_factor=self.prefetch_factor,
            )
        ]
        return loaders


class ProteinMemmapDataModule(LightningDataModule):
    """Data module for training on text memmap datasets."""

    def __init__(
        self,
        train_dataset: Dataset,
        val_dataset: Dataset,
        data_dir: str,
        val_dataset_batch_sizes: Dict[str, int],
        batch_size: int = 8,
        num_workers: Optional[int] = None,
        ignore_gaps: bool = False,
        total_num_train_samples: Optional[int] = None,
        feature_names: Optional[List[str]] = None,
        pack_to_max_tokens: Optional[int] = None,
        prefetch_factor: Optional[int] = None,
        test_dataset: Optional[Dataset] = None,
        tokenizer: Optional[ProFamTokenizer] = None,
    ):
        super().__init__()
        print("\n\n\n\n THIS CLASS SHOULD BE USED FOR DEBUGGING ONLY \n\n\n\n\n")
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.tokenizer = tokenizer
        self.data_dir = data_dir
        self.val_dataset_batch_sizes = val_dataset_batch_sizes
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.feature_names = feature_names
        self.pack_to_max_tokens = pack_to_max_tokens
        self.ignore_gaps = ignore_gaps
        self.total_num_train_samples = total_num_train_samples
        self.tokenizer = tokenizer
        self.train_collator = DocumentBatchCollator(
            self.tokenizer,
            ignore_gaps=ignore_gaps,
            feature_names=self.feature_names,
            pack_to_max_tokens=self.pack_to_max_tokens,
        )
        self.val_collator = DocumentBatchCollator(
            self.tokenizer,
            ignore_gaps=ignore_gaps,
            feature_names=None,
            pack_to_max_tokens=self.pack_to_max_tokens,
        )

    def setup(self, stage: Optional[str] = None) -> None:
        self.val_datasets = [self.val_dataset]
        self.val_dataset_names = ["val"]

    def train_dataloader(self) -> DataLoader:
        # Get samples_seen from trainer if available
        samples_seen = (
            getattr(self.trainer, "samples_seen", 0) if self.trainer is not None else 0
        )
        if samples_seen > 0:
            print(
                f"Checkpoint state has {samples_seen} samples seen: RESUMING NOT YET IMPLEMENTED"
            )

        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            collate_fn=self.train_collator,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers
            > 0,  # https://lightning.ai/docs/pytorch/stable/advanced/speed.html
            prefetch_factor=self.prefetch_factor,
        )

    def val_dataloader(self) -> List[DataLoader]:
        loaders = [
            DataLoader(
                val_ds,
                batch_size=1,
                # batch_size=int(self.val_dataset_batch_sizes[val_ds_name]),
                collate_fn=self.val_collator,
                shuffle=False,
                num_workers=self.num_workers // 2,
                persistent_workers=self.num_workers > 1,
                prefetch_factor=self.prefetch_factor,
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
                num_workers=self.num_workers // 2,
                persistent_workers=self.num_workers > 1,
                prefetch_factor=self.prefetch_factor,
            )
        ]
        return loaders
