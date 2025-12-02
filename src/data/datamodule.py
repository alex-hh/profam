import os
from typing import Dict, List, Optional

from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from src.constants import SEQUENCE_FEATURE_NAMES
from src.data.builders import ProteinGymDataset
from src.data.collators import DocumentBatchCollator
from src.data.online_sample_mapping import (
    OffsetOnlineDataset,
    OnlineSampleMappingDataset,
    WeightedConcatOnlineDataset,
)
from src.data.samplers import MaxTokensDynamicBatchSampler
from src.data.tokenizers import ProFamTokenizer


class ProteinDataMixture(LightningDataModule):
    """Data module for training on mixture of datasets."""

    def __init__(
        self,
        dataset_builders: Dict[str, Dataset],
        data_weights: Dict[str, float],
        tokenizer: ProFamTokenizer,
        data_dir: str,
        val_dataset_batch_sizes: Dict[str, int],
        batch_size: int = 8,
        num_workers: Optional[int] = None,
        shuffle: bool = True,
        interleaved: bool = True,
        interleaved_block_size: int = 1000,
        ignore_gaps: bool = False,
        feature_names: Optional[List[str]] = None,
        pack_to_max_tokens: Optional[int] = None,
        prefetch_factor: Optional[int] = None,
        test_dataset: Optional[Dataset] = None,
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
        self.interleaved_block_size = interleaved_block_size
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
        self.test_dataset = test_dataset

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
                    if isinstance(dataset_builder, ProteinGymDataset):
                        if getattr(dataset_builder, "_tokenizer", None) is None:
                            dataset_builder._tokenizer = self.tokenizer
                    dataset = dataset_builder

                    print(
                        f"Dataset {data_key} example batch types",
                        {k: type(v) for k, v in next(iter(dataset)).items()},
                    )
                    train_datasets.append(dataset)
                    train_data_weights.append(dataset_weight)
                    train_dataset_names.append(data_key)
            train_data_weights = [
                w / sum(train_data_weights) for w in train_data_weights
            ]
            if stage != "test":
                assert len(train_datasets) > 0
            if len(train_datasets) > 1:
                assert (
                    len(set([type(ds) for ds in train_datasets])) == 1
                ), "All train datasets must be same type"
                print(
                    f"Using interleaved train dataset with {len(train_datasets)} datasets, shuffle = {self.shuffle}, interleaved = {self.interleaved}"
                )
                print(f"train_dataset_names = {train_dataset_names}")
                print(f"train_data_weights = {train_data_weights}")
                self.train_dataset = WeightedConcatOnlineDataset(
                    datasets=train_datasets,
                    weights=train_data_weights,
                    seed=42,
                    shuffle=self.shuffle,
                    interleaved=self.interleaved,
                    interleaved_block_size=self.interleaved_block_size,
                )
                print(
                    "Interleaved train dataset example types",
                    {k: type(v) for k, v in next(iter(self.train_dataset)).items()},
                )
            elif len(train_datasets) == 1:
                print("Using single dataset", flush=True)
                print(f"Using sampled mapped train dataset, shuffle = {self.shuffle}")
                print(f"train_dataset_names = {train_dataset_names}")
                self.train_dataset = OnlineSampleMappingDataset(
                    dataset=train_datasets[0],
                    seed=42,
                    shuffle=self.shuffle,
                )
            if len(train_datasets) > 0:
                # Wrap with OffsetOnlineDataset so that we can skip samples that were
                # already seen when resuming from a checkpoint.  This is required both
                # for OnlineSampleMappingDataset (single-dataset case) and for
                # WeightedConcatOnlineDataset (multi-dataset case).
                if isinstance(
                    self.train_dataset,
                    (OnlineSampleMappingDataset, WeightedConcatOnlineDataset),
                ):
                    self.train_dataset = OffsetOnlineDataset(self.train_dataset)

                # # test speed of loading 1000 samples (uncomment to activate)
                # N = 10000
                # import time
                # print(f"=======> Loading {N} samples from train dataset to test speed...")
                # it = iter(self.train_dataset)
                # start = time.time()
                # for _ in range(N):
                #     sample = next(it)
                # end = time.time()
                # print(f"=======> Loaded {N} samples in {end - start:.2f} seconds, {N / (end - start):.2f} samples/sec")

            if self.num_workers is None:
                self.num_workers = max(int(os.cpu_count() * 3 // 4), 1)
                print(f"Setting num_workers to {self.num_workers}")

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
                # Build validation dataset: use direct dataset for memmap and ProteinGym
                if isinstance(dataset_builder, ProteinGymDataset):
                    if getattr(dataset_builder, "_tokenizer", None) is None:
                        dataset_builder._tokenizer = self.tokenizer
                dataset = dataset_builder

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
        # If resuming from checkpoint, skip already seen samples on iterable datasets
        if samples_seen > 0:
            if isinstance(self.train_dataset, OffsetOnlineDataset):
                # Skip the number of samples already seen
                self.train_dataset = self.train_dataset.set_offset(samples_seen)
                print(
                    f"Skipped first {samples_seen} samples to resume training dataset correctly"
                )
            else:
                print(
                    f"Checkpoint state has {samples_seen} samples seen: RESUMING NOT TAKING EFFECT"
                )

        dataset = self.train_dataset
        world_size = self.trainer.world_size
        rank = self.trainer.global_rank
        batch_sampler = MaxTokensDynamicBatchSampler(
            dataset=dataset,
            size_fn=lambda x: len(x["input_ids"]) if "input_ids" in x else 0,
            world_size=world_size,
            rank=rank,
            max_tokens=self.pack_to_max_tokens if self.pack_to_max_tokens else None,
            batch_size=self.batch_size if not self.pack_to_max_tokens else None,
        )

        return DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            collate_fn=self.train_collator,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers is not None and self.num_workers > 1,
            prefetch_factor=self.prefetch_factor,
        )

    def val_dataloader(self) -> List[DataLoader]:

        loaders = [
            DataLoader(
                val_ds,
                batch_size=int(self.val_dataset_batch_sizes[val_ds_name]),
                collate_fn=self.val_collator,
                shuffle=False,
                num_workers=self.num_workers,
                persistent_workers=self.num_workers is not None
                and self.num_workers > 1,
                prefetch_factor=self.prefetch_factor,
            )
            for val_ds, val_ds_name in zip(self.val_datasets, self.val_dataset_names)
        ]

        world_size = self.trainer.world_size if self.trainer is not None else 1
        rank = self.trainer.global_rank if self.trainer is not None else 0
        loaders = []
        for val_ds, val_ds_name in zip(self.val_datasets, self.val_dataset_names):
            # Explicitly shard non-iterable validation datasets across devices
            # to avoid each rank evaluating the full set.
            sampler = None
            if world_size > 1 and val_ds_name == "proteingym":
                print(
                    f"Using distributed sampler for {val_ds_name} on device rank {rank}"
                )
                sampler = DistributedSampler(
                    val_ds,
                    num_replicas=world_size,
                    rank=rank,
                    shuffle=False,
                    drop_last=False,
                )

            loaders.append(
                DataLoader(
                    val_ds,
                    batch_size=int(self.val_dataset_batch_sizes[val_ds_name]),
                    collate_fn=self.val_collator,
                    shuffle=False,
                    sampler=sampler,
                    num_workers=self.num_workers,
                    persistent_workers=self.num_workers is not None
                    and self.num_workers > 1,
                    prefetch_factor=self.prefetch_factor,
                )
            )
        return loaders

    def test_dataloader(self) -> List[DataLoader]:
        if self.test_dataset is None:
            return self.val_dataloader()
        else:
            loaders = [
                DataLoader(
                    self.test_dataset,
                    batch_size=self.batch_size,
                    collate_fn=self.val_collator,
                    shuffle=False,
                    num_workers=self.num_workers,
                    persistent_workers=self.num_workers is not None
                    and self.num_workers > 1,
                    prefetch_factor=self.prefetch_factor,
                )
            ]
            return loaders
