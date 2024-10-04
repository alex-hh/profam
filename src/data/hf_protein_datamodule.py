import os
from typing import Dict, List, Optional

from datasets import interleave_datasets
from datasets.distributed import split_dataset_by_node
from datasets.iterable_dataset import IterableDataset
from lightning import LightningDataModule
from torch.utils.data import DataLoader

from src.constants import SEQUENCE_FEATURE_NAMES
from src.data.datasets import ProteinDatasetConfig, load_protein_dataset
from src.data.family_classification import (
    load_classifier_dataset,
    load_ec_cluster_classifier_dataset,
)
from src.data.hf_datasets import repeat
from src.data.pfam_classification import load_pfam_classification_dataset
from src.data.proteingym import load_gym_dataset
from src.data.utils import CustomDataCollator
from src.utils.tokenizers import ProFamTokenizer


class ProteinDataModule(LightningDataModule):
    """Data module for single dataset training."""

    pass


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
        dataset_cfgs: Dict[str, ProteinDatasetConfig],
        data_weights: Dict[str, float],
        tokenizer: ProFamTokenizer,
        data_dir: str,
        val_dataset_names: List[str],
        batch_size: int = 8,
        max_tokens: int = 8192,
        evaluate_gym: bool = False,
        keep_gym_gaps: bool = False,
        gym_data_dir: Optional[str] = None,
        max_gym_sequences: Optional[int] = None,
        gym_dms_ids: Optional[List[str]] = None,
        use_filtered_gym_msas: bool = False,
        num_workers: Optional[int] = None,
        evaluate_ec_class: bool = True,
        evaluate_ec_cluster_class: bool = True,
        evaluate_pfam_class: bool = False,
        shuffle: bool = True,
        ignore_gaps: bool = False,
        total_num_train_samples: Optional[int] = None,
        feature_names: Optional[List[str]] = None,
    ):
        super().__init__()
        self.dataset_cfgs = dataset_cfgs
        self.data_weights = data_weights
        self.batch_size = batch_size
        self.data_dir = data_dir
        self.val_dataset_names = val_dataset_names
        self.num_workers = num_workers
        self.evaluate_gym = evaluate_gym
        self.keep_gym_gaps = keep_gym_gaps
        self.max_tokens = max_tokens
        self.shuffle = shuffle
        if self.evaluate_gym:
            self.gym_data_dir = os.path.join(self.data_dir, gym_data_dir)
        self.evaluate_ec_class = evaluate_ec_class
        self.evaluate_ec_cluster_class = evaluate_ec_cluster_class
        self.evaluate_pfam_class = evaluate_pfam_class
        self.max_gym_sequences = max_gym_sequences
        self.gym_dms_ids = gym_dms_ids
        self.tokenizer = tokenizer
        self.use_filtered_gym_msas = use_filtered_gym_msas
        # feature names are only required for train collator, when we have different datasets
        self.feature_names = feature_names or SEQUENCE_FEATURE_NAMES
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
        )
        self._is_setup = False
        self.total_num_train_samples = total_num_train_samples

    def setup(self, stage: Optional[str] = None) -> None:
        # happens on every gpu
        if not self._is_setup:
            train_datasets = []
            train_data_weights = []
            train_dataset_names = []
            world_size = self.trainer.world_size
            print("World size", world_size)
            for data_key, dataset_config in self.dataset_cfgs.items():
                if data_key not in self.val_dataset_names:
                    dataset = load_protein_dataset(
                        dataset_config,
                        self.tokenizer,
                        dataset_name=data_key,
                        data_dir=self.data_dir,
                        shuffle=self.shuffle,
                        max_tokens_per_example=self.max_tokens,
                        feature_names=self.feature_names,
                        world_size=world_size,
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
                print("Num shards", self.train_dataset.n_shards)
                if self.num_workers is None:
                    self.num_workers = min(os.cpu_count(), self.train_dataset.n_shards)
                    print(f"Using {self.num_workers} workers for data loading")
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
                    assert (
                        self.total_num_train_samples is not None
                    ), "total_num_train_samples must be set for distributed iterable datasets"
                    print(
                        f"Using {self.total_num_train_samples//world_size} samples for training on each device"
                    )
                    max_train_samples = self.total_num_train_samples // world_size

                    # in case we have fewer samples than we want on some devices, we repeat the dataset (post shuffle)
                    # https://github.com/huggingface/datasets/issues/6623#issuecomment-2377741298
                    # perhaps we could test similar to https://github.com/huggingface/datasets/issues/7156
                    print("Repeating dataset to avoid running out of samples")
                    self.train_dataset = repeat(
                        self.train_dataset, num_times=None
                    ).take(max_train_samples)
                elif self.total_num_train_samples is None:
                    print(
                        "Warning: total_num_train_samples not needed for world size 1 and will be ignored"
                    )
            else:
                if self.num_workers is None:
                    self.num_workers = os.cpu_count()
                self.train_dataset = self.train_dataset.shuffle(seed=42)
                if self.total_num_train_samples is not None:
                    print(
                        "Warning: total_num_train_samples not needed for non iterable datasets and will be ignored"
                    )

            self.val_datasets = []
            for v_ds_name in self.val_dataset_names:
                val_dataset_cfg = self.dataset_cfgs[v_ds_name]
                assert not val_dataset_cfg.shuffle
                # load_protein_dataset subsamples files to a multiple of world_size
                # if using different numbers of devices this could lead to different
                # validation datasets unless we hard-code world_size=8 for val here
                # and always use a world size that is a factor or multiple of 8.
                # TODO: make this more flexible
                val_dataset = load_protein_dataset(
                    self.dataset_cfgs[v_ds_name],
                    self.tokenizer,
                    dataset_name=v_ds_name,
                    data_dir=self.data_dir,
                    max_tokens_per_example=self.max_tokens,
                    world_size=8 if world_size > 1 else 1,  # HACK: hard-coded for now
                )
                if world_size > 1:
                    # https://github.com/huggingface/datasets/issues/6623
                    assert 8 % world_size == 0, (
                        f"HACK: To ensure consistent val we are currently hard-coding world_size=8 for val in"
                        f"load_protein_dataset to ensure n_shards % 8 == 0"
                        f"This will produce consistent val datasets for world sizes that are factors of 8."
                    )
                    assert (
                        val_dataset.n_shards % world_size == 0
                        and val_dataset.n_shards % 8 == 0
                    )
                    val_dataset = split_dataset_by_node(
                        val_dataset,
                        rank=self.trainer.global_rank,
                        world_size=world_size,
                    )
                self.val_datasets.append(val_dataset)

            if self.evaluate_gym:
                # https://huggingface.co/docs/datasets/use_with_pytorch#distributed

                self.gym_dataset = load_gym_dataset(
                    dms_ids=self.gym_dms_ids,
                    tokenizer=self.tokenizer,
                    max_mutated_sequences=self.max_gym_sequences,
                    gym_data_dir=self.gym_data_dir,
                    max_tokens=self.max_tokens,
                    keep_gaps=self.keep_gym_gaps,
                    num_proc=self.num_workers,
                    use_filtered_msa=self.use_filtered_gym_msas,
                )
                if world_size > 1:
                    # calls shard contiguous=True under the hood, which does
                    # https://github.com/huggingface/datasets/blob/3.0.0/src/datasets/arrow_dataset.py#L4609
                    # div = len(dataset) // world_size  0 if len(dataset) < world_size
                    # mod = len(dataset) % world_size
                    # start = rank * div + min(rank, mod)  min(rank, mod) if len(dataset) < world_size
                    # end = start + div + (1 if rank < mod else 0)
                    # so we get an empty slice on ranks for which rank > len(dataset) - hopefully handled fine?
                    self.gym_dataset = split_dataset_by_node(
                        self.gym_dataset,
                        rank=self.trainer.global_rank,
                        world_size=world_size,
                    )

            if self.evaluate_ec_class:
                # TODO: add other classifier dataset kwargs to config
                self.ec_class_dataset = load_classifier_dataset(
                    "data/example_data/expasy_ec/*.fasta",
                    self.tokenizer,
                    max_tokens=self.max_tokens,
                )
                if world_size > 1:
                    self.ec_class_dataset = split_dataset_by_node(
                        self.ec_class_dataset,
                        rank=self.trainer.global_rank,
                        world_size=world_size,
                    )
            if self.evaluate_ec_cluster_class:
                self.ec_cluster_class_dataset = load_ec_cluster_classifier_dataset(
                    tokenizer=self.tokenizer,
                    fasta_dir="../data/ec/ec_fastas",
                    val_df_path="data/val/ec_val_clustered_seqs_w_different_ec_nums.csv",
                    max_tokens=self.max_tokens,
                )
                if world_size > 1:
                    self.ec_cluster_class_dataset = split_dataset_by_node(
                        self.ec_cluster_class_dataset,
                        rank=self.trainer.global_rank,
                        world_size=world_size,
                    )

            if self.evaluate_pfam_class:
                self.pfam_class_dataset = load_pfam_classification_dataset(
                    tokenizer=self.tokenizer,
                    keep_insertions=True,  # TODO: should be val config
                    to_upper=True,  # TODO: should be val config
                    keep_gaps=False,  # TODO: should be val config
                    pfam_dir="data/val_test/pfam/val/clustered_split_fastas",
                    max_tokens=self.max_tokens,
                    num_workers=self.num_workers,
                    max_eval_per_fam=4,
                    use_msa_pos=False,
                )
                if world_size > 1:
                    self.pfam_class_dataset = split_dataset_by_node(
                        self.pfam_class_dataset,
                        rank=self.trainer.global_rank,
                        world_size=world_size,
                    )
            self._is_setup = True

    def train_dataloader(self) -> List[DataLoader]:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            collate_fn=self.train_collator,
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
            collate_fn=self.val_collator,
            shuffle=False,
        )

    def val_dataloader(self) -> List[DataLoader]:
        loaders = [
            DataLoader(
                val_ds,
                batch_size=self.batch_size,
                collate_fn=self.val_collator,
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
                    collate_fn=self.val_collator,
                    shuffle=False,
                )
            )

        if self.evaluate_ec_cluster_class:
            loaders.append(
                DataLoader(
                    self.ec_cluster_class_dataset,
                    batch_size=1,
                    collate_fn=self.val_collator,
                    shuffle=False,
                )
            )

        if self.evaluate_pfam_class:
            loaders.append(
                DataLoader(
                    self.pfam_class_dataset,
                    batch_size=1,
                    collate_fn=self.val_collator,
                    shuffle=False,
                )
            )
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
                    collate_fn=self.val_collator,
                    shuffle=False,
                )
            )
        return loaders
