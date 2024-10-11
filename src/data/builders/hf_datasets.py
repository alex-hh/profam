"""For core hf related dataset building c.f. https://huggingface.co/docs/datasets/en/dataset_script.

These classes are different in that they are more focussed on preprocessing.
It might be useful however to move towards the standardised splits.
- although this is basically just directory-based
"""
import glob
import math
import os
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from datasets import Dataset, Features, load_dataset
from omegaconf import ListConfig

from src.constants import TOKENIZED_FEATURE_TYPES
from src.data.objects import ProteinDocument
from src.data.processors import (
    ProteinDocumentPreprocessor,
    backbone_coords_from_example,
)
from src.data.tokenizers import ProFamTokenizer
from src.utils.utils import np_random

from .base import BaseProteinDataset
from .utils import filter_on_length, uniformly_sample_clusters


@dataclass
class HFProteinDatasetConfig:
    # construction from files
    data_path_pattern: Optional[str] = None
    holdout_data_files: Optional[str] = None
    data_path_file: Optional[str] = None
    minimum_sequences: Optional[int] = None
    file_repeats: int = 1
    file_type: str = "parquet"  # or "text", "json"
    # filters
    max_sequences_per_document: Optional[int] = None
    holdout_identifiers: Optional[List[str]] = None
    required_keys: Optional[List[str]] = None
    length_filter: Optional[str] = None  # max_tokens, max_res_pos_in_seq
    minimum_mean_plddt: Optional[float] = None
    # processing
    batched_map: bool = False
    map_batch_size: int = 100
    process_online: bool = True
    # document-building
    sequence_col: str = "sequences"
    identifier_col: Optional[str] = "fam_id"
    structure_tokens_col: Optional[str] = None
    infer_representative_from_identifier: bool = False
    sample_uniformly_from_col: Optional[str] = None  # for redundancy-aware sampling
    concatenate_short_documents: bool = False
    # n.b. pack_to_max_tokens could be applied either before or after interleaving
    # doing it before is a bit more flexible because it allows for different max tokens
    # for different datasets
    pack_to_max_tokens: Optional[
        int
    ] = None  # only really compatible with map so specific to HF for now

    def __post_init__(self):
        if self.concatenate_short_documents:
            assert self.batched_map, "concatenate_short_documents requires batched_map"
            assert (
                self.padding == "do_not_pad"
            ), "padding must be do_not_pad if concatenate_short_documents is True"


def random_subsample(arr, n, seed: Optional[int] = None):
    rnd = np_random(seed)
    return rnd.choice(arr, min(n, len(arr)), replace=False)


def prepare_data_files(
    data_dir: str,
    cfg: HFProteinDatasetConfig,
) -> List[str]:
    """
    Prepare and filter data files based on configuration.

    Args:
        data_dir (str): Directory containing data files.
        cfg (HFProteinDatasetConfig): Configuration object.
        world_size (int): Number of parallel processes.
        stream (bool): Whether to stream the data.

    Returns:
        List[str]: List of prepared data file paths.
    """
    # Resolve data files
    if cfg.data_path_pattern:
        data_files = glob.glob(os.path.join(data_dir, cfg.data_path_pattern))
        assert (
            data_files
        ), f"No files found for pattern {cfg.data_path_pattern} in {data_dir}"
    elif cfg.data_path_file:
        with open(os.path.join(data_dir, cfg.data_path_file), "r") as f:
            data_files = [os.path.join(data_dir, line.strip()) for line in f]
        assert all(
            os.path.exists(f) for f in data_files
        ), "Some specified files do not exist"
    else:
        raise ValueError("Either data_path_pattern or data_path_file must be specified")

    # Handle holdout files
    if cfg.holdout_data_files:
        holdout_files = [
            os.path.join(data_dir, f)
            for f in (
                cfg.holdout_data_files
                if isinstance(cfg.holdout_data_files, (list, ListConfig))
                else [cfg.holdout_data_files]
            )
        ]
        assert all(
            f in data_files for f in holdout_files
        ), "Not all holdout files found in data files"
        data_files = [f for f in data_files if f not in holdout_files]
        print(
            f"Excluded {len(holdout_files)} holdout files. {len(data_files)} files remaining."
        )
        assert data_files, "No files left after holdout"

    # Prepare final list of data files
    data_files = sorted(data_files) * cfg.file_repeats
    print(f"Loading dataset from {len(data_files)} files ({cfg.file_repeats} repeats)")

    return data_files


class FileBasedHFProteinDataset(BaseProteinDataset):
    def __init__(
        self,
        name: str,
        cfg: HFProteinDatasetConfig,
        preprocessor: Optional[ProteinDocumentPreprocessor] = None,
        required_keys: Optional[List[str]] = None,
    ):
        super().__init__(name, preprocessor)
        self.cfg = cfg
        self.required_keys = required_keys

    def get_data_files(self, data_dir: str, world_size: int):
        return prepare_data_files(data_dir, self.cfg)

    def map_fn(
        self,
        example_or_examples,
        tokenizer,
        feature_names: Optional[List[str]] = None,
    ):
        if self.cfg.batched_map:
            # Assert that tokenizer isn't padding to fixed length
            examples = self.batched_preprocess_examples(
                example_or_examples,
                tokenizer,
                pack_to_max_tokens=self.cfg.pack_to_max_tokens,
            )
            return examples
        else:
            example = self.preprocess_example(example_or_examples, tokenizer)
            return example

    def load(
        self,
        data_dir="data",
        world_size: int = 1,
        verbose: bool = False,
    ):
        data_files = self.get_data_files(data_dir, world_size=world_size)
        print("data_files", data_files[:5])
        load_kwargs = {"sample_by": "document"} if self.cfg.file_type == "text" else {}
        dataset = load_dataset(
            self.cfg.file_type,
            data_files=data_files,
            split="train",  # just automatically assigns all files to train - get this 'split'
            streaming=True,
            **load_kwargs,
        ).with_format(
            self.cfg.return_format
        )  # formatting needs to be set before filter/map

        print(f"Dataset n shards: {dataset.n_shards}")
        if verbose:
            print("Verifying dataset content:")
            for i, item in enumerate(dataset.take(3), 1):
                print(f"  Item {i}:")
                for key, value in item.items():
                    value_to_print = (
                        value[:100]
                        if isinstance(value, str)
                        else f"[{value[0][:10]},...]"
                        if isinstance(value, list) and isinstance(value[0], list)
                        else f"{value[:3]}..."
                        if isinstance(value, list) and len(value) > 3
                        else value
                    )
                    print(f"    {key}: {value_to_print}")
                print()

        return dataset

    def filter_fn(
        self,
        example,
        tokenizer: ProFamTokenizer = None,
    ):
        if self.cfg.required_keys is not None:
            for k in self.cfg.required_keys:
                if k not in example or example[k] is None:
                    return False

        sequence_count = len(example[self.cfg.sequence_col])
        min_required = self.cfg.minimum_sequences or 1

        filters = [
            sequence_count >= min_required,
            self.cfg.holdout_identifiers is None
            or example[self.cfg.identifier_col] not in self.cfg.holdout_identifiers,
            filter_on_length(
                example,
                filter_type=self.cfg.length_filter,
                max_tokens=self.preprocessor.cfg.max_tokens_per_example,
                tokenizer=tokenizer,
                sequence_col=self.cfg.sequence_col,
                identifier_col=self.cfg.identifier_col,
                interleave_structure_sequence=self.interleave_structure_sequence,
            ),
        ]

        return all(filters)

    def filter(
        self,
        dataset,
        tokenizer: ProFamTokenizer,
    ):
        return dataset.filter(
            self.filter_fn,
            fn_kwargs={
                "tokenizer": tokenizer,
            },
        )


class MemoryMappedHFProteinDataset(FileBasedHFProteinDataset):
    """File-based builder for a non-iterable Dataset.

    (Not actually necessarily memory-mapped, because can be used with in_memory=True)
    """

    def __init__(
        self,
        name: str,
        cfg: HFProteinDatasetConfig,
        preprocessor: Optional[ProteinDocumentPreprocessor] = None,
        required_keys: Optional[List[str]] = None,
    ):
        """process_online defaults to true for consistency with StreamedProteinDatasetBuilder."""
        super().__init__(
            name, cfg=cfg, preprocessor=preprocessor, required_keys=required_keys
        )

    def process(
        self,
        dataset: Dataset,
        tokenizer: ProFamTokenizer,
        feature_names: Optional[List[str]] = None,
    ):
        """Speed issues:

        https://huggingface.co/docs/datasets/en/about_mapstyle_vs_iterable#speed-differences

        However as soon as your Dataset has an indices mapping (via Dataset.shuffle() for example),
        the speed can become 10x slower. This is because there is an extra step to get the row index
        to read using the indices mapping, and most importantly, you aren’t reading contiguous chunks
        of data anymore. To restore the speed, you’d need to rewrite the entire dataset on your disk
        again using Dataset.flatten_indices(), which removes the indices mapping. This may take a lot
        of time depending on the size of your dataset though:

        In this case, we recommend switching to an IterableDataset and leveraging its fast approximate
        shuffling method IterableDataset.shuffle(). It only shuffles the shards order and adds a shuffle
        buffer to your dataset, which keeps the speed of your dataset optimal. You can also reshuffle
        the dataset easily:

        If you want to shuffle your dataset or use it with a PyTorch DataLoader, we recommend generating a sharded IterableDataset:

        Copied
        my_iterable_dataset = my_dataset.to_iterable_dataset(num_shards=1024)
        my_iterable_dataset.n_shards  # 1024
        """
        dataset = self.filter(dataset, tokenizer)
        if self.preprocessor is not None and not self.cfg.process_online:
            if dataset.column_names is not None:
                # Q: what causes None? maybe loading text rather than parquet
                remove_columns = [
                    c for c in dataset.column_names if c not in (feature_names or [])
                ]  # shouldnt be necessary but is for plddts - bug?
            else:
                remove_columns = None
            dataset = dataset.map(
                self.map_fn,
                batched=self.cfg.batched_map,
                batch_size=self.cfg.map_batch_size,
                remove_columns=remove_columns,
                fn_kwargs={
                    "tokenizer": tokenizer,
                },
                features=Features(
                    **{f: TOKENIZED_FEATURE_TYPES[f] for f in feature_names}
                )
                if feature_names is not None
                else None,
            )
        else:
            raise NotImplementedError("Must implement online processing method")

        return dataset


class IterableHFProteinDataset(FileBasedHFProteinDataset):
    def __init__(
        self,
        name: str,
        cfg: HFProteinDatasetConfig,
        preprocessor: Optional[ProteinDocumentPreprocessor] = None,
        required_keys: Optional[List[str]] = None,
    ):
        super().__init__(name, preprocessor, required_keys=required_keys)
        self.cfg = cfg

    def get_data_files(self, data_dir: str, world_size: int):
        data_files = super().get_data_files(data_dir, world_size)
        if world_size is not None and len(data_files) < world_size:
            print(
                f"WARNING: Fewer data files ({len(data_files)}) than world size ({world_size})."
                " Repeating data files to match world size."
            )
            data_files = data_files * math.ceil(world_size / len(data_files))

        leftover_files = data_files[len(data_files) // world_size * world_size :]
        data_files = data_files[: (len(data_files) // world_size) * world_size]
        # worst case scenario is leftover_files=1: handle this or any other case by repeating maximal amount and slicing
        repeated_leftovers = leftover_files * world_size
        data_files = data_files + repeated_leftovers[:world_size]
        assert len(data_files) % world_size == 0, "Data files not evenly divisible"
        print(
            f"Ensuring even partition of shards across devices by upsampling to {len(data_files)} shards"
        )
        return data_files

    def process(
        self,
        dataset: Dataset,
        tokenizer: ProFamTokenizer,
        feature_names: Optional[List[str]] = None,
    ):
        """
        Process a dataset with a preprocessor.

        feature_names: names of features to keep
        """
        dataset = self.filter(dataset, tokenizer)
        if self.preprocessor is not None:
            # Q. how does batched map interact with interleave datasets?
            if dataset.column_names is not None:
                # Q: what causes None? maybe loading text rather than parquet
                remove_columns = [
                    c for c in dataset.column_names if c not in (feature_names or [])
                ]  # shouldnt be necessary but is for plddts - bug?
            else:
                remove_columns = None
            dataset = dataset.map(
                self.map_fn,
                batched=self.cfg.batched_map,
                batch_size=self.cfg.map_batch_size,
                remove_columns=remove_columns,
                fn_kwargs={
                    "tokenizer": tokenizer,
                    "feature_names": feature_names,
                },
            )
        return dataset


class SequenceDocumentDataset(IterableHFProteinDataset):
    def __init__(
        self,
        name: str,
        cfg: HFProteinDatasetConfig,
        preprocessor: Optional[ProteinDocumentPreprocessor] = None,
    ):
        super().__init__(
            name=name,
            cfg=cfg,
            preprocessor=preprocessor,
            required_keys=[cfg.sequence_col],
        )

    @staticmethod
    def build_document(
        example,
        # max_tokens: Optional[int] = None,
        # shuffle_proteins_in_document: bool = True,
        sequence_col: str = "sequences",
        identifier_col: str = "fam_id",
        max_sequences: Optional[int] = None,
        infer_representative_from_identifier: bool = False,
    ):
        sequences = example[sequence_col]
        # (max_tokens // 40)
        max_sequences_to_preprocess = (
            len(sequences) if max_sequences is None else max_sequences
        )
        sequences = sequences[:max_sequences_to_preprocess]

        return ProteinDocument(
            sequences=sequences,
            representative_accession=example[identifier_col]
            if infer_representative_from_identifier
            else None,
            original_size=len(sequences),
            identifier=example[identifier_col],
        )

    def _build_document(self, example):
        proteins = self.build_document(
            example,
            max_sequences=self.cfg.max_sequences_per_document,
            sequence_col=self.cfg.sequence_col,
            identifier_col=self.cfg.identifier_col,
            infer_representative_from_identifier=self.cfg.infer_representative_from_identifier,
        )
        proteins.identifier = self.name + "/" + proteins.identifier
        return proteins


class StructureDocumentDataset(IterableHFProteinDataset):
    def __init__(
        self,
        name: str,
        cfg: HFProteinDatasetConfig,
        preprocessor: Optional[ProteinDocumentPreprocessor] = None,
    ):
        super().__init__(
            name=name,
            cfg=cfg,
            preprocessor=preprocessor,
            required_keys=[cfg.sequence_col, cfg.structure_tokens_col]
            if cfg.structure_tokens_col is not None
            else [cfg.sequence_col],
        )

    @staticmethod
    def build_document(
        example,
        max_tokens: Optional[int] = None,
        shuffle: bool = True,
        max_sequences: Optional[int] = None,
        sample_uniformly_from_col: Optional[str] = None,
        structure_tokens_col: Optional[str] = None,
        sequence_col: str = "sequences",
        identifier_col: str = "fam_id",
        infer_representative_from_identifier: bool = False,
    ):
        # TODO: configure whether or not to use alignments, structure tokens col, etc.
        max_sequences_to_preprocess = (
            (max_tokens or 1e8) // 40 if max_sequences is None else max_sequences
        )
        if sample_uniformly_from_col is not None:
            assert shuffle
            sequence_ids = uniformly_sample_clusters(
                example[sequence_col],
                example[sample_uniformly_from_col],
                max_tokens - 3,
            )
        elif shuffle:
            sequence_ids = random_subsample(
                np.arange(len(example[sequence_col])),
                max_sequences_to_preprocess,
            )
        else:
            sequence_ids = np.arange(
                min(max_sequences_to_preprocess, len(example[sequence_col]))
            )
        sequences = [example[sequence_col][i] for i in sequence_ids]
        accessions = [example["accessions"][i] for i in sequence_ids]
        # we assume sequence processing and structure token processing are consistent.
        # later we will check that everything ends up the same length - which is important
        # because otherwise incorrect config could easily lead to misalignment
        if structure_tokens_col is not None:
            structure_tokens_iterator = example[structure_tokens_col]
            if structure_tokens_col == "msta_3di":
                # TODO: fix this; Hardcoded for now until we support aligning all representations
                structure_tokens = [
                    structure_tokens_iterator[i].replace("-", "").lower()
                    for i in sequence_ids
                ]
            else:
                structure_tokens = [
                    structure_tokens_iterator[i].lower() for i in sequence_ids
                ]
        else:
            # in fill missing values this gets set to mask, which in collate gets set to -100 in labels
            structure_tokens = None
        if "N" in example:
            coords, is_pdb = backbone_coords_from_example(
                example, sequence_col=sequence_col
            )
            coords = [coords[i] for i in sequence_ids]
            plddts = example["plddts"]
            plddts = [plddts[i] for i in sequence_ids]
        else:
            # TODO: support aligned coords, plddts
            coords = None
            plddts = None

        return ProteinDocument(
            sequences=sequences,
            accessions=accessions,
            plddts=plddts,
            backbone_coords=coords,
            structure_tokens=structure_tokens,
            representative_accession=example[identifier_col]
            if infer_representative_from_identifier
            else None,
            original_size=len(example["sequences"]),
            identifier=example[identifier_col],
        )

    def _build_document(
        self,
        example,
        max_tokens: Optional[int] = None,
        shuffle_proteins_in_document: bool = True,
    ):
        proteins = self.build_document(
            example,
            max_tokens=max_tokens,
            shuffle_proteins_in_document=shuffle_proteins_in_document,
            max_sequences=self.cfg.max_sequences_per_document,
            sample_uniformly_from_col=self.cfg.sample_uniformly_from_col,
            structure_tokens_col=self.cfg.structure_tokens_col,
            sequence_col=self.cfg.sequence_col,
            identifier_col=self.cfg.identifier_col,
            infer_representative_from_identifier=self.cfg.infer_representative_from_identifier,
        )
        proteins.identifier = self.name + "/" + proteins.identifier
        return proteins

    # TODO: write a test for this
    def filter(self, example, tokenizer: ProFamTokenizer = None):
        # Apply base class filter
        if not super().filter(example, tokenizer=tokenizer):
            return False

        # Check for structure tokens
        if (
            self.cfg.structure_tokens_col
            and example[self.cfg.structure_tokens_col] is None
        ):
            return False

        # Apply pLDDT filter if configured
        if self.minimum_mean_plddt is not None:
            if "plddts" not in example:
                return True

            mean_plddt = np.mean([np.mean(plddt) for plddt in example["plddts"]])
            return mean_plddt >= (self.cfg.minimum_mean_plddt or 0.0)

        return True
