import glob
import os
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from datasets import Dataset, load_dataset
from omegaconf.listconfig import ListConfig

from src.data.objects import ProteinDocument
from src.data.preprocessing import ProteinDocumentPreprocessor
from src.data.utils import examples_to_list_of_dicts
from src.sequence.fasta import read_fasta_sequences
from src.constants import TENSOR_FEATURES
from src.utils.tokenizers import ProFamTokenizer


@dataclass
class ProteinDatasetConfig:
    """Config for file-based datasets.

    TODO: rename (what should we call file-based datasets?)
    """

    data_path_pattern: Optional[str] = None
    holdout_data_files: Optional[str] = None
    holdout_identifiers: Optional[List[str]] = None
    identifier_col: Optional[str] = "fam_id"
    data_path_file: Optional[str] = None
    minimum_sequences: Optional[int] = None
    file_repeats: int = 1
    file_type: str = "parquet"  # or "text", "json"
    shuffle: bool = True


def build_documents_helper(
    examples,
    document_builder,
    max_tokens: Optional[int] = None,
    shuffle: bool = True,
):
    """We assume that documents should be concatenated up to max_tokens.

    TODO: implement document-aware attention masking
    """
    example_dicts = examples_to_list_of_dicts(examples)
    proteins_list = [
        document_builder(example_dict, max_tokens=max_tokens, shuffle=shuffle)
        for example_dict in example_dicts
    ]
    document_lengths = [sum(proteins.sequence_lengths) for proteins in proteins_list]
    merged_documents = []
    current_document = None
    total_sequence_length = 0
    for proteins, length in zip(proteins_list, document_lengths):
        if current_document is None:
            current_document = proteins.clone()
        else:
            if sum(current_document.sequence_lengths) + length <= (max_tokens or 1e8):
                current_document = current_document.extend(proteins)
                total_sequence_length += sum(proteins.sequence_lengths)
            else:
                merged_documents.append(current_document)
                current_document = proteins.clone()
                total_sequence_length = sum(current_document.sequence_lengths)
    if current_document is not None:
        merged_documents.append(current_document)

    return merged_documents


def prepare_data_files(data_dir, cfg, world_size=1, stream: bool = True):
    if cfg.data_path_pattern is not None:
        # replace hf path resolution with manual glob, to allow repetition
        # https://github.com/huggingface/datasets/blob/98fdc9e78e6d057ca66e58a37f49d6618aab8130/src/datasets/data_files.py#L323
        data_files = glob.glob(os.path.join(data_dir, cfg.data_path_pattern))
        assert (
            len(data_files) > 0
        ), f"No files found for pattern {cfg.data_path_pattern} in {data_dir}"
    else:
        assert cfg.data_path_file is not None
        with open(os.path.join(data_dir, cfg.data_path_file), "r") as f:
            data_files = [
                os.path.join(data_dir, data_file) for data_file in f.read().splitlines()
            ]
            assert all([os.path.exists(f) for f in data_files])

    if cfg.holdout_data_files is not None:
        if isinstance(cfg.holdout_data_files, str):
            holdout_files = [cfg.holdout_data_files]
        else:
            assert isinstance(cfg.holdout_data_files, list) or isinstance(
                cfg.holdout_data_files, ListConfig
            ), f"holdout files is {type(cfg.holdout_data_files)} not list"
            holdout_files = cfg.holdout_data_files

        holdout_files = [os.path.join(data_dir, f) for f in holdout_files]
        assert all(
            [f in data_files for f in holdout_files]
        ), f"Not all holdout files {holdout_files} found in data files"

        all_files = len(data_files)
        data_files = [f for f in data_files if f not in holdout_files]
        print("Excluding", all_files - len(data_files), "holdout files")
        assert len(data_files) > 0, "No files left after holdout"

    assert isinstance(data_files, list)
    data_files = sorted(data_files) * cfg.file_repeats
    print(
        f"Loading dataset from {len(data_files)} files, "
        f"({cfg.file_repeats} repeats), "
        f"{os.path.join(data_dir, cfg.data_path_pattern)}"
    )

    if stream:
        # ensure no issues with ddp by skipping shards
        # should result in each worker using the same number of shards
        # https://github.com/huggingface/datasets/issues/6623
        # print("Slicing data files to ensure equal distribution across workers")
        if len(data_files) // world_size == 0:
            raise ValueError(
                f"Fewer data files ({len(data_files)}) than world size ({world_size}), difficult to handle in ddp"
            )
        else:
            data_files = data_files[: (len(data_files) // world_size) * world_size]
    return data_files


class BaseProteinDatasetBuilder:
    """For core hf related dataset building c.f. https://huggingface.co/docs/datasets/en/dataset_script.

    This class is different in that it is more focussed on preprocessing.
    It might be useful however to move towards the standardised splits.
    - although this is basically just directory-based
    """

    def __init__(
        self,
        name: str,
        preprocessor: Optional[ProteinDocumentPreprocessor] = None,
        required_keys: Optional[List[str]] = None,
    ):
        self.name = name
        self.preprocessor = preprocessor
        self.required_keys = required_keys

    def process(
        self,
        dataset: Dataset,
        tokenizer: ProFamTokenizer,
        max_tokens_per_example: Optional[int] = None,
        shuffle_proteins_in_document: bool = True,
    ):
        raise NotImplementedError("Must implement process method")

    def load(self, data_dir="data", world_size: int = 1, verbose: bool = False):
        raise NotImplementedError("Must implement load method")

    def _build_documents(
        self,
        examples,
        max_tokens: Optional[int] = None,
        shuffle: bool = True,
    ):
        return build_documents_helper(
            examples,
            self._build_document,
            max_tokens=max_tokens,
            shuffle=shuffle,
        )

    def _build_document(
        self, example, max_tokens: Optional[int] = None, shuffle: bool = True
    ):
        # private method has fixed signature; static methods can have variable signature
        raise NotImplementedError(
            "Must implement build_document method on child dataset builder"
        )

    def filter_fn(self, example, **kwargs):
        if self.required_keys is not None:
            for k in self.required_keys:
                if k not in example or not example[k]:
                    return False
        return True


class FileBasedProteinDatasetBuilder(BaseProteinDatasetBuilder):
    def __init__(
        self,
        name: str,
        cfg: ProteinDatasetConfig,
        preprocessor: Optional[ProteinDocumentPreprocessor] = None,
        required_keys: Optional[List[str]] = None,
    ):
        super().__init__(name, preprocessor, required_keys=required_keys)
        self.cfg = cfg

    def load(
        self,
        data_dir="data",
        world_size: int = 1,
        verbose: bool = False,
    ):
        # TODO: maybe handle world size elsewhere by shard slicing...
        data_files = prepare_data_files(
            data_dir, self.cfg, world_size=world_size, stream=True
        )
        print("data_files", data_files[:5])
        load_kwargs = {"sample_by": "document"} if self.cfg.file_type == "text" else {}
        dataset = load_dataset(
            self.cfg.file_type,
            data_files=data_files,
            split="train",  # just automatically assigns all files to train - get this 'split'
            streaming=True,
            **load_kwargs,
        )

        print("Dataset n shards", dataset.n_shards)
        if verbose:
            print("Verifying dataset content:")
            for i, item in enumerate(dataset.take(3)):
                print(f"  Item {i + 1}:")
                for key, value in item.items():
                    if isinstance(value, str):
                        value_to_print = value[:100]
                    elif isinstance(value, list):
                        # TODO: if its a list of lists we want to print only first few elements
                        if isinstance(value[0], list):
                            value_to_print = f"[{value[0][:10]},...]"
                        else:
                            value_to_print = (
                                f"{value[:3]}..." if len(value) > 3 else value
                            )
                    else:
                        value_to_print = value
                    print(f"    {key}: {value_to_print}")
                print()

        return dataset

    def filter(self, dataset, tokenizer: ProFamTokenizer):
        return dataset.filter(
            self.filter_fn,
            fn_kwargs={
                "min_sequences": self.cfg.minimum_sequences,
                "holdout_identifiers": self.cfg.holdout_identifiers,
                "tokenizer": tokenizer,
            },
        )

    def preprocess_examples(
        self,
        examples,
        tokenizer: ProFamTokenizer,
        max_tokens_per_example: Optional[int] = None,
        shuffle_proteins_in_document: bool = True,
    ):
        """Function to be mapped.

        a map is an instruction for converting an example to a new example.
        it should return a datapoint dict.

        a batched map is an instruction for converting a set of examples to a
        new set of examples (not necessarily of the same size). it should return a dict of lists,
        where the length of the lists determines the size of the new set of examples.
        """

        if self.batched_map:
            proteins_list = self._build_documents(
                examples,
                max_tokens=max_tokens_per_example,
                shuffle=shuffle_proteins_in_document,
            )
            examples = self.preprocessor.batched_preprocess_protein_data(
                proteins_list,
                tokenizer,
                max_tokens=max_tokens_per_example,
                shuffle=shuffle_proteins_in_document,
            )
            examples["ds_name"] = [self.name] * len(examples["input_ids"])
            if "identifier" in examples:
                examples["identifier"] = [
                    self.name + "/" + ident for ident in examples["identifier"]
                ]
            return examples
        else:
            # examples is a single row dict in this case
            proteins = self._build_document(
                examples,
                max_tokens=max_tokens_per_example,
                shuffle=shuffle_proteins_in_document,
            )
            examples = self.preprocessor.preprocess_protein_data(
                proteins,
                tokenizer,
                max_tokens=max_tokens_per_example,
                shuffle=shuffle_proteins_in_document,
            )
            examples["ds_name"] = self.name
            if "identifier" in examples:
                examples["identifier"] = self.name + "/" + examples["identifier"]
            return examples


class ProteinDatasetBuilder(FileBasedProteinDatasetBuilder):
    """File-based builder for a non-iterable Dataset.

    (Not necessarily memory-mapped, because can be used with in_memory=True)
    """

    def __init__(
        self,
        name: str,
        cfg: ProteinDatasetConfig,
        preprocessor: Optional[ProteinDocumentPreprocessor] = None,
        required_keys: Optional[List[str]] = None,
        process_online: bool = True,
        map_batch_size: int = 100,
        batched_map: bool = False,
    ):
        """process_online defaults to true for consistency with StreamedProteinDatasetBuilder."""
        super().__init__(name, preprocessor, required_keys=required_keys)
        self.cfg = cfg
        self.process_online = process_online
        self.map_batch_size = map_batch_size
        self.batched_map = batched_map

    def process(
        self,
        dataset: Dataset,
        tokenizer: ProFamTokenizer,
        max_tokens_per_example: Optional[int] = None,
        shuffle_proteins_in_document: bool = True,
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
        if self.preprocessor is not None and not self.process_online:
            if dataset.column_names is not None:
                # Q: what causes None? maybe loading text rather than parquet
                remove_columns = [
                    c for c in dataset.column_names if c not in (feature_names or [])
                ]  # shouldnt be necessary but is for plddts - bug?
            else:
                remove_columns = None
            dataset = dataset.map(
                self.preprocess_examples,
                batched=self.batched_map,
                batch_size=self.map_batch_size,
                remove_columns=remove_columns,
                fn_kwargs={
                    "tokenizer": tokenizer,
                    "max_tokens_per_example": max_tokens_per_example,
                    "shuffle_proteins_in_document": shuffle_proteins_in_document,
                },
            )
        else:
            raise NotImplementedError("Must implement online processing method")
        return dataset


# TODO: consider breaking out document building into a separate class
# for re-use in pipelines
class StreamedProteinDatasetBuilder(FileBasedProteinDatasetBuilder):
    def __init__(
        self,
        name: str,
        cfg: ProteinDatasetConfig,
        preprocessor: Optional[ProteinDocumentPreprocessor] = None,
        batched_map: bool = False,
        map_batch_size: int = 100,
        max_sequences_per_document: Optional[int] = None,
        required_keys: Optional[List[str]] = None,
    ):
        super().__init__(name, preprocessor, required_keys=required_keys)
        self.cfg = cfg
        self.batched_map = batched_map
        self.map_batch_size = map_batch_size
        self.max_sequences_per_document = max_sequences_per_document

    def process(
        self,
        dataset: Dataset,
        tokenizer: ProFamTokenizer,
        max_tokens_per_example: Optional[int] = None,
        shuffle_proteins_in_document: bool = True,
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
                self.preprocess_examples,
                batched=self.batched_map,
                batch_size=self.map_batch_size,
                remove_columns=remove_columns,
                fn_kwargs={
                    "tokenizer": tokenizer,
                    "max_tokens_per_example": max_tokens_per_example,
                    "shuffle_proteins_in_document": shuffle_proteins_in_document,
                },
            )
            # n.b. coords is returned as a list...

        return dataset


def subsample_fasta_lines(lines, n_lines, shuffle=True):
    start_ix = np.array([i for i, l in enumerate(lines) if l[0] == ">"])
    end_ix = start_ix[1:]
    end_ix = np.append(end_ix, len(lines))
    lines_per_seq = len(lines) // len(start_ix)
    n_samples = min(n_lines // lines_per_seq, len(start_ix))
    if shuffle:
        sample_indices = np.random.choice(len(start_ix), n_samples, replace=False)
    else:
        sample_indices = np.arange(n_samples)
    starts = start_ix[sample_indices]
    ends = end_ix[sample_indices]
    sampled_lines = []
    for start, end in zip(starts, ends):
        assert lines[end - 1][0] != ">"
        sampled_lines.extend(lines[start:end])
    return sampled_lines


# TODO: infer identifier from different column
class FastaProteinDatasetBuilder(StreamedProteinDatasetBuilder):
    def __init__(
        self,
        name: str,
        cfg: ProteinDatasetConfig,
        preprocessor: Optional[ProteinDocumentPreprocessor] = None,
        batched_map: bool = False,
        map_batch_size: int = 100,
    ):
        super().__init__(
            name=name,
            cfg=cfg,
            preprocessor=preprocessor,
            batched_map=batched_map,
            map_batch_size=map_batch_size,
            required_keys=["text"],
        )

    def filter_fn(
        self,
        example,
        min_sequences: Optional[int] = None,
        holdout_identifiers: Optional[List[str]] = None,
        tokenizer: ProFamTokenizer = None,
    ):
        super_filter = super().filter_fn(example)
        if super_filter:
            assert (
                holdout_identifiers is None
            ), "Holdout identifiers not supported for fasta"
            filter_num_seqs = len(example["text"].split("\n")) // 2 >= (
                min_sequences or 1
            )
            return filter_num_seqs
        return False

    @staticmethod
    def build_document(
        text,
        max_tokens: Optional[int] = None,
        shuffle: bool = True,
        max_sequences: Optional[int] = None,
        identifier: Optional[str] = None,
    ):
        lines = text.split("\n")
        if not len(lines[-1]):
            lines = lines[:-1]
        # rough upper bound: min 2 lines per seq, assume at least 10 tks per line
        max_fasta_lines_to_preprocess = (
            (max_tokens or 1e8) // 5 if max_sequences is None else max_sequences * 50
        )
        if len(lines) > max_fasta_lines_to_preprocess:
            lines = subsample_fasta_lines(
                lines,
                max_fasta_lines_to_preprocess,
                shuffle=shuffle,
            )

        sequences = [
            seq
            for seq in read_fasta_sequences(
                lines,
                # preserve original sequences before further preprocessing
                keep_gaps=True,
                keep_insertions=True,
                to_upper=False,
            )
        ]

        return ProteinDocument(
            sequences=sequences,
            original_size=len(lines) // 2,
            identifier=identifier,
        )  # upper bound estimate of number of sequences

    def _build_document(
        self, example, max_tokens: Optional[int] = None, shuffle: bool = True
    ):
        if isinstance(example, str):
            return self.build_document(
                example,
                max_tokens,
                shuffle,
                max_sequences=self.max_sequences_per_document,
            )
        else:
            return self.build_document(
                example["text"],
                max_tokens,
                shuffle,
                max_sequences=self.max_sequences_per_document,
                identifier=self.name + "/" + example[self.cfg.identifier_col]
                if self.cfg.identifier_col is not None
                else None,
            )
