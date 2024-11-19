import copy
import json
import os
from typing import List, Optional

import numpy as np
import torch
import tqdm
from datasets import Dataset, concatenate_datasets, load_dataset

from src import constants
from src.data.objects import Protein, ProteinDocument
from src.data.processors import ProteinDocumentPreprocessor
from src.data.tokenizers import ProFamTokenizer

from .base import BaseProteinDataset

CATH_43_JSONL_FILE = os.path.join(
    constants.PROFAM_DATA_DIR, "cath/cath43/chain_set.jsonl"
)
CATH_42_JSONL_FILE = os.path.join(
    constants.PROFAM_DATA_DIR, "cath/cath42/chain_set.jsonl"
)
CATH_43_SPLITS_FILE = os.path.join(constants.PROFAM_DATA_DIR, "cath/cath43/splits.json")
CATH_42_SPLITS_FILE = os.path.join(
    constants.PROFAM_DATA_DIR, "cath/cath42/chain_set_splits.json"
)


def cath_43_splits():
    with open(CATH_43_SPLITS_FILE) as f:
        return json.load(f)


def cath_42_splits():
    with open(CATH_42_SPLITS_FILE) as f:
        return json.load(f)


def coords_dict_to_list(coords_dict):
    coords_dict = copy.deepcopy(coords_dict)
    coords_dict["coords"] = list(
        zip(
            coords_dict["coords"]["N"],
            coords_dict["coords"]["CA"],
            coords_dict["coords"]["C"],
            coords_dict["coords"]["O"],
        )
    )
    return coords_dict


def coords_list_to_dict(coords_dict):
    coords_dict = copy.deepcopy(coords_dict)
    coords_dict["coords"] = {
        "N": [coords[0] for coords in coords_dict["coords"]],
        "CA": [coords[1] for coords in coords_dict["coords"]],
        "C": [coords[2] for coords in coords_dict["coords"]],
        "O": [coords[3] for coords in coords_dict["coords"]],
    }
    return coords_dict


def protein_from_coords_dict(coords_dict):
    backbone_coords = np.stack(
        [
            np.array(coords_dict["coords"]["N"]),
            np.array(coords_dict["coords"]["CA"]),
            np.array(coords_dict["coords"]["C"]),
            np.array(coords_dict["coords"]["O"]),
        ],
        axis=1,
    ).astype(np.float32)
    return Protein(
        sequence=coords_dict["seq"],
        accession=coords_dict["name"].replace(".", ""),
        backbone_coords=backbone_coords,
    )


def _load_coords(
    jsonl_file,
    convert_to_protein_list: bool = False,
    disable_tqdm: bool = False,
    split_ids: Optional[List[str]] = None,
):
    """Split-specific jsonl files should be created by running data_creation_scripts/create_cath_splits.py"""
    entries = []
    with open(jsonl_file) as f:
        lines = (
            f.readlines()
        )  # get a list rather than iterator to allow tqdm to know progress
        for line in tqdm.tqdm(lines, disable=disable_tqdm):
            coords_dict = json.loads(line)
            if split_ids is not None and coords_dict["name"] not in split_ids:
                continue
            if coords_dict["name"] == "3j7y.K":
                coords_dict["name"] = "3j7y.KK"  # disambiguate from 3j7y.k
            if convert_to_protein_list:
                entries.append(protein_from_coords_dict(coords_dict))
            else:
                entries.append(coords_dict)
    return entries


def load_cath43_coords(
    convert_to_protein_list: bool = False,
    disable_tqdm: bool = False,
    split_name: Optional[str] = None,
):
    split_ids = cath_43_splits()[split_name]
    return _load_coords(
        CATH_43_JSONL_FILE,
        convert_to_protein_list=convert_to_protein_list,
        disable_tqdm=disable_tqdm,
        split_ids=split_ids,
    )


def load_cath42_coords(
    convert_to_protein_list: bool = False,
    disable_tqdm: bool = False,
    split_name: Optional[str] = None,
):
    split_ids = cath_42_splits()[split_name]
    return _load_coords(
        CATH_42_JSONL_FILE,
        convert_to_protein_list=convert_to_protein_list,
        disable_tqdm=disable_tqdm,
        split_ids=split_ids,
    )


class ListDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        # Return the total number of items in the dataset
        return len(self.data)

    def __getitem__(self, idx):
        # Get the item at index `idx`
        return self.data[idx]


class BaseCATHDataset(BaseProteinDataset):
    def __init__(
        self,
        name: str,
        use_cath_43: bool = False,
        preprocessor: Optional[ProteinDocumentPreprocessor] = None,
        repeats: int = 1,
        document_token: str = "[RAW]",
        split_name: str = "validation",
    ):
        super().__init__(name)
        self.use_cath_43 = use_cath_43
        self.preprocessor = preprocessor
        if preprocessor is not None:
            self.preprocessor.single_protein_documents = True
            self.interleave_structure_sequence = (
                self.preprocessor.interleave_structure_sequence
            )
        else:
            self.interleave_structure_sequence = False
        self.document_token = document_token
        self.repeats = repeats
        self.split_name = split_name


class CATHTorchDataset(BaseCATHDataset):
    def load(self, data_dir: str, world_size: int = 1, verbose: bool = False):
        print("Loading CATH dataset")
        if self.use_cath_43:
            data = load_cath43_coords(
                convert_to_protein_list=True, split_name=self.split_name
            )
        else:
            data = load_cath42_coords(
                convert_to_protein_list=True, split_name=self.split_name
            )
        print("Done loading CATH dataset of length ", len(data))
        return ListDataset(data)

    def process(
        self,
        dataset: ListDataset,
        tokenizer: ProFamTokenizer,
        shuffle_proteins_in_document: bool = True,
        feature_names: Optional[List[str]] = None,
    ):
        processed_dataset = []
        print("Processing CATH dataset")
        for protein in tqdm.tqdm(dataset, disable=False):
            if (
                self.interleave_structure_sequence
                and len(protein.sequence)
                > (self.preprocessor.cfg.max_tokens_per_example // 2) - 3
            ):
                continue
            elif len(protein.sequence) > (
                self.preprocessor.cfg.max_tokens_per_example - 3
            ):
                continue
            proteins = ProteinDocument.from_proteins(
                [protein],
                representative_accession=protein.accession,
                identifier=protein.accession,
            )
            example = self.preprocessor.preprocess_protein_data(
                proteins,
                tokenizer,
                shuffle=False,
                return_tensors=True,
            )
            example["ds_name"] = self.name
            processed_dataset.append(example)
        return ListDataset(processed_dataset)


# TODO: inherit from FileBasedHFProteinDataset
class CATHHFDataset(BaseCATHDataset):
    def __init__(
        self,
        name: str,
        document_token: str = "[RAW]",
        split_name: str = "validation",
        use_cath_43: bool = False,
        num_proc: Optional[int] = None,
        preprocessor: Optional[ProteinDocumentPreprocessor] = None,
        repeats: int = 1,
        to_torch_dataset: bool = True,
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
        super().__init__(
            name=name,
            preprocessor=preprocessor,
            repeats=repeats,
            document_token=document_token,
            split_name=split_name,
            use_cath_43=use_cath_43,
        )
        self.num_proc = num_proc
        if self.num_proc is None:
            print(
                "Warning: num_proc is None for CATHDatasetBuilder, may be handled differently to other datasets"
            )
            print("Automatically setting num_proc to ", os.cpu_count())
            self.num_proc = os.cpu_count()
        self.to_torch_dataset = to_torch_dataset
        self.split_ids = (
            cath_43_splits()[split_name]
            if self.use_cath_43
            else cath_42_splits()[split_name]
        )
        self.split_ids = [pdb_id.replace(".", "") for pdb_id in self.split_ids]
        self.jsonl_file = CATH_43_JSONL_FILE if self.use_cath_43 else CATH_42_JSONL_FILE

    def load(self, data_dir: str, world_size: int = 1, verbose: bool = False):
        # TODO: we need to use correct types to make this faster.
        dataset = load_dataset(
            path="json",
            data_files=self.jsonl_file,
            split="train",
            keep_in_memory=True,
        )
        dataset = dataset.filter(
            lambda x: x["name"].replace(".", "") in self.split_ids,
            num_proc=self.num_proc,
        )
        return dataset

    @staticmethod
    def build_document(example):
        example["name"] = example["name"].replace(".", "")
        protein = protein_from_coords_dict(example)
        proteins = ProteinDocument.from_proteins(
            [protein],
            representative_accession=protein.accession,
            identifier=protein.accession,
        )
        return proteins

    def _build_document(self, example):
        return self.build_document(example)

    def process(
        self,
        dataset: Dataset,
        tokenizer: ProFamTokenizer,
        feature_names: Optional[List[str]] = None,
        return_format: str = "numpy",
        pack_to_max_tokens: Optional[int] = None,
    ):
        if pack_to_max_tokens is not None:
            raise NotImplementedError(
                "pack_to_max_tokens not implemented for CATHHFDataset"
            )
        # commented out due to issues. TODO: make minimal example and report
        # https://github.com/huggingface/datasets/issues/6319
        # https://discuss.huggingface.co/t/progress-bar-of-dataset-map-with-num-proc-1-hangs/64776/2
        def filter_fn(x):
            return len(x["seq"]) <= (
                self.preprocessor.cfg.max_tokens_per_example // 2
                if self.interleave_structure_sequence
                else self.preprocessor.cfg.max_tokens_per_example
            )

        dataset = dataset.filter(
            filter_fn, num_proc=self.num_proc, keep_in_memory=True
        ).map(
            self.preprocess_example,
            batched=False,
            num_proc=self.num_proc,
            batch_size=100,
            writer_batch_size=100,
            fn_kwargs={
                "tokenizer": tokenizer,
            },
            remove_columns=["seq", "name", "CATH"],
            keep_in_memory=True,
        )
        if self.repeats > 1:
            # TODO: test we still get shuffling - we should because of map style
            dataset = concatenate_datasets([dataset] * self.repeats)
        # https://discuss.huggingface.co/t/dataset-set-format/1961/4
        tensor_features = [
            c
            for c in constants.SEQUENCE_TENSOR_FEATURES
            + constants.STRUCTURE_TENSOR_FEATURES
            if c in dataset.column_names
        ]
        dataset.set_format(
            type=return_format,
            columns=tensor_features,  # may be unnecessary
            output_all_columns=True,  # also output string features
        )
        # https://github.com/huggingface/datasets/issues/5841
        if self.to_torch_dataset:
            print("Converting to torch dataset")
            processed_dataset = []
            # TODO: this is SUPER slow; maybe use iter with batch_size?
            for example in dataset:
                processed_dataset.append(example)
            dataset = ListDataset(processed_dataset)
            print("Done converting to torch dataset of length ", len(dataset))
        return dataset
        # processed_dataset = []
        # for example in dataset:
        #     processed_dataset.append(
        #         self.preprocess_example(example, tokenizer, max_tokens_per_example)
        #     )
        # return Dataset.from_list(processed_dataset)
