import copy
import json
import os
import time
from typing import List, Optional

import numpy as np
from datasets import Dataset, load_dataset

from src import constants
from src.data import transforms
from src.data.datasets import BaseProteinDatasetBuilder
from src.data.objects import Protein, ProteinDocument
from src.data.preprocessing import PreprocessingConfig, ProteinDocumentPreprocessor
from src.utils.tokenizers import ProFamTokenizer

CATH_43_JSONL_FILE = os.path.join(
    constants.PROFAM_DATA_DIR, "cath/cath43/chain_set.jsonl"
)
CATH_42_JSONL_FILE = os.path.join(
    constants.PROFAM_DATA_DIR, "cath/cath42/chain_set.jsonl"
)


def cath_43_splits():
    return json.load(
        open(os.path.join(constants.PROFAM_DATA_DIR, "cath/cath43/splits.json"))
    )


def cath_42_splits():
    return json.load(
        open(os.path.join(constants.PROFAM_DATA_DIR, "cath/cath42/splits.json"))
    )


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


def load_coords(jsonl_file):
    """Split-specific jsonl files should be created by running data_creation_scripts/create_cath_splits.py"""
    entries = []
    with open(jsonl_file) as f:
        for line in f:
            coords_dict = json.loads(line)
            if coords_dict["name"] == "3j7y.K":
                coords_dict["name"] = "3j7y.KK"  # disambiguate from 3j7y.k
            entries.append(coords_dict)
    return entries


class CATHDatasetBuilder(BaseProteinDatasetBuilder):
    def __init__(
        self,
        name: str,
        document_token: str = "[RAW]",
        split_name: str = "validation",
        use_cath_43: bool = False,
        num_proc: Optional[int] = None,
        preprocessor: Optional[ProteinDocumentPreprocessor] = None,
    ):
        super().__init__(name)
        self.use_cath_43 = use_cath_43
        self.split_name = split_name
        self.split_ids = (
            cath_43_splits()[split_name]
            if use_cath_43
            else cath_42_splits()[split_name]
        )
        self.split_ids = [pdb_id.replace(".", "") for pdb_id in self.split_ids]
        self.jsonl_file = CATH_43_JSONL_FILE if use_cath_43 else CATH_42_JSONL_FILE
        self.num_proc = num_proc
        self.preprocessor = preprocessor
        if preprocessor is not None:
            self.preprocessor.single_protein_documents = True
            self.interleave_structure_sequence = (
                self.preprocessor.interleave_structure_sequence
            )
        else:
            self.interleave_structure_sequence = False
        self.document_token = document_token

    def load(self, data_dir: str, world_size: int = 1, verbose: bool = False):
        dataset = load_dataset(path="json", data_files=self.jsonl_file, split="train")
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

    def preprocess_example(
        self,
        example,
        tokenizer: ProFamTokenizer,
        max_tokens_per_example: Optional[int] = None,
    ):
        proteins = self.build_document(example)
        example = self.preprocessor.preprocess_protein_data(
            proteins,
            tokenizer,
            max_tokens=max_tokens_per_example,  # handles padding
            shuffle=False,
        )
        example["ds_name"] = self.name
        return example

    def process(
        self,
        dataset: Dataset,
        tokenizer: ProFamTokenizer,
        max_tokens_per_example: Optional[int] = None,
        shuffle_proteins_in_document: bool = True,
        feature_names: Optional[List[str]] = None,
    ):
        # commented out due to issues. TODO: make minimal example and report
        # https://github.com/huggingface/datasets/issues/6319
        # https://discuss.huggingface.co/t/progress-bar-of-dataset-map-with-num-proc-1-hangs/64776/2
        return dataset.filter(
            lambda x: len(x["seq"])
            <= (
                max_tokens_per_example // 2
                if self.interleave_structure_sequence
                else max_tokens_per_example
            )
        ).map(
            self.preprocess_example,
            batched=False,
            num_proc=self.num_proc,
            batch_size=100,
            writer_batch_size=100,
            fn_kwargs={
                "tokenizer": tokenizer,
                "max_tokens_per_example": max_tokens_per_example,
            },
            remove_columns=["seq", "name"],
        )
        # processed_dataset = []
        # for example in dataset:
        #     processed_dataset.append(
        #         self.preprocess_example(example, tokenizer, max_tokens_per_example)
        #     )
        # return Dataset.from_list(processed_dataset)
