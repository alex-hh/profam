"""CATH pipeline for backbones from the CATH splits used in the inverse folding literature."""
import copy
import json
import os

import numpy as np

from src import constants
from src.data.objects import Protein, ProteinDocument
from src.pipelines.pipeline import GenerationsEvaluatorPipeline


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


def load_coords(jsonl_file):
    """Split-specific jsonl files should be created by running data_creation_scripts/create_cath_splits.py"""
    entries = []
    exclude_names = exclude_names or []
    with open(jsonl_file) as f:
        for line in f:
            coords_dict = json.loads(line)
            if coords_dict["name"] == "3j7y.K":
                coords_dict["name"] = "3j7y.KK"  # disambiguate from 3j7y.k
            if coords_dict["name"] not in exclude_names:
                entries.append(coords_dict)
    return entries


class CATHEvaluationPipeline(GenerationsEvaluatorPipeline):
    def __init__(
        self, *args, split_name: str = "test", use_cath_43: bool = False, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.use_cath_43 = use_cath_43
        self.split_name = split_name
        self.entries = load_coords(
            self.split_name, convert_to_list=True, exclude_names=self.exclude_names
        )
        self.instance_dicts = {entry["name"]: entry for entry in self.entries}

    def instance_ids(self):
        return [entry["name"].replace(".", "") for entry in self.entries]

    def load_protein_document(self, instance_id):
        if instance_id == "3j7y.K":
            instance_id = "3j7y.KK"  # disambiguate from 3j7y.k
        elif instance_id == "3j7yK":
            instance_id = "3j7yKK"
        entry = self.entries[instance_id]
        backbone_coords = np.stack(
            [
                np.array(entry["coords"]["N"]),
                np.array(entry["coords"]["CA"]),
                np.array(entry["coords"]["C"]),
                np.array(entry["coords"]["O"]),
            ],
            axis=1,
        ).astype(np.float32)
        sequence = entry["sequence"]
        return ProteinDocument.from_proteins(
            [
                Protein(
                    sequence=sequence,
                    accession=instance_id,
                    backbone_coords=backbone_coords,
                )
            ]
        )
