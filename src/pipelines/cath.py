"""CATH pipeline for backbones from the CATH splits used in the inverse folding literature."""
import copy
import json
import os
import time

import numpy as np

from src import constants
from src.data import cath
from src.pipelines.pipeline import GenerationsEvaluatorPipeline


class CATHEvaluationPipeline(GenerationsEvaluatorPipeline):
    def __init__(
        self, *args, split_name: str = "test", use_cath_43: bool = False, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.use_cath_43 = use_cath_43
        self.split_name = split_name
        jsonl_file = (
            cath.CATH_43_JSONL_FILE if self.use_cath_43 else cath.CATH_42_JSONL_FILE
        )
        t0 = time.time()
        splits = cath.cath_43_splits() if self.use_cath_43 else cath.cath_42_splits()
        self.instance_dicts = {
            entry["name"].replace(".", ""): entry
            for entry in cath.load_coords(jsonl_file)
            if entry["name"] in splits[split_name]
        }
        t1 = time.time()
        version = "4.3" if self.use_cath_43 else "4.2"
        print(
            f"Loaded CATH {version} ({split_name} split) entries: ",
            len(self.instance_dicts),
            f"in {t1-t0:.2f}s",
        )

    def instance_ids(self):
        return sorted(list(self.instance_dicts.keys()))

    def load_protein_document(self, instance_id):
        if instance_id == "3j7y.K":
            instance_id = "3j7y.KK"  # disambiguate from 3j7y.k
        elif instance_id == "3j7yK":
            instance_id = "3j7yKK"
        entry = self.instance_dicts[instance_id]
        protein = protein_from_coords_dict(entry)
        return ProteinDocument.from_proteins(
            [protein], representative_accession=protein.accession
        )

    def get_instance_summary(self, instance_id):
        return {
            "target_length": len(self.instance_dicts[instance_id]["seq"]),
            "target_has_coords_frac": (
                ~np.isnan(self.instance_dicts[instance_id]["coords"]["CA"])
            ).mean(),
        }
