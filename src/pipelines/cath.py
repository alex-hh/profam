"""CATH pipeline for backbones from the CATH splits used in the inverse folding literature."""
import time

import numpy as np

from src.data import cath
from src.data.objects import ProteinDocument
from src.pipelines.pipeline import GenerationsEvaluatorPipeline


class CATHEvaluationPipeline(GenerationsEvaluatorPipeline):
    def __init__(
        self, *args, split_name: str = "test", use_cath_43: bool = False, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.use_cath_43 = use_cath_43
        self.split_name = split_name
        t0 = time.time()
        if self.use_cath_43:
            self.targets = cath.load_cath43_coords(
                convert_to_protein_list=True, split_name=self.split_name
            )
        else:
            self.targets = cath.load_cath42_coords(
                convert_to_protein_list=True, split_name=self.split_name
            )
        self.targets = sorted(self.targets, key=lambda x: x.accession)
        self.instance_dicts = {target.accession: target for target in self.targets}
        t1 = time.time()
        version = "4.3" if self.use_cath_43 else "4.2"
        print(
            f"Loaded CATH {version} ({split_name} split) entries: ",
            len(self.instance_dicts),
            f"in {t1-t0:.2f}s",
        )

    def instance_ids(self):
        return [target.accession for target in self.targets]

    def load_protein_document(self, instance_id):
        if instance_id == "3j7y.K":
            instance_id = "3j7y.KK"  # disambiguate from 3j7y.k
        elif instance_id == "3j7yK":
            instance_id = "3j7yKK"
        entry = self.instance_dicts[instance_id]
        proteins = ProteinDocument.from_protein(
            entry, representative_accession=instance_id, identifier=instance_id
        )
        return proteins

    def get_instance_summary(self, instance_id):
        return {
            "target_length": len(self.instance_dicts[instance_id]["seq"]),
            "target_has_coords_frac": (
                ~np.isnan(self.instance_dicts[instance_id]["coords"]["CA"])
            ).mean(),
        }
