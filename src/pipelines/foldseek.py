from typing import Dict

import pandas as pd

from src.data.objects import ProteinDocument
from src.pipelines.mixins import ParquetMixin
from src.pipelines.pipeline import GenerationsEvaluatorPipeline


class FoldseekGenerationsPipeline(ParquetMixin, GenerationsEvaluatorPipeline):
    def __init__(
        self,
        *args,
        evaluation_parquet: str = None,
        evaluation_accessions_file: str = None,
        parquet_index: str = None,
        evaluation_accessions: list = None,
        load_structure: bool = False,
        **kwargs
    ):
        super().__init__(
            *args,
            instance_id_col="cluster_id",
            evaluation_parquet=evaluation_parquet,
            evaluation_accessions_file=evaluation_accessions_file,
            evaluation_accessions=evaluation_accessions,
            parquet_index=parquet_index,
            **kwargs,
        )
        self.load_structure = load_structure
    
    def instance_ids(self):
        return self.evaluation_accessions

    def get_instance_summary(self, instance_id: str) -> Dict[str, float]:
        return {}

    def load_protein_document(self, instance_id: str):
        protein_example = self.get_protein_example(instance_id)
        if self.load_structure:
            raise NotImplementedError()
        return ProteinDocument(
            identifier=instance_id,
            sequences=protein_example["sequences"],
            accessions=protein_example.get("accessions", None),
        )
