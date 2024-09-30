from typing import Dict, Optional

from src.data.objects import ProteinDocument
from src.pipelines.mixins import ParquetMixin
from src.pipelines.pipeline import GenerationsEvaluatorPipeline


# TODO: try to standardise across datasets?
class PfamGenerationsPipeline(ParquetMixin, GenerationsEvaluatorPipeline):
    def __init__(
        self,
        *args,
        evaluation_parquet: str = None,
        evaluation_accessions_file: str = None,
        parquet_index: str = None,
        evaluation_accessions: list = None,
        **kwargs
    ):
        super().__init__(
            *args,
            instance_id_col="pfam_acc",
            evaluation_parquet=evaluation_parquet,
            evaluation_accessions_file=evaluation_accessions_file,
            evaluation_accessions=evaluation_accessions,
            parquet_index=parquet_index,
            **kwargs,
        )

    def instance_ids(self):
        return self.evaluation_accessions

    def get_instance_summary(
        self, instance_id: str, protein_document: Optional[ProteinDocument] = None
    ) -> Dict[str, float]:
        return {}
