from typing import Dict

from src.data.objects import ProteinDocument
from src.data.preprocessing import backbone_coords_from_example
from src.pipelines.mixins import ParquetMixin
from src.pipelines.pipeline import GenerationsEvaluatorPipeline


class FoldseekGenerationsPipeline(ParquetMixin, GenerationsEvaluatorPipeline):
    def __init__(
        self,
        *args,
        cluster_id_col: str = "fam_id",
        evaluation_parquet: str = None,
        evaluation_accessions_file: str = None,
        parquet_index: str = None,
        evaluation_accessions: list = None,
        **kwargs
    ):
        super().__init__(
            *args,
            instance_id_col=cluster_id_col,
            evaluation_parquet=evaluation_parquet,
            evaluation_accessions_file=evaluation_accessions_file,
            evaluation_accessions=evaluation_accessions,
            parquet_index=parquet_index,
            **kwargs,
        )

    def instance_ids(self):
        return self.evaluation_accessions

    def get_instance_summary(self, instance_id: str) -> Dict[str, float]:
        return {}

    def load_protein_document(self, instance_id: str):
        protein_example = self.get_protein_example(instance_id)
        # TODO: use preprocessor build_document? handles interleaving and transforms...
        return ProteinDocument(
            identifier=instance_id,
            sequences=protein_example["sequences"],
            accessions=protein_example["accessions"],
            backbone_coords=backbone_coords_from_example(protein_example),
            plddts=protein_example["plddts"],
            structure_tokens=[
                s.replace("-", "").lower() for s in protein_example["msta_3di"]
            ],
        )
