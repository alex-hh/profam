import pandas as pd
from typing import Dict
from src.data.objects import ProteinDocument
from src.pipelines.pipeline import GenerationsEvaluatorPipeline

class FunfamEvaluationPipeline(GenerationsEvaluatorPipeline):
    """
    A minimal pipeline that loads a single row (the first row) from a 
    parquet file and returns a ProteinDocument without any structural 
    features.
    """

    def __init__(
        self,
        parquet_path: str = "../data/funfams/s100_noali_parquets/train_val_test_split/val/val_000.parquet",
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        df = pd.read_parquet(parquet_path)
        # We'll store just the first row
        self._row = df.iloc[0]
        self._instance_id = self._row.fam_id

    def instance_ids(self):
        # There is only one instance in this minimal pipeline
        return [self._instance_id]

    def load_protein_document(self, instance_id: str) -> ProteinDocument:
        # Create a ProteinDocument from the parquet row
        if instance_id != self._instance_id:
            raise ValueError(f"Unknown instance_id: {instance_id}")
        return ProteinDocument(
            sequences=list(self._row.sequences),
            accessions=list(self._row.accessions),
            identifier=instance_id,
        )

    def get_instance_summary(self, instance_id: str) -> Dict[str, float]:
        # Minimal summary about the number of sequences
        if instance_id != self._instance_id:
            raise ValueError(f"Unknown instance_id: {instance_id}")
        return {"num_sequences": float(len(self._row.sequences))}