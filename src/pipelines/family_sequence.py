from typing import Dict

import pandas as pd

from src.data.objects import ProteinDocument
from src.pipelines.pipeline import GenerationsEvaluatorPipeline


class FamilySequenceEvaluationPipeline(GenerationsEvaluatorPipeline):
    """
    A minimal pipeline that loads a single row (the first row) from a
    parquet file and returns a ProteinDocument without any structural
    features.
    """

    def __init__(
        self,
        parquet_path: str = "../data/funfams/s100_noali_parquets/train_val_test_split/val/val_000.parquet",
        first_row_only=True,
        max_tokens=8192,
        *args,
        **kwargs,
    ):
        super().__init__(max_tokens=max_tokens, *args, **kwargs)
        df = pd.read_parquet(parquet_path)
        if first_row_only:
            df = df.iloc[:1]
        self.df = df

    def instance_ids(self):
        return self.df.fam_id.values

    def load_protein_document(self, instance_id: str) -> ProteinDocument:
        matched_rows = self.df[self.df.fam_id==instance_id]
        assert len(matched_rows)==1
        instance_row = matched_rows.iloc[0]
        return ProteinDocument(
            sequences=list([s.replace("-", "") for s in instance_row.sequences]),
            accessions=list(instance_row.accessions),
            identifier=instance_id,
        )

    def get_instance_summary(self, instance_id: str) -> Dict[str, float]:
        instance_row = self.df[self.df.fam_id==instance_id].iloc[0]
        return {"num_sequences": float(len(instance_row.sequences))}
