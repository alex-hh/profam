from typing import Dict

import pandas as pd

from src.data.objects import ProteinDocument
from src.pipelines.pipeline import GenerationsEvaluatorPipeline


# TODO: try to standardise across datasets?
class PfamGenerationsPipeline(GenerationsEvaluatorPipeline):
    def __init__(
        self,
        *args,
        evaluation_parquet: str = None,
        evaluation_accessions_file: str = None,
        parquet_index: str = None,
        evaluation_accessions: list = None,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        if evaluation_parquet is not None:
            assert evaluation_accessions_file is None
            self.evaluation_df = pd.read_parquet(evaluation_parquet).set_index(
                "pfam_acc"
            )
            if evaluation_accessions is not None:
                self.evaluation_df = self.evaluation_df.loc[evaluation_accessions]
            self.evaluation_accessions = list(self.evaluation_df.index.values)
            self.parquet_index = None
        else:
            assert (
                evaluation_parquet is None
                and evaluation_accessions is None
                and evaluation_accessions_file is not None
            )
            evaluation_accessions = []
            with open(evaluation_accessions_file, "r") as f:
                for line in f:
                    evaluation_accessions.append(line.strip())
            self.evaluation_accessions = evaluation_accessions
            self.evaluation_df = None
            self.parquet_index = (
                pd.read_csv(parquet_index)
                .set_index("pfam_acc")["parquet_file"]
                .to_dict()
            )

    def instance_ids(self):
        return self.evaluation_accessions

    def get_instance_summary(self, instance_id: str) -> Dict[str, float]:
        return {}

    def get_protein_example(self, instance_id: str):
        if self.evaluation_df is not None:
            evaluation_df = self.evaluation_df
        else:
            parquet_file = self.parquet_index[instance_id]
            evaluation_df = pd.read_parquet(parquet_file)
        dict = evaluation_df.loc[instance_id].to_dict()
        return dict

    def load_protein_document(self, instance_id: str):
        protein_example = self.get_protein_example(instance_id)
        return ProteinDocument.from_fasta_str(instance_id, protein_example["text"])
