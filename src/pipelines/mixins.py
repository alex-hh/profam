from typing import Dict, Optional

import pandas as pd

from src.pipelines.pipeline import GenerationsEvaluatorPipeline


class ParquetMixin:
    instance_id_col: str

    def __init__(
        self,
        *args,
        instance_id_col="cluster_id",
        evaluation_parquet: str = None,
        evaluation_accessions_file: str = None,
        parquet_index: str = None,
        evaluation_accessions: list = None,
        max_instances: Optional[int] = None,
        max_sequence_length: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.instance_id_col = instance_id_col
        self.max_instances = max_instances
        self.max_sequence_length = max_sequence_length

        if evaluation_parquet is not None:
            assert evaluation_accessions_file is None
            self.evaluation_df = pd.read_parquet(evaluation_parquet).set_index(
                self.instance_id_col, drop=False
            )
            if evaluation_accessions is not None:
                self.evaluation_df = self.evaluation_df.loc[evaluation_accessions]
            self.parquet_index = None
            if self.max_sequence_length is not None:
                self.evaluation_df["min_sequence_lengths"] = self.evaluation_df[
                    "sequences"
                ].apply(lambda x: min([len(seq) for seq in x]))
                self.evaluation_df = self.evaluation_df[
                    self.evaluation_df["min_sequence_lengths"]
                    <= self.max_sequence_length
                ]
            self.evaluation_accessions = list(self.evaluation_df.index.values)
        else:
            assert self.max_sequence_length is None
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
                .set_index(self.instance_id_col)["parquet_file"]
                .to_dict()
            )
        if self.max_instances is not None:
            # Limit the number of instances - parquets often pre-shuffled
            self.evaluation_accessions = self.evaluation_accessions[
                : self.max_instances
            ]

    def get_protein_example(self, instance_id: str):
        if self.evaluation_df is not None:
            evaluation_df = self.evaluation_df
        else:
            parquet_file = self.parquet_index[instance_id]
            evaluation_df = pd.read_parquet(parquet_file)
        dict = evaluation_df.loc[instance_id].to_dict()
        return dict


class ParquetGenerationsPipeline(ParquetMixin, GenerationsEvaluatorPipeline):
    def __init__(
        self,
        *args,
        cluster_id_col: str = "fam_id",
        evaluation_parquet: str = None,
        evaluation_accessions_file: str = None,
        parquet_index: str = None,
        evaluation_accessions: list = None,
        **kwargs,
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
