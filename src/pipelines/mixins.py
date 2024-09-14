from typing import Optional

import pandas as pd


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
            self.evaluation_accessions = list(self.evaluation_df.index.values)
            self.parquet_index = None
            if self.max_sequence_length is not None:
                self.evaluation_df["min_sequence_lengths"] = self.evaluation_df[
                    "sequences"
                ].apply(lambda x: min([len(seq) for seq in x]))
                self.evaluation_df = self.evaluation_df[
                    self.evaluation_df["min_sequence_lengths"]
                    <= self.max_sequence_length
                ]
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
