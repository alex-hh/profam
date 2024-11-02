from typing import List, Optional

import pandas as pd

from src.data.builders.hf_datasets import SequenceDocumentMixin, StructureDocumentMixin
from src.data.objects import ProteinDocument


class ParquetMixin:
    instance_id_col: str

    def __init__(
        self,
        *args,
        instance_id_col="fam_id",
        evaluation_parquet: Optional[str] = None,
        evaluation_accessions_file: Optional[str] = None,
        parquet_index: Optional[str] = None,
        evaluation_accessions: Optional[List[str]] = None,
        sequence_col: str = "sequences",
        max_instances: Optional[int] = None,
        **kwargs,
    ):
        """preprocessor: a bare preprocessor (no transform_fns), to build document from raw data."""
        super().__init__(*args, **kwargs)
        self.instance_id_col = instance_id_col
        self.sequence_col = sequence_col
        self.max_instances = max_instances
        if evaluation_parquet is not None:
            self.evaluation_df = pd.read_parquet(evaluation_parquet).set_index(
                self.instance_id_col, drop=False
            )
            if evaluation_accessions is not None:
                self.evaluation_df = self.evaluation_df.loc[evaluation_accessions]
            self.parquet_index = None
            self.evaluation_accessions = list(self.evaluation_df.index.values)
            assert (
                evaluation_accessions_file is None
            ), "Cannot specify both parquet and accessions file"
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

    def load_protein_document(self, instance_id: str) -> ProteinDocument:
        example = self.get_protein_example(instance_id)
        return SequenceDocumentMixin.build_document(
            example,
            sequence_col=self.sequence_col,
            max_tokens=None,
            max_sequences=None,
            sample_uniformly_from_col=None,
            identifier_col=self.instance_id_col,
            infer_representative_from_identifier=True,
        )

    def instance_ids(self):
        return self.evaluation_accessions


class ParquetStructureMixin(ParquetMixin):
    def __init__(
        self,
        *args,
        cluster_id_col: str = "fam_id",
        evaluation_parquet: str = None,
        evaluation_accessions_file: str = None,
        parquet_index: str = None,
        evaluation_accessions: list = None,
        sequence_col: str = "sequences",
        structure_tokens_col: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(
            *args,
            instance_id_col=cluster_id_col,
            evaluation_parquet=evaluation_parquet,
            evaluation_accessions_file=evaluation_accessions_file,
            evaluation_accessions=evaluation_accessions,
            parquet_index=parquet_index,
            sequence_col=sequence_col,
            **kwargs,
        )
        self.structure_tokens_col = structure_tokens_col

    def load_protein_document(self, instance_id: str) -> ProteinDocument:
        example = self.get_protein_example(instance_id)
        return StructureDocumentMixin.build_document(
            example,
            sequence_col=self.sequence_col,
            structure_tokens_col=self.structure_tokens_col,
            max_tokens=None,
            max_sequences=None,
            sample_uniformly_from_col=None,
            identifier_col=self.instance_id_col,
            infer_representative_from_identifier=True,
        )
