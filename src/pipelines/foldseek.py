from typing import Dict, Optional

from src.data.builders.parquet import build_representative_df
from src.pipelines.mixins import ParquetStructureMixin


# TODO: perhaps this should actually be a generic parquet representatives
# pipeline - if we manage to standardise data enough
class FoldseekRepresentativesPipeline(ParquetStructureMixin):
    def __init__(
        self,
        *args,
        max_instances: Optional[int] = None,
        max_protein_length: Optional[int] = None,
        min_plddt: Optional[float] = None,
        **kwargs,
    ):
        # TODO: make this cooperate better with the base class
        super().__init__(*args, **kwargs)
        self.max_protein_length = max_protein_length
        self.min_plddt = min_plddt
        assert self.evaluation_df is not None
        if self.max_protein_length is not None or self.min_plddt is not None:
            orig_len = len(self.evaluation_df)
            rep_df = build_representative_df(
                self.evaluation_df, has_structure=True
            ).set_index(self.instance_id_col, drop=False)
            if self.max_protein_length is not None:
                rep_df = rep_df[rep_df["length"] <= self.max_protein_length]
            if self.min_plddt is not None:
                rep_df = rep_df[rep_df["mean_plddt"] >= self.min_plddt]

            self.evaluation_df = self.evaluation_df[
                self.evaluation_df[self.instance_id_col].isin(
                    rep_df[self.instance_id_col].values
                )
            ]
            self.evaluation_df["mean_plddt"] = rep_df["mean_plddt"]
            self.evaluation_df["length"] = rep_df["length"]
            self.evaluation_accessions = list(
                self.evaluation_df[self.instance_id_col].values
            )
            length_msg = (
                f"with length <={self.max_protein_length}"
                if self.max_protein_length is not None
                else ""
            )
            plddt_msg = (
                f"; with pLDDT >={self.min_plddt}" if self.min_plddt is not None else ""
            )
            print(
                f"Filtered {orig_len} to {len(self.evaluation_df)} proteins"
                f"{length_msg}{plddt_msg}"
            )
        if max_instances is not None:
            self.evaluation_df = self.evaluation_df.head(max_instances)
            self.evaluation_accessions = list(
                self.evaluation_df[self.instance_id_col].values
            )
            print(f"Filtered to {max_instances} proteins")

    def get_instance_summary(self, instance_id: str) -> Dict[str, float]:
        summary = super().get_instance_summary(instance_id)
        summary["target_plddt"] = self.evaluation_df.loc[instance_id, "mean_plddt"]
        summary["target_length"] = self.evaluation_df.loc[instance_id, "length"]
        return summary
