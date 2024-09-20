from src.data.parquet import build_representative_df
from src.pipelines.mixins import ParquetGenerationsPipeline


# TODO: perhaps this should actually be a generic parquet representatives
# pipeline - if we manage to standardise data enough
class FoldseekRepresentativesPipeline(ParquetGenerationsPipeline):
    def __init__(
        self,
        *args,
        max_protein_length: int = 768,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.max_protein_length = max_protein_length
        assert self.evaluation_df is not None
        orig_len = len(self.evaluation_df)
        rep_df = build_representative_df(self.evaluation_df)
        rep_df = rep_df[rep_df["length"] <= self.max_protein_length]
        self.evaluation_df = self.evaluation_df[
            self.evaluation_df[self.instance_id_col].isin(
                rep_df[self.instance_id_col].values
            )
        ]
        print(
            f"Filtered {orig_len} to {len(self.evaluation_df)} proteins with length <={self.max_protein_length}"
        )
