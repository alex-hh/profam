import functools
import os
from typing import Callable, Dict, List, Optional, Tuple

import pandas as pd

from src.data.objects import Protein, ProteinDocument
from src.data.proteingym import load_completions, load_msa_document
from src.pipelines.pipeline import CompletionScoringEvaluatorPipeline


def default_proteingym_msa(dms_id, meta_df, gym_data_dir, use_filtered_msa=False):
    msa_file = os.path.join(gym_data_dir, meta_df.loc[dms_id]["MSA_filename"])
    if use_filtered_msa:
        msa_file = msa_file.replace(".a2m", "_reformat_hhfilter.a3m")

    return load_msa_document(msa_file)


class ProteinGymPipeline(CompletionScoringEvaluatorPipeline):
    def __init__(
        self,
        *args,
        category="substitutions",
        gym_data_dir="../data/ProteinGym",
        dms_ids: Optional[List[str]] = None,
        msa_loading_fn: Optional[
            Callable
        ] = None,  # TODO: maybe create a whole abstraction for this...
        use_filtered_msa: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        assert category == "substitutions", "Only substitutions are supported for now"
        self.dms_ids = dms_ids
        self.gym_data_dir = gym_data_dir
        self.meta_df = pd.read_csv(
            os.path.join(gym_data_dir, "DMS_substitutions.csv")
        ).set_index("DMS_id")
        self.msa_loading_fn = msa_loading_fn or functools.partial(
            default_proteingym_msa, use_filtered_msa=use_filtered_msa
        )

    def instance_ids(self) -> List[str]:
        if self.dms_ids is None:
            return list(self.meta_df.index.values)
        else:
            return self.dms_ids

    def load_protein_document(self, instance_id: str):
        return self.msa_loading_fn(
            instance_id, meta_df=self.meta_df, gym_data_dir=self.gym_data_dir
        )

    def load_completions(
        self, instance_id: str
    ) -> Tuple[pd.DataFrame, ProteinDocument]:
        dms_file = os.path.join(
            self.gym_data_dir,
            "DMS_ProteinGym_substitutions",
            self.meta_df.loc[instance_id]["DMS_filename"],
        )
        return load_completions(dms_file, seed=None, max_mutated_sequences=None)

    def get_instance_summary(
        self, protein_document: ProteinDocument, mutation_df: pd.DataFrame
    ) -> Dict[str, float]:
        return {}


class ProteinGymInverseFoldingPipeline(ProteinGymPipeline):
    def load_protein_document(self, instance_id: str):
        uniprot_id = self.meta_df.loc[instance_id]["UniProt_ID"]
        pdb_file = os.path.join(
            self.gym_data_dir, "ProteinGym_AF2_structures", f"{uniprot_id}.pdb"
        )
        protein = Protein.from_pdb(pdb_file, bfactor_is_plddt=True)
        return ProteinDocument.from_proteins(
            [protein], representative_accession=uniprot_id
        )
