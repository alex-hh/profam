from typing import Dict

import pandas as pd

from src.data.objects import ProteinDocument
from src.pipelines.pipeline import GenerationsEvaluatorPipeline


class UnconditionalSequenceEvaluationPipeline(GenerationsEvaluatorPipeline):
    """
    Unconditionally sample single sequences.
    """

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

    def instance_ids(self):
        return ["unconditional_sequence"]

    def load_protein_document(self, instance_id: str) -> ProteinDocument:
        return ProteinDocument(
            sequences=[''],
            accessions=self.instance_ids(),
            identifier=instance_id,
        )

    def get_instance_summary(self, instance_id: str) -> Dict[str, float]:
        return {"num_sequences": 1}