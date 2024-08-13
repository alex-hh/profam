from typing import Dict, List

from src.data.objects import ProteinDocument
from src.models.base import BaseFamilyLitModule

# class MultipleEvaluator:


class SamplingEvaluator:
    def __init__(self, name: str):
        self.name = name

    def evaluate_samples(
        self, protein_document: ProteinDocument, samples: List[str]
    ) -> Dict[str, float]:
        raise NotImplementedError("should be implemented on child class")

    def build_prompt(self, protein_document: ProteinDocument):
        raise NotImplementedError("should be implemented on child class")

    def build_inputs_from_prompt(self, prompt, num_samples: int):
        if isinstance(prompt, list) and isinstance(prompt[0], str):
            return {"sequences": prompt, "num_samples": num_samples}
        elif isinstance(prompt, tuple) and isinstance(prompt[0], list):
            return {
                "input_ids": prompt[0],
                "position_indices": prompt[1],
                "num_samples": num_samples,
            }

    def run_sampling(self, model, protein_document, num_samples):
        prompt = self.build_prompt(protein_document)
        inputs = self.build_inputs_from_prompt(prompt, num_samples)
        samples = model.sample_seqs(**inputs)
        return samples

    def __call__(
        self,
        model: BaseFamilyLitModule,
        protein_document: ProteinDocument,
        num_samples: int,
    ):
        samples = self.run_sampling(model, protein_document, num_samples)
        return self.evaluate_samples(protein_document, samples)
