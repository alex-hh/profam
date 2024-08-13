from collections import defaultdict
from typing import Dict, List

import numpy as np

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

    def build_prompt(self, protein_document: ProteinDocument) -> List[str]:
        raise NotImplementedError("should be implemented on child class")

    def run_sampling(self, model, protein_document, num_samples):
        prompt = self.build_prompt(protein_document)
        samples = model.sample_seqs(prompt, num_samples)
        return samples

    def __call__(
        self,
        model: BaseFamilyLitModule,
        protein_document: ProteinDocument,
        num_samples: int,
    ):
        samples = self.run_sampling(model, protein_document, num_samples)
        return self.evaluate_samples(protein_document, samples)


class SamplingEvaluatorCallback:

    sequence_prompts: List[List[str]]

    def __init__(self, evaluator, num_samples):
        self.evaluator = evaluator
        self.num_samples = num_samples

    def on_train_epoch_end(self, trainer, model):
        if trainer.is_global_zero:
            # Q: how does logging work across ranks? if i log only from rank 0, what happens?
            all_metrics = defaultdict(list)
            for sequence_prompt in self.sequence_prompts:
                metrics = self.evaluator(model, sequence_prompt, self.num_samples)
                for key, value in metrics.items():
                    all_metrics[key].append(value)
            all_metrics = {"sampling/{k}": np.mean(v) for k, v in all_metrics.items()}
            # https://lightning.ai/docs/pytorch/stable/visualize/logging_advanced.html#rank-zero-only
            trainer.log_dict(all_metrics, on_epoch=True, rank_zero_only=True)
