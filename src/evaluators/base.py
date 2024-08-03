import numpy as np
from collections import defaultdict
from typing import List, Optional


class SamplingEvaluator:

    def evaluate_samples(self, sequence_prompt, samples):
        raise NotImplementedError("should be implemented on child class")

    def __call__(self, model, sequence_prompt, num_samples):
        samples = model.sample_seqs(sequence_prompt, num_samples)
        return self.evaluate_samples(sequence_prompt, samples)


class SamplingEvaluatorCallback:

    sequence_prompts: List[str]

    def __init__(self, evaluator, num_samples):
        self.evaluator = evaluator
        self.num_samples = num_samples
    
    def on_train_epoch_end(self, trainer, model):
        # TODO: move esmfold to device?
        all_metrics = defaultdict(list)
        for sequence_prompt in self.sequence_prompts:
            metrics = self.evaluator(model, sequence_prompt, self.num_samples)
            for key, value in metrics.items():
                all_metrics[key].append(value)
        all_metrics = {"sampling/{k}": np.mean(v) for k, v in all_metrics.items()}
        trainer.log_dict(all_metrics, on_epoch=True)