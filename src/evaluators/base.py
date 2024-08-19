from typing import Dict, List, Optional

import numpy as np

from src.data.fasta import convert_sequence_with_positions
from src.data.objects import ProteinDocument
from src.data.utils import random_subsample, sample_to_max_tokens

# class MultipleEvaluator:


class SamplingEvaluator:
    def __init__(self, name: str, seed: int = 52, num_samples: Optional[int] = None):
        self.name = name
        self.seed = seed
        self.num_samples = num_samples

    def evaluate_samples(
        self,
        protein_document: ProteinDocument,
        samples: List[str],
        num_samples: Optional[int] = None,
    ) -> Dict[str, float]:
        if num_samples is not None and len(samples) != num_samples:
            assert len(samples) >= num_samples, f"Need at least {num_samples} samples"
            samples = samples[:num_samples]  # assuming samples are unsorted
        return self._evaluate_samples(protein_document, samples)

    def _evaluate_samples(
        self, protein_document: ProteinDocument, samples: List[str]
    ) -> Dict[str, float]:
        raise NotImplementedError("should be implemented on child class")

    def build_prompt(self, protein_document: ProteinDocument):
        sequences = random_subsample(
            protein_document.sequences, self.max_tokens // 10, seed=self.seed
        )
        max_len = max([len(seq) for seq in sequences])
        sequences = []
        positions = []
        # TODO: subsample before convert sequence with positions.
        for sequence in protein_document.sequences:
            seq, pos, _ = convert_sequence_with_positions(
                sequence,
                keep_gaps=self.keep_gaps,
                keep_insertions=self.keep_insertions,
                to_upper=self.to_upper,
            )
            sequences.append(seq)
            positions.append(pos)
        sequences, positions = sample_to_max_tokens(
            sequences, positions, self.max_tokens - max_len, seed=self.seed
        )
        return sequences, positions

    def sample_document(
        self,
        protein_document: ProteinDocument,
        num_samples: int,
        keep_gaps: bool = False,
        keep_insertions: bool = True,
        to_upper: bool = True,
    ):
        rng = np.random.default_rng(self.seed)
        reference_sequence_indices = rng.choice(
            len(protein_document.sequences),
            min(num_samples, len(protein_document.sequences)),
            replace=False,
        )
        reference_sequences = [
            convert_sequence_with_positions(
                protein_document.sequences[i],
                keep_gaps=keep_gaps,
                keep_insertions=keep_insertions,
                to_upper=to_upper,
            )[0]
            for i in reference_sequence_indices
        ]
        return reference_sequences

    def build_inputs_from_prompt(self, prompt, num_samples: int):
        if isinstance(prompt, list) and isinstance(prompt[0], str):
            return {"sequence_prompt": prompt, "num_samples": num_samples}
        elif isinstance(prompt, tuple) and isinstance(prompt[0], list):
            return {
                "sequence_prompt": prompt[0],
                "position_indices": prompt[1],
                "num_samples": num_samples,
            }
        else:
            raise ValueError("Prompt should be a list of strings or a tuple of lists")

    # TODO: I think this should be a method on the model, that both the pipeline
    # and the evaluator can call. Different models might build prompt differently
    # based on their configuration - this should not be a function of evaluator configuration.
    # The only issue is that in some cases we want to share the same prompt when comparing
    # different models...this is a little tricky.
    def run_sampling(
        self, model, protein_document, num_samples: Optional[int] = None, **model_kwargs
    ):
        num_samples = num_samples or self.num_samples
        assert num_samples is not None, "num_samples should be provided"
        if (
            self.num_samples is not None
            and num_samples is not None
            and num_samples != self.num_samples
        ):
            print(f"Warning: self.num_samples ({self.num_samples}) overriden by num_samples ({num_samples})")

        prompt = self.build_prompt(protein_document)
        inputs = self.build_inputs_from_prompt(prompt, num_samples)
        samples = model.sample_seqs(
            **inputs, **model_kwargs
        )  # TODO: figure out how to configure model_kwargs
        return samples

    def __call__(
        self,
        model,
        protein_document: ProteinDocument,
        num_samples: int,
    ):
        samples = self.run_sampling(model, protein_document, num_samples)
        return self.evaluate_samples(protein_document, samples)
