from typing import Dict, List, Optional

import numpy as np

from src.data.fasta import convert_sequence_with_positions
from src.data.objects import ProteinDocument


class SamplingEvaluator:
    def __init__(
        self,
        name: str,
        seed: int = 52,
        num_samples: Optional[int] = None,
    ):
        self.name = name
        self.seed = seed
        self.num_samples = num_samples

    def evaluate_samples(
        self,
        prompt: ProteinDocument,
        protein_document: ProteinDocument,
        samples: List[str],
        num_samples: Optional[int] = None,
        output_dir: Optional[str] = None,
    ) -> Dict[str, float]:
        if num_samples is not None and len(samples) != num_samples:
            assert len(samples) >= num_samples, f"Need at least {num_samples} samples"
            samples = samples[:num_samples]  # assuming samples are unsorted

        return self._evaluate_samples(
            prompt, protein_document, samples, output_dir=output_dir
        )

    def _evaluate_samples(
        self,
        prompt: ProteinDocument,
        protein_document: ProteinDocument,
        samples: List[str],
        output_dir: Optional[str] = None,
    ) -> Dict[str, float]:
        raise NotImplementedError("should be implemented on child class")

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

    def run_sampling(
        self,
        sampler,
        protein_document: ProteinDocument,
        num_samples: Optional[int] = None,
    ):
        num_samples = num_samples or self.num_samples
        assert num_samples is not None, "num_samples should be provided"
        if (
            self.num_samples is not None
            and num_samples is not None
            and num_samples != self.num_samples
        ):
            print(
                f"Warning: self.num_samples ({self.num_samples}) overriden by num_samples ({num_samples})"
            )
            assert (
                num_samples >= self.num_samples
            ), f"Expecting at least {self.num_samples} samples"

        samples, prompt = sampler.sample_seqs(protein_document, num_samples)
        return samples, prompt

    def __call__(
        self,
        sampler,
        protein_document: ProteinDocument,
        num_samples: int,
    ):
        samples, prompt = self.run_sampling(sampler, protein_document, num_samples)
        return self.evaluate_samples(prompt, samples)
