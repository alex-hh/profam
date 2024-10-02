from typing import Dict, List, Optional

from src.data.objects import ProteinDocument


class SamplingEvaluator:
    def __init__(
        self,
        name: str,
        num_samples: Optional[int] = None,
    ):
        self.name = name
        self.num_samples = num_samples

    def evaluate_samples(
        self,
        prompt: ProteinDocument,
        protein_document: ProteinDocument,
        samples: List[str],
        num_samples: Optional[int] = None,
        output_dir: Optional[str] = None,
        device: Optional[str] = None,
    ) -> Dict[str, float]:
        if num_samples is not None and len(samples) != num_samples:
            assert len(samples) >= num_samples, f"Need at least {num_samples} samples"
            samples = samples[:num_samples]  # assuming samples are unsorted

        return self._evaluate_samples(
            prompt, protein_document, samples, output_dir=output_dir, device=device
        )

    def _evaluate_samples(
        self,
        prompt: ProteinDocument,
        protein_document: ProteinDocument,
        samples: List[str],
        output_dir: Optional[str] = None,
        device: Optional[str] = None,
    ) -> Dict[str, float]:
        raise NotImplementedError("should be implemented on child class")

    def __call__(
        self,
        sampler,
        protein_document: ProteinDocument,
        num_samples: int,
        device: Optional[str] = None,
    ):
        sampler.to(device)
        samples, prompt = self.run_sampling(sampler, protein_document, num_samples)
        return self.evaluate_samples(prompt, samples, device=device)
