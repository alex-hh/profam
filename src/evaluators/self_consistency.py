from typing import Dict, List, Optional
from src.data.objects import ProteinDocument
from src.evaluators.base import SamplingEvaluator
from src.tools.self_consistency.self_consistency import SelfConsistencyPDBEvaluator


class SelfConsistencyEvaluator(SamplingEvaluator):
    def __init__(
        self,
        name: str,
        seed: int = 52,
        num_samples: Optional[int] = None,
    ):
        super().__init__(name, seed=seed, num_samples=num_samples)
        self.evaluator = SelfConsistencyPDBEvaluator()

    def _evaluate_samples(
        self,
        protein_document: ProteinDocument,
        samples: List[str],
        output_dir: Optional[str] = None,
    ) -> Dict[str, float]:
        # return self.evaluator.evaluate_samples(protein_document, samples, output_dir=output_dir)
        raise NotImplementedError("should be implemented on child class")
