from typing import Dict, List, Optional

from src.data.objects import ProteinDocument
from src.evaluators.base import SamplingEvaluator
from src.tools.self_consistency.self_consistency import SelfConsistencyPDBEvaluator


class SelfConsistencyEvaluator(SamplingEvaluator):
    def __init__(
        self,
        name: str,
        pmpnn_dir: str,
        seed: int = 52,
        num_samples: Optional[int] = None,
        sequences_per_design: int = 1,
        sampling_temp: float = 0.1,
        calc_tm_score: bool = True,
        calc_rmsd: bool = True,
        evaluate_native_seq: bool = False,
    ):
        super().__init__(name, seed=seed, num_samples=num_samples)
        # this runs over saved pdb files containing designs
        self.evaluator = SelfConsistencyPDBEvaluator(
            device_name="cuda:0",
            seq_per_sample=sequences_per_design,
            pmpnn_dir=pmpnn_dir,
            sampling_temp=sampling_temp,
            calc_tm_score=calc_tm_score,
            calc_rmsd=calc_rmsd,
            evaluate_native_seq=evaluate_native_seq,
        )

    def _evaluate_samples(
        self,
        protein_document: ProteinDocument,
        samples: List[str],
        output_dir: Optional[str] = None,
    ) -> Dict[str, float]:
        # TODO: we need to save the pdb files for the designs
        # TODO: samples are not lists of strings here.
        # return self.evaluator.evaluate_samples(protein_document, samples, output_dir=output_dir)
        raise NotImplementedError("should be implemented on child class")
