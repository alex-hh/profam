"""We can implement both sequence recovery, requiring fixed length, and pairwise sequence identity."""
from typing import Optional

from src.sequence.utils import sequence_identity
from src.evaluators.base import SamplingEvaluator


class SequenceIdentityEvaluator(SamplingEvaluator):
    pass


class SequenceRecoveryEvaluator(SamplingEvaluator):
    def __init__(
        self,
        name: str,
        verbose: bool = False,
        num_samples: Optional[int] = None,
    ):
        super().__init__(name, num_samples)
        self.verbose = verbose

    def _evaluate_samples(
        self,
        prompt,
        protein_document,
        samples,
        output_dir: Optional[str] = None,
        device: Optional[str] = None,
    ):
        # TODO: compute recovery in unmasked prompt regions.
        target_sequence = protein_document.representative.sequence
        mismatched_lengths = 0
        recoveries = []
        if self.verbose:
            print("Target")
            print(target_sequence)
        for i, seq in enumerate(samples):
            if self.verbose and i == 0:
                print(seq)
            if len(seq) == len(target_sequence):
                seq_id = sequence_identity(target_sequence, seq)
                print("Seq ID", seq_id)
                recoveries.append(seq_id)
            else:
                mismatched_lengths += 1
        return {
            "mean_recovery": sum(recoveries) / len(recoveries),
            "mismatched_lengths": mismatched_lengths / len(samples),
        }
