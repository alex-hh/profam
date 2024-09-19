"""We can implement both sequence recovery, requiring fixed length, and pairwise sequence identity."""
from typing import Optional

import numpy as np

from src.evaluators.base import SamplingEvaluator
from src.sequence.utils import sequence_identity


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
        representative = prompt.representative
        if representative.sequence.endswith("|"):
            representative = representative.slice_arrays(
                slice(0, len(representative) - 1)
            )
        backbone_coords_mask = representative.backbone_coords_mask.any(axis=(-1, -2))
        target_sequence = protein_document.representative.sequence
        unmasked_target_sequence = [
            aa for aa, m in zip(target_sequence, backbone_coords_mask) if m
        ]
        mismatched_lengths = 0
        recoveries = []
        unmasked_recoveries = []
        if self.verbose:
            print("Target")
            print(target_sequence)
        for i, seq in enumerate(samples):
            if self.verbose and i == 0:
                print(seq)
            if len(seq) == len(target_sequence):
                unmasked_seq = [aa for aa, m in zip(seq, backbone_coords_mask) if m]
                seq_id = sequence_identity(target_sequence, seq)
                unmasked_seq_id = sequence_identity(
                    unmasked_target_sequence, unmasked_seq
                )
                print("Seq ID", seq_id)
                recoveries.append(seq_id)
                unmasked_recoveries.append(unmasked_seq_id)
            else:
                mismatched_lengths += 1
        return {
            "mean_recovery": np.mean(recoveries),
            "mean_recovery_at_residues_with_coords": np.mean(unmasked_recoveries),
            "mismatched_lengths": mismatched_lengths / len(samples),
        }
