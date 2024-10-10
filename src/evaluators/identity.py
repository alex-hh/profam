"""We can implement both sequence recovery, requiring fixed length, and pairwise sequence identity."""
import itertools
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
        recoveries = []
        unmasked_recoveries = []
        for i, seq in enumerate(samples):
            if len(seq) == len(target_sequence):
                unmasked_seq = [aa for aa, m in zip(seq, backbone_coords_mask) if m]
                seq_id = sequence_identity(target_sequence, seq)
                if self.verbose and i == 0:
                    print(f"Target sequence and first sample (identity: {seq_id:.3f})")
                    print(target_sequence + "\n" + seq)
                unmasked_seq_id = sequence_identity(
                    unmasked_target_sequence, unmasked_seq
                )
                recoveries.append(seq_id)
                unmasked_recoveries.append(unmasked_seq_id)
            else:
                raise Exception("Sequence length mismatch")

        metrics = {
            "mean_recovery": np.mean(recoveries),
            "mean_recovery_at_residues_with_coords": np.mean(unmasked_recoveries),
        }
        if len(samples) > 1:
            pairwise_identities = [
                sequence_identity(seq1, seq2)
                for seq1, seq2 in itertools.combinations(samples, 2)
            ]
            metrics["pairwise_identities"] = np.mean(pairwise_identities)
        return metrics
