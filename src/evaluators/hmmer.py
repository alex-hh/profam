import os
from typing import List

import numpy as np
import pyhmmer
from scipy.stats import pearsonr

from src.data.fasta import convert_sequence_with_positions
from src.data.objects import ProteinDocument
from src.evaluators.alignment import MSANumeric, aa_letters_wgap
from src.evaluators.base import SamplingEvaluator


class BaseHMMEREvaluator(SamplingEvaluator):
    def load_hmm(self, protein_document: ProteinDocument):
        raise NotImplementedError("should be implemented on child class")


class PFAMHMMERMixin:
    def __init__(
        self,
        *args,
        max_tokens: int = 8192,
        seed: int = 52,
        pfam_hmm_dir="../data/pfam_hmms",
        keep_gaps=False,
        keep_insertions=True,
        to_upper=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.pfam_hmm_dir = pfam_hmm_dir
        self.seed = seed
        self.keep_gaps = keep_gaps
        self.keep_insertions = keep_insertions
        self.to_upper = to_upper
        self.max_tokens = max_tokens

    def hmm_file_from_identifier(self, identifier: str):
        return os.path.join(self.pfam_hmm_dir, f"{identifier}.hmm")

    def load_hmm(self, protein_document: ProteinDocument):
        hmm_file = self.hmm_file_from_identifier(protein_document.identifier)
        with pyhmmer.plan7.HMMFile(hmm_file) as hmm_f:
            hmm = hmm_f.read()
        return hmm

    def build_prompt(self, protein_document: ProteinDocument):
        sequences = []
        positions = []
        # TODO: subsample before convert sequence with positions.
        for sequence in protein_document.sequences:
            seq, pos = convert_sequence_with_positions(
                sequence,
                keep_gaps=self.keep_gaps,
                keep_insertions=self.keep_insertions,
                to_upper=self.to_upper,
            )
            sequences.append(seq)
            positions.append(pos)

        return sequences, positions


class ProfileHMMEvaluator(BaseHMMEREvaluator):
    """
    The parameters control 'reporting' and 'inclusion' thresholds, which determine attributes of hits.

    (I guess anything passing reporting threshold gets included in the hits?)

    http://eddylab.org/software/hmmer/Userguide.pdf
    """

    # TODO: write msa statistics evaluator via hmmalign
    # Any additional arguments passed to the hmmsearch function will be passed transparently to the Pipeline to be created. For instance, to run a hmmsearch using a bitscore cutoffs of 5 instead of the default E-value cutoff, use:
    def __init__(self, E=1000, hit_threshold_for_metrics=0.001):
        self.E = E  # E-value cutoff (large values are more permissive. we want to include everything.)
        self.alphabet = pyhmmer.easel.Alphabet.amino()
        self.hit_threshold_for_metrics = hit_threshold_for_metrics

    def evaluate_samples(self, protein_document: ProteinDocument, samples: List[str]):
        hmm = self.load_hmm(protein_document)
        # TODO: we want to not return ordered...
        sequences = [
            pyhmmer.easel.DigitalSequence(self.alphabet, name=f"seq{i}", sequence=seq)
            for i, seq in enumerate(samples)
        ]
        hits = pyhmmer.hmmsearch(hmm, sequences, E=self.E, incE=self.E)
        hits.sort(by="seqidx")
        evalues = []
        for hit in hits.reported():
            print(hit.evalue, hit.score, hit.name)
            evalues.append(hit.evalue)
        return {
            "evalue": np.mean(evalues),
            "hit_percentage": (
                np.array(evalues) < self.hit_threshold_for_metrics
            ).mean(),
        }


class PFAMProfileHMM(PFAMHMMERMixin, ProfileHMMEvaluator):
    pass


class HMMAlignmentStatisticsEvaluator(BaseHMMEREvaluator):

    """First aligns generations to HMM, then computes statistics from alignment.

    Statistics are compared with those computed from a reference MSA.
    """

    def evaluate_samples(self, protein_document: ProteinDocument, samples: List[str]):
        sequences = [
            pyhmmer.easel.DigitalSequence(self.alphabet, name=f"seq{i}", sequence=seq)
            for i, seq in enumerate(samples)
        ]
        msa = pyhmmer.hmmalign(self.hmm, sequences, trim=True, all_consensus_cols=True)
        sequences = [seq for _, seq in msa.alignment]
        sampled_msa = MSANumeric.from_sequences(sequences, aa_letters_wgap)
        reference_sequences = [
            convert_sequence_with_positions(
                seq, keep_gaps=True, keep_insertions=False, to_upper=False
            )[0]
            for seq in protein_document.sequences
        ]
        reference_msa = MSANumeric.from_sequences(reference_sequences, aa_letters_wgap)
        sampled_f = sampled_msa.frequencies().flatten()
        sampled_fij = sampled_msa.pair_frequencies().flatten()
        sampled_cov = sampled_msa.covariances().flatten()
        ref_f = reference_msa.frequencies().flatten()
        ref_fij = reference_msa.pair_frequencies().flatten()
        ref_cov = reference_msa.covariances().flatten()
        # compute correlations
        f_correlation = pearsonr(sampled_f, ref_f)[0]
        fij_correlation = pearsonr(sampled_fij, ref_fij)[0]
        cov_correlation = pearsonr(sampled_cov, ref_cov)[0]
        return {
            "frequency_pearson": f_correlation,
            "pair_frequency_pearson": fij_correlation,
            "covariance_pearson": cov_correlation,
        }


class PFAMHMMAlignmentStatistics(PFAMHMMERMixin, HMMAlignmentStatisticsEvaluator):
    pass
