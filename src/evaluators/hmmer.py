import os
import subprocess
from typing import List

import numpy as np
import pyhmmer
from scipy.stats import pearsonr

from src.data.fasta import convert_sequence_with_positions
from src.data.objects import ProteinDocument
from src.data.utils import random_subsample, sample_to_max_tokens
from src.evaluators.alignment import MSANumeric, aa_letters_wgap
from src.evaluators.base import SamplingEvaluator


class BaseHMMEREvaluator(SamplingEvaluator):
    def load_hmm(self, protein_document: ProteinDocument):
        raise NotImplementedError("should be implemented on child class")


class PFAMHMMERMixin:
    """Given the full PFAM HMM database, use hmmfetch to extract specific models."""

    def __init__(
        self,
        *args,
        max_tokens: int = 8192,
        seed: int = 52,
        pfam_hmm_dir="../data/pfam/hmms",
        pfam_database="../data/pfam/Pfam-A.hmm",
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
        self.pfam_database = pfam_database

    def extract_hmm(self, identifier, hmm_file):
        subprocess.run("hmmfetch", self.pfam_database, identifier, stdout=hmm_file)

    def hmm_file_from_identifier(self, identifier: str):
        # other option would be to never build separate hmm files:
        # hmmfetch pfam_db.hmm HMM_NAME | hmmsearch - sequence_db.fasta > search_results.txt
        hmm_file = os.path.join(self.pfam_hmm_dir, f"{identifier}.hmm")
        if not os.path.isfile(hmm_file):
            self.extract_hmm(identifier, hmm_file)
        return hmm_file

    def load_hmm(self, protein_document: ProteinDocument):
        hmm_file = self.hmm_file_from_identifier(protein_document.identifier)
        with pyhmmer.plan7.HMMFile(hmm_file) as hmm_f:
            hmm = hmm_f.read()
        return hmm

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
        names = [f"seq{i}".encode() for i in range(len(samples))]
        sequences = [
            pyhmmer.easel.TextSequence(name=f"seq{i}".encode(), sequence=seq).digitize(self.alphabet)
            for i, seq in enumerate(samples)
        ]
        hits = next(pyhmmer.hmmsearch(hmm, sequences, E=self.E, incE=self.E))
        evalues = {}
        for hit in hits.reported:
            evalues[hit.name] = hit.evalue
        evalues = [evalues[name] for name in names]  # not actually necessary here since we take average but poss helpful
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
