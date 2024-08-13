import numpy as np
import pyhmmer
from scipy.stats import pearsonr

from src.evaluators.alignment import MSANumeric, aa_letters_wgap
from src.evaluators.base import SamplingEvaluator


class ProfileHMMEvaluator(SamplingEvaluator):
    """
    The parameters control 'reporting' and 'inclusion' thresholds, which determine attributes of hits.

    (I guess anything passing reporting threshold gets included in the hits?)

    http://eddylab.org/software/hmmer/Userguide.pdf
    """

    # TODO: write msa statistics evaluator via hmmalign
    # Any additional arguments passed to the hmmsearch function will be passed transparently to the Pipeline to be created. For instance, to run a hmmsearch using a bitscore cutoffs of 5 instead of the default E-value cutoff, use:
    def __init__(self, hmm_file, E=1000, hit_threshold_for_metrics=0.001):
        with pyhmmer.plan7.HMMFile(hmm_file) as hmm_f:
            self.hmm = hmm_f.read()
        self.E = E  # E-value cutoff (large values are more permissive. we want to include everything.)
        self.alphabet = pyhmmer.easel.Alphabet.amino()
        self.hit_threshold_for_metrics = hit_threshold_for_metrics

    def evaluate_samples(self, sequence_prompt, samples):
        # TODO: we want to not return ordered...
        sequences = [
            pyhmmer.easel.DigitalSequence(self.alphabet, name=f"seq{i}", sequence=seq)
            for i, seq in enumerate(samples)
        ]
        hits = pyhmmer.hmmsearch(self.hmm, sequences, E=self.E, incE=self.E)
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


class HMMAlignmentStatisticsEvaluator(SamplingEvaluator):

    """First aligns generations to HMM, then computes statistics from alignment.

    Statistics are compared with those computed from a reference MSA.
    """

    def __init__(self, hmm_file, reference_msa):
        with pyhmmer.plan7.HMMFile(hmm_file) as hmm_f:
            self.hmm = hmm_f.read()
        self.reference_msa = reference_msa

    def evaluate_samples(self, sequence_prompt, samples):
        sequences = [
            pyhmmer.easel.DigitalSequence(self.alphabet, name=f"seq{i}", sequence=seq)
            for i, seq in enumerate(samples)
        ]
        msa = pyhmmer.hmmalign(self.hmm, sequences, trim=True, all_consensus_cols=True)
        # with...msa.write(f, format="a3m")
        sequences = [seq for _, seq in msa.alignment]
        sampled_msa = MSANumeric.from_sequences(sequences, aa_letters_wgap)
        reference_msa = MSANumeric.from_sequences(sequences, aa_letters_wgap)
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
