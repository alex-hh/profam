import itertools
import os
import re
import subprocess
from typing import List, Optional

import numpy as np
import pyhmmer
from scipy.stats import pearsonr

from src.data.objects import ProteinDocument
from src.evaluators.base import SamplingEvaluator
from src.sequence.alignment import MSANumeric, aa_letters_wgap
from src.sequence.fasta import convert_sequence_with_positions


def hamming_distance(seq_a, seq_b, ignore_gaps=False):
    assert len(seq_a) == len(
        seq_b
    ), f"Sequences must be the same length but are {len(seq_a)} and {len(seq_b)}"
    if ignore_gaps:
        d = sum([a != b for (a, b) in zip(seq_a, seq_b) if a != "-" and b != "-"])
    else:
        d = sum([a != b for (a, b) in zip(seq_a, seq_b)])
    return d


class BaseHMMEREvaluator(SamplingEvaluator):
    def __init__(
        self,
        name: str,
        num_samples: Optional[int] = None,
        seed: int = 52,
    ):
        super().__init__(name, num_samples=num_samples)
        self.seed = seed

    def load_hmm(self, identifier: str):
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


class PFAMHMMERMixin:
    """Given the full PFAM HMM database, use hmmfetch to extract specific models."""

    def __init__(
        self,
        name,
        num_samples: Optional[int] = None,
        seed: int = 52,
        pfam_hmm_dir="../data/pfam/hmms",
        pfam_database="../data/pfam/Pfam-A.hmm",
        **kwargs,
    ):
        super().__init__(name, seed=seed, num_samples=num_samples, **kwargs)
        self.pfam_hmm_dir = pfam_hmm_dir
        self.pfam_database = pfam_database

    def extract_hmm(self, identifier, hmm_file):
        with open(hmm_file, "w") as hmm_file:
            subprocess.run(
                ["hmmfetch", self.pfam_database, identifier], stdout=hmm_file
            )

    def hmm_file_from_identifier(self, identifier: str):
        # other option would be to never build separate hmm files:
        # hmmfetch pfam_db.hmm HMM_NAME | hmmsearch - sequence_db.fasta > search_results.txt
        hmm_file = os.path.join(self.pfam_hmm_dir, f"{identifier}.hmm")
        if not os.path.isfile(hmm_file):
            self.extract_hmm(identifier, hmm_file)
        return hmm_file

    def load_hmm(self, identifier: str):
        hmm_file = self.hmm_file_from_identifier(identifier)
        with pyhmmer.plan7.HMMFile(hmm_file) as hmm_f:
            hmm = hmm_f.read()
        return hmm


class ProfileHMMEvaluator(BaseHMMEREvaluator):
    """
    The parameters control 'reporting' and 'inclusion' thresholds, which determine attributes of hits.

    (I guess anything passing reporting threshold gets included in the hits?)

    http://eddylab.org/software/hmmer/Userguide.pdf
    """

    # TODO: write msa statistics evaluator via hmmalign
    # Any additional arguments passed to the hmmsearch function will be passed transparently to the Pipeline to be created. For instance, to run a hmmsearch using a bitscore cutoffs of 5 instead of the default E-value cutoff, use:
    def __init__(
        self,
        name,
        E=1e8,
        num_reference: int = 1000,
        hit_threshold_for_metrics=0.001,
        **kwargs,
    ):
        super().__init__(name, **kwargs)
        self.E = E  # E-value cutoff (large values are more permissive. we want to include everything.)
        self.alphabet = pyhmmer.easel.Alphabet.amino()
        self.hit_threshold_for_metrics = hit_threshold_for_metrics
        self.num_reference = num_reference

    def _evaluate_samples(
        self,
        prompt: ProteinDocument,
        protein_document: ProteinDocument,
        samples: List[str],
        output_dir: Optional[str] = None,
    ):
        hmm = self.load_hmm(protein_document.identifier)
        # TODO: we want to not return ordered...
        names = [f"seq{i}".encode() for i in range(len(samples))]
        sequences = [
            pyhmmer.easel.TextSequence(name=f"seq{i}".encode(), sequence=seq).digitize(
                self.alphabet
            )
            for i, seq in enumerate(samples)
        ]
        reference_sequences = self.sample_document(
            protein_document,
            self.num_reference,
            keep_gaps=False,
            keep_insertions=True,
            to_upper=True,
        )
        reference_sequences = [
            pyhmmer.easel.TextSequence(
                name=f"ref_seq{i}".encode(), sequence=seq
            ).digitize(self.alphabet)
            for i, seq in enumerate(reference_sequences)
        ]
        hits = next(pyhmmer.hmmsearch(hmm, sequences, E=self.E, incE=self.E))
        if len(hits) > 0:
            evalues = {}
            for hit in hits.reported:
                evalues[hit.name] = hit.evalue

            evalues = [
                evalues[name] for name in names
            ]  # not actually necessary here since we take average but poss helpful
            evalue = np.mean(evalues)
            hit_percentage = (np.array(evalues) < self.hit_threshold_for_metrics).mean()
        else:
            evalue = self.E
            hit_percentage = 0.0
        reference_hits = next(
            pyhmmer.hmmsearch(hmm, reference_sequences, E=self.E, incE=self.E)
        )
        assert len(reference_hits) == len(reference_sequences)
        ref_evalue = np.mean([hit.evalue for hit in reference_hits.reported])
        return {
            "evalue": evalue,
            "ref_evalue": ref_evalue,
            "hit_percentage": hit_percentage,
        }


class PFAMProfileHMM(PFAMHMMERMixin, ProfileHMMEvaluator):
    pass


# TODO: write evaluator for pre-aligned sequences
class HMMAlignmentStatisticsEvaluator(BaseHMMEREvaluator):

    """First aligns generations to HMM, then computes statistics from alignment.

    Statistics are compared with those computed from a reference MSA.
    """

    def __init__(
        self,
        name,
        is_pre_aligned: bool = False,
        num_reference: int = 5000,
        seed: int = 52,
        **kwargs,
    ):
        super().__init__(name, seed=seed, **kwargs)
        self.is_pre_aligned = is_pre_aligned
        self.alphabet = pyhmmer.easel.Alphabet.amino()
        self.num_reference = num_reference
        self.seed = seed

    def _evaluate_samples(
        self,
        prompt: ProteinDocument,
        protein_document: ProteinDocument,
        samples: List[str],
        output_dir: Optional[str] = None,
    ):
        # TODO: add uniqueness, diversity hamming distance-based metrics
        if self.is_pre_aligned:
            sequences = samples
        else:
            hmm = self.load_hmm(protein_document.identifier)
            sequences = [
                pyhmmer.easel.TextSequence(
                    name=f"seq{i}".encode(), sequence=seq
                ).digitize(self.alphabet)
                for i, seq in enumerate(samples)
            ]
            msa = pyhmmer.hmmalign(hmm, sequences, trim=True, all_consensus_cols=True)
            # TODO: identify match cols...
            sequences = [re.sub(r"[a-z\.]", "", seq) for seq in msa.alignment]
            if any(["." in seq for seq in sequences]):
                raise ValueError("Insert present in alignment: todo - debug")
            print("Aligned sequences", [len(s) for s in sequences], sequences)

        reference_sequences = self.sample_document(
            protein_document,
            self.num_reference,
            keep_gaps=True,
            keep_insertions=False,
            to_upper=True,
        )

        diversity = [
            hamming_distance(seq_1, seq_2)
            for (seq_1, seq_2) in itertools.combinations(sequences, 2)
        ]
        distances = [
            [hamming_distance(seq, ref) for ref in reference_sequences]
            for seq in sequences
        ]
        minimum_distances = np.mean(
            [min(distances_to_refs) for distances_to_refs in distances]
        )
        diversity = np.mean(diversity)
        minimum_distances = np.mean(minimum_distances)

        sampled_msa = MSANumeric.from_sequences(sequences, aa_letters_wgap)
        num_gaps = np.mean([sum([aa == "-" for aa in seq]) for seq in sequences])
        num_gaps_ref = np.mean(
            [sum([aa == "-" for aa in seq]) for seq in reference_sequences]
        )

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
            "diversity": diversity,
            "minimum_distances": minimum_distances,
            "num_gaps": num_gaps,
            "num_gaps_ref": num_gaps_ref,
        }


class PFAMHMMAlignmentStatistics(PFAMHMMERMixin, HMMAlignmentStatisticsEvaluator):
    pass
