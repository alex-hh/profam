import pyhmmer


class ProfileHMMEvaluator:
    """
    The parameters control 'reporting' and 'inclusion' thresholds, which determine attributes of hits.

    (I guess anything passing reporting threshold gets included in the hits?)

    http://eddylab.org/software/hmmer/Userguide.pdf
    """

    # TODO: write msa statistics evaluator via hmmalign
    # Any additional arguments passed to the hmmsearch function will be passed transparently to the Pipeline to be created. For instance, to run a hmmsearch using a bitscore cutoffs of 5 instead of the default E-value cutoff, use:
    def __init__(self, hmm_file, E=1000):
        with pyhmmer.plan7.HMMFile(hmm_file) as hmm_f:
            self.hmm = hmm_f.read()
        self.E = E  # E-value cutoff (large values are more permissive. we want to include everything.)
        self.alphabet = pyhmmer.easel.Alphabet.amino()

    def evaluate_samples(self, sequence_prompt, samples):
        # TODO: we want to not return ordered...
        sequences = [
            pyhmmer.easel.DigitalSequence(self.alphabet, name=f"seq{i}", sequence=seq)
            for i, seq in enumerate(samples)
        ]
        hits = pyhmmer.hmmsearch(self.hmm, sequences, E=self.E, incE=self.E)
        hits.sort(by="seqidx")
        for hit in hits.reported():
            print(hit.evalue, hit.score, hit.name)
        raise NotImplementedError("should be implemented on child class")


class HMMAlignmentStatisticsEvaluator:

    """First aligns generations to HMM, then computes statistics from alignment.

    Statistics are compared with those computed from a reference MSA.
    """

    def __init__(self, hmm_file, reference_msa, E=1000):
        with pyhmmer.plan7.HMMFile(hmm_file) as hmm_f:
            self.hmm = hmm_f.read()

    def evaluate_samples(self, sequence_prompt, samples):
        sequences = [
            pyhmmer.easel.DigitalSequence(self.alphabet, name=f"seq{i}", sequence=seq)
            for i, seq in enumerate(samples)
        ]
        msa = pyhmmer.hmmalign(self.hmm, sequences, trim=True, all_consensus_cols=True)
        # with...msa.write(f, format="a3m")
        sequences = [seq for _, seq in msa.alignment]
        # TODO: create MSA and compute statistics
