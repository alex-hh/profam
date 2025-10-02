import os
from pathlib import Path
import numpy as np


# Ensure project root is on sys.path so we can import from `scripts/`
import sys  # noqa: E402

PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.evaluate_covariance import (  # noqa: E402
    compute_symbol_covariances,
    assess_correlation_preservation,
    ALPHABET_20,
)


def _write_aligned_fasta(fasta_path: str, seqs):
    with open(fasta_path, "w") as f:
        for i, s in enumerate(seqs):
            f.write(f">seq{i}\n{s}\n")


def test_compute_symbol_covariances_simple_two_state(tmp_path):
    # Construct an aligned MSA of length 2 with perfect co-variation between A and C
    # Sequences: AA, AA, CC, CC
    msa_path = os.path.join(tmp_path, "toy_aligned.fasta")
    _write_aligned_fasta(msa_path, [
        "AA",
        "AA",
        "CC",
        "CC",
    ])

    out = compute_symbol_covariances(msa_path)
    cov = out["covariance"]  # (L, L, S, S)
    counts_single = out["counts_single"]
    counts_pair = out["counts_pair"]
    support_single = out["support_single"]
    support_pair = out["support_pair"]

    # Basic shape checks
    assert cov.shape == (2, 2, len(ALPHABET_20), len(ALPHABET_20))

    idx_A = ALPHABET_20.index("A")
    idx_C = ALPHABET_20.index("C")

    # Support and counts
    assert support_single.tolist() == [4, 4]
    assert counts_single[0, idx_A] == 2 and counts_single[0, idx_C] == 2
    assert counts_single[1, idx_A] == 2 and counts_single[1, idx_C] == 2
    assert support_pair[0, 1] == 4
    assert counts_pair[0, 1, idx_A, idx_A] == 2
    assert counts_pair[0, 1, idx_C, idx_C] == 2

    # Expected covariances for perfect positive co-variation:
    # P(A)=0.5, P(C)=0.5, P(AA)=0.5, P(CC)=0.5
    # cov(A,A) = 0.5 - 0.25 = 0.25
    # cov(C,C) = 0.5 - 0.25 = 0.25
    # cov(A,C) = 0.0 - 0.25 = -0.25
    # cov(C,A) = 0.0 - 0.25 = -0.25
    assert np.isclose(cov[0, 1, idx_A, idx_A], 0.25)
    assert np.isclose(cov[0, 1, idx_C, idx_C], 0.25)
    assert np.isclose(cov[0, 1, idx_A, idx_C], -0.25)
    assert np.isclose(cov[0, 1, idx_C, idx_A], -0.25)


def test_assess_correlation_preservation_asymmetric_row_masking(tmp_path):
    # Build aligned natural and synthetic MSAs (length 2) with N=6 sequences each.
    # Position 0 (i) is always 'A' in both MSAs -> row filter at i=0 passes for 'A'.
    # Position 1 (j) uses 6 distinct residues in the natural MSA so that at i=1
    # no residue reaches min_support=2. As the row filter only applies to i (row),
    # pair (i=0, j=1) should have features, but (i=1, j=0) should not.
    nat_path = os.path.join(tmp_path, "nat_aligned.fasta")
    syn_path = os.path.join(tmp_path, "syn_aligned.fasta")

    # Use a set of distinct residues for position 1 so max count is 1
    diverse_residues = ["C", "D", "E", "F", "G", "H"]
    nat_seqs = ["A" + r for r in diverse_residues]
    syn_seqs = ["A" + r for r in reversed(diverse_residues)]

    _write_aligned_fasta(nat_path, nat_seqs)
    _write_aligned_fasta(syn_path, syn_seqs)

    results = assess_correlation_preservation(
        nat_path,
        [syn_path],
        min_support=2,
        include_diagonal=False,
        min_symbol_counts=None,  # disable column-wise max-count thresholds for this small MSA
    )

    res = results[syn_path]

    # With L=2, off-diagonal pairs are (0,1) and (1,0): denom_pairs=2
    # Row filter applies only to i. At i=0 we have 'A' counted 6 times -> features present for (0,1).
    # At i=1 no residue reaches count>=2 -> no features for (1,0).
    # Therefore coverage should be 1 / 2 = 0.5.
    assert np.isclose(float(res["coverage"]), 0.5)

    # pair_r may be NaN (all-zero covariances along the single kept row), but coverage confirms masking.
    pair_r = res["pair_r_matrix"]
    assert pair_r.shape == (2, 2)
    # Ensure (1,0) had no features (remains NaN)
    assert np.isnan(pair_r[1, 0])

if __name__ == "__main__":
    tmp_path = Path(__file__).resolve().parents[1] / "tmp"
    os.makedirs(tmp_path, exist_ok=True)
    test_assess_correlation_preservation_asymmetric_row_masking(tmp_path)
    test_compute_symbol_covariances_simple_two_state(tmp_path)
    