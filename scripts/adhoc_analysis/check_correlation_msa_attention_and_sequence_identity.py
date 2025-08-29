"""
for each fasta file calculate the the percentage identity and coverage
of each sequence in the MSA against the query sequence (first row)
Extract the MSA attention score from the identifier line in the fasta file
from scipy.stats import spearmanr

for each file make a scatter plot with sequence identity on the x_axis and
msa attention (sequence weight) on the y_axis, color should reflect the coverage

First sequence is the query sequence: exclude it from the plot and spearman calculation

finally create a histogram of all the spearman values
"""

import argparse
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
from src.sequence import fasta
from src.data.builders.proteingym import extract_sequence_weights_from_seq_ids
from scipy.stats import spearmanr


msa_files = glob.glob("../data/ProteinGym/msa_pairformer_ranked_msas/*.fasta")
print(f"Found {len(msa_files)} MSA files")

ranked_correlations = []

plt_save_dir = "../msa_pariformer_attention_identity"

def get_identity_and_coverage(seqs):
    """
    Compute percent identity and coverage of each sequence vs the first (query).

    - Identity: matches / (# non-gap positions in query)
    - Coverage: non-gap positions in sequence over non-gap positions in query

    If sequences are not aligned, fall back to comparing up to min length.
    Returns arrays including the query sequence (index 0 will be 1.0, 1.0).
    """
    if len(seqs) == 0:
        return np.array([]), np.array([])
    ref = seqs[0]

    def is_gap(ch: str) -> bool:
        return ch in {"-", "."}

    aligned_like = all(len(s) == len(ref) for s in seqs)
    has_gap = ("-" in ref or "." in ref) or any(("-" in s or "." in s) for s in seqs)

    if aligned_like and has_gap:
        ref_positions = [i for i, ch in enumerate(ref) if not is_gap(ch)]
        denom = len(ref_positions) if len(ref_positions) > 0 else 1
        identities = []
        coverages = []
        for s in seqs:
            covered = 0
            matches = 0
            for i in ref_positions:
                if i < len(s) and not is_gap(s[i]):
                    covered += 1
                    if s[i] == ref[i]:
                        matches += 1
            identities.append(matches / denom)
            coverages.append(covered / denom)
        return np.asarray(identities, dtype=float), np.asarray(coverages, dtype=float)
    else:
        identities = []
        coverages = []
        denom = len(ref) if len(ref) > 0 else 1
        for s in seqs:
            L = min(len(s), len(ref))
            matches = sum(1 for i in range(L) if s[i] == ref[i])
            identities.append(matches / denom)
            coverages.append(L / denom)
        return np.asarray(identities, dtype=float), np.asarray(coverages, dtype=float)

os.makedirs(plt_save_dir, exist_ok=True)

for msa_file in msa_files:
    seq_ids, seqs = fasta.read_fasta(
        msa_file,
        keep_insertions=False,
        to_upper=True,
        keep_gaps=True,
    )
    identities, coverages = get_identity_and_coverage(seqs)
    try:
        sequence_weights = extract_sequence_weights_from_seq_ids(seq_ids)
    except Exception:
        sequence_weights = np.ones(len(seqs), dtype=float)

    # Exclude the first (query) sequence
    if len(seqs) <= 1:
        continue
    x_identity = identities[1:] * 100.0
    y_weight = np.asarray(sequence_weights[1:], dtype=float)
    z_coverage = coverages[1:] * 100.0

    # Spearman correlation
    mask = np.isfinite(x_identity) & np.isfinite(y_weight)
    if mask.sum() >= 2:
        coef, _ = spearmanr(x_identity[mask], y_weight[mask])
    else:
        coef = np.nan
    ranked_correlations.append(coef)

    # Scatter plot colored by coverage
    fig, ax = plt.subplots(figsize=(6, 4))
    sc = ax.scatter(x_identity, y_weight, c=z_coverage, cmap="viridis", s=8, alpha=0.8)
    ax.set_xlabel("Sequence identity to query (%)")
    ax.set_ylabel("MSA attention (sequence weight)")
    title = os.path.basename(msa_file)
    if isinstance(coef, float) and np.isfinite(coef):
        ax.set_title(f"{title}  Spearman={coef:.3f}")
    else:
        ax.set_title(f"{title}  Spearman=nan")
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Coverage (%)")
    plt.tight_layout()
    out_file = os.path.join(
        plt_save_dir, os.path.basename(msa_file).replace(".fasta", "_scatter.png")
    )
    plt.savefig(out_file, dpi=200)
    plt.close(fig)

# Histogram of Spearman correlations
valid = [c for c in ranked_correlations if isinstance(c, float) and np.isfinite(c)]
if len(valid) > 0:
    plt.figure(figsize=(6, 4))
    plt.hist(valid, bins=20, range=(-1, 1), color="steelblue", edgecolor="white")
    plt.xlabel("Spearman correlation (identity vs attention)")
    plt.ylabel("Count")
    plt.title("Distribution of Spearman correlations across MSAs")
    plt.tight_layout()
    plt.savefig(os.path.join(plt_save_dir, "spearman_hist.png"), dpi=200)
    plt.close()