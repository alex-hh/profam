import glob
import os
import shutil
import subprocess
import tempfile
from Bio import SeqIO
import numpy as np
import random
from typing import Dict, List, Optional, Tuple
from data_creation_scripts.ec_clustered_validation_dataset import align_with_mafft, run_hhfilter

# =========================
# SCA / covariance utilities
# =========================

# 20 standard amino acids plus gap
ALPHABET_21 = "ACDEFGHIKLMNPQRSTVWY-"
ALPHABET_20 = "ACDEFGHIKLMNPQRSTVWY"


def parse_aligned_fasta_to_indices(
    aligned_fasta_path: str,
    alphabet: str = ALPHABET_20,
) -> Tuple[np.ndarray, List[str]]:
    """
    Parse an aligned FASTA into an integer array of shape (num_sequences, L),
    using 0..20 for the 20 amino acids plus gap ('-'), and -1 for unknown tokens.

    Returns (indices, sequence_ids).
    """
    records = list(SeqIO.parse(aligned_fasta_path, "fasta"))
    if len(records) == 0:
        raise ValueError(f"No sequences found in {aligned_fasta_path}")
    seq_ids = [r.id for r in records]
    seq_strs = [str(r.seq) for r in records]
    L = len(seq_strs[0])
    for s in seq_strs:
        if len(s) != L:
            raise ValueError("All sequences must be the same aligned length")

    aa_to_index = {aa: i for i, aa in enumerate(alphabet)}
    x = np.full((len(seq_strs), L), -1, dtype=np.int16)
    for n, s in enumerate(seq_strs):
        for i, ch in enumerate(s):
            x[n, i] = aa_to_index.get(ch, -1)
    return x, seq_ids


def compute_symbol_covariances(
    aligned_fasta_path: str,
    alphabet: str = ALPHABET_20,
    return_probabilities: bool = False,
    dtype: np.dtype = np.float32,
) -> Dict[str, np.ndarray]:
    """
    Compute SCA-style covariances for an aligned MSA, treating gap as the 21st state.

    For each position i, j and symbols a, b in the 21-state alphabet:
      covariance[i, j, a, b] = P_ij(a, b) - P_i(a) * P_j(b)

    where probabilities are estimated by relative frequencies across sequences that
    have valid symbols at both positions (unknown/non-standard residues are excluded).

    Returns a dict with:
      - 'covariance': (L, L, S, S) float32 array
      - 'support_pair': (L, L) int32 array (# sequences contributing to each (i, j))
      - 'p_single': (L, S) float32 array of marginal probabilities
      - 'support_single': (L,) int32 array (# sequences contributing to each position)
      - 'counts_single': (L, S) int32 array (symbol-specific counts per position)
      - 'counts_pair': (L, L, S, S) int32 array (symbol-pair counts per position-pair)
      - optionally 'p_pair': (L, L, S, S) if return_probabilities=True
      - 'alphabet': str

    Notes:
      - S = len(alphabet) = 21 by default.
      - The support for each (i, j, a, b) entry is the same across (a, b) and equals support_pair[i, j].
    """
    x, _ = parse_aligned_fasta_to_indices(aligned_fasta_path, alphabet=alphabet)
    num_sequences, L = x.shape
    S = len(alphabet)

    # Known masks: unknown residues are encoded as -1
    known = (x >= 0)

    # One-hot encode known states only (unknown -> all zeros)
    one_hot = np.zeros((num_sequences, L, S), dtype=dtype)
    rows, cols = np.where(known)
    one_hot[rows, cols, x[rows, cols]] = 1.0

    # Single-position counts and probabilities
    counts_single = one_hot.sum(axis=0).astype(np.int32)  # (L, S)
    support_single = known.sum(axis=0).astype(np.int32)  # (L,)
    with np.errstate(divide="ignore", invalid="ignore"):
        p_single = counts_single / np.maximum(support_single[:, None], 1)
        p_single = np.where(np.isfinite(p_single), p_single, 0).astype(dtype)

    # Pairwise counts and support
    # counts_pair[i, j, a, b] = sum_n one_hot[n, i, a] * one_hot[n, j, b]
    counts_pair = np.einsum("nia,njb->ijab", one_hot, one_hot, optimize=True).astype(np.int32)
    support_pair = np.einsum("ni,nj->ij", known.astype(np.int32), known.astype(np.int32), optimize=True).astype(np.int32)

    with np.errstate(divide="ignore", invalid="ignore"):
        p_pair = counts_pair / np.maximum(support_pair[:, :, None, None], 1)

        p_pair = np.where(np.isfinite(p_pair), p_pair, 0).astype(dtype)

    # Covariance: P_ij(a,b) - P_i(a) * P_j(b)
    expected = p_single[:, None, :, None] * p_single[None, :, None, :]
    covariance = (p_pair - expected).astype(dtype)

    out: Dict[str, np.ndarray] = {
        "covariance": covariance,
        "support_pair": support_pair,
        "p_single": p_single,
        "support_single": support_single,
        "counts_single": counts_single,
        "counts_pair": counts_pair,
        "alphabet": np.array(list(alphabet)),
    }
    if return_probabilities:
        out["p_pair"] = p_pair
    return out


def assess_correlation_preservation(
    natural_aligned_fasta: str,
    synthetic_aligned_fastas: List[str],
    min_support: int = 10,
    include_diagonal: bool = False,
    min_symbol_counts: Optional[Tuple[int, int]] = None,
    # e.g., (min_count_pos_i, min_count_pos_j). If set, require that for each (i,j)
    # the most constrained symbols at i and j (max over symbols actually used in r)
    # have at least these counts in both natural and synthetic.
    require_non_gap: bool = False,
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Compare SCA covariances between a natural MSA and one or more synthetic MSAs.

    For each synthetic MSA, returns metrics including:
      - 'global_r': Pearson correlation across all (i, j, a, b) entries passing support filters
      - 'pair_r_matrix': (L, L) matrix of Pearson r per position pair across 21x21 entries
      - 'mean_pair_r', 'median_pair_r': summary stats of pairwise r values (ignoring NaNs)
      - 'coverage': fraction of position pairs (i, j) passing the support filter

    Filtering: a pair (i, j) is included if both natural and synthetic support_pair[i, j] >= min_support.
    """
    # Natural
    nat = compute_symbol_covariances(natural_aligned_fasta, return_probabilities=False)
    cov_nat = nat["covariance"].astype(np.float64)
    sup_nat = nat["support_pair"]
    cnt_single_nat = nat["counts_single"]  # (L, S)
    alphabet = "".join(nat["alphabet"].tolist()) if isinstance(nat["alphabet"], np.ndarray) else nat["alphabet"]
    gap_index = alphabet.find("-") if "-" in alphabet else -1
    L = cov_nat.shape[0]

    results: Dict[str, Dict[str, np.ndarray]] = {}

    for syn_path in synthetic_aligned_fastas:
        syn = compute_symbol_covariances(syn_path, return_probabilities=False)
        cov_syn = syn["covariance"].astype(np.float64)
        sup_syn = syn["support_pair"]
        cnt_single_syn = syn["counts_single"]

        if cov_syn.shape != cov_nat.shape:
            raise ValueError(
                f"Shape mismatch between natural {cov_nat.shape} and synthetic {syn_path} {cov_syn.shape}. Alignments must have same length and alphabet."
            )

        # Mask by column-pair support and diagonal
        mask_pairs = (sup_nat >= min_support) & (sup_syn >= min_support)
        if not include_diagonal:
            diag = np.eye(L, dtype=bool)
            mask_pairs &= ~diag

        # Optional: symbol-aware constraints at each single column
        if min_symbol_counts is not None:
            min_i, min_j = int(min_symbol_counts[0]), int(min_symbol_counts[1])
            # We do not know in advance which symbols will dominate the covariance at a pair,
            # so use a conservative filter: require that the maximum count across non-gap symbols
            # at i (and j) is at least min_i (min_j) in both nat and syn.
            def best_non_gap_count(counts_per_pos: np.ndarray) -> np.ndarray:
                if require_non_gap and gap_index >= 0:
                    mask = np.ones(counts_per_pos.shape[1], dtype=bool)
                    mask[gap_index] = False
                    return counts_per_pos[:, mask].max(axis=1)
                return counts_per_pos.max(axis=1)

            best_i_nat = best_non_gap_count(cnt_single_nat)
            best_j_nat = best_i_nat  # same array, re-index per column later
            best_i_syn = best_non_gap_count(cnt_single_syn)
            best_j_syn = best_i_syn

            # Build a (L, L) mask from per-column thresholds
            col_ok_nat_i = best_i_nat[:, None] >= min_i
            col_ok_nat_j = best_j_nat[None, :] >= min_j
            col_ok_syn_i = best_i_syn[:, None] >= min_i
            col_ok_syn_j = best_j_syn[None, :] >= min_j
            mask_pairs &= col_ok_nat_i & col_ok_nat_j & col_ok_syn_i & col_ok_syn_j

        # Pairwise r per (i, j) across 21x21 entries; optionally exclude gap rows/cols
        if require_non_gap and gap_index >= 0:
            # Remove gap row/col from the (S, S) block before flattening
            X = np.delete(np.delete(cov_nat, gap_index, axis=2), gap_index, axis=3).reshape(L, L, -1)
            Y = np.delete(np.delete(cov_syn, gap_index, axis=2), gap_index, axis=3).reshape(L, L, -1)
        else:
            X = cov_nat.reshape(L, L, -1)
            Y = cov_syn.reshape(L, L, -1)

        # Initialize r matrix with NaNs
        pair_r = np.full((L, L), np.nan, dtype=np.float64)

        valid_i, valid_j = np.where(mask_pairs)
        if valid_i.size > 0:
            # Compute means and stds along the 441-dim feature axis
            x_sel = X[valid_i, valid_j, :]
            y_sel = Y[valid_i, valid_j, :]
            x_mean = x_sel.mean(axis=1)
            y_mean = y_sel.mean(axis=1)
            x_center = x_sel - x_mean[:, None]
            y_center = y_sel - y_mean[:, None]
            x_std = np.sqrt((x_center * x_center).sum(axis=1))
            y_std = np.sqrt((y_center * y_center).sum(axis=1))
            denom = x_std * y_std
            # Avoid division by zero
            with np.errstate(divide="ignore", invalid="ignore"):
                num = (x_center * y_center).sum(axis=1)
                r_vals = np.where(denom > 0, num / denom, np.nan)
            pair_r[valid_i, valid_j] = r_vals

        # Global r over all entries that pass the pair mask
        if mask_pairs.any():
            mask_entries = np.repeat(mask_pairs[:, :, None], X.shape[-1], axis=2)
            x_all = X[mask_entries]
            y_all = Y[mask_entries]
            if x_all.size > 0 and y_all.size > 0:
                # Pearson correlation of flattened vectors
                x_all_center = x_all - x_all.mean()
                y_all_center = y_all - y_all.mean()
                denom = np.sqrt((x_all_center ** 2).sum() * (y_all_center ** 2).sum())
                global_r = float(x_all_center.dot(y_all_center) / denom) if denom > 0 else float("nan")
            else:
                global_r = float("nan")
        else:
            global_r = float("nan")

        coverage = float(mask_pairs.sum()) / float((L * L) - (0 if include_diagonal else L)) if L > 0 else 0.0

        results[syn_path] = {
            "global_r": np.array(global_r),
            "pair_r_matrix": pair_r,
            "mean_pair_r": np.array(np.nanmean(pair_r)),
            "median_pair_r": np.array(np.nanmedian(pair_r)),
            "coverage": np.array(coverage),
        }

    return results

def make_aligned_filtered_fasta(fasta_path: str):
    alignment_dir = os.path.join(os.path.dirname(fasta_path), "alignments")
    os.makedirs(alignment_dir, exist_ok=True)
    aligned_fasta_path = os.path.join(alignment_dir, os.path.basename(fasta_path).replace(".fasta", "_aln.fasta"))
    if not os.path.exists(aligned_fasta_path):
        align_with_mafft(fasta_path, aligned_fasta_path)
    filtered_fasta_path = os.path.join(alignment_dir, os.path.basename(fasta_path).replace(".fasta", "_aln.filtered.fasta"))
    if not os.path.exists(filtered_fasta_path):
        run_hhfilter(aligned_fasta_path, filtered_fasta_path)
    return filtered_fasta_path


if __name__ == "__main__":
    synthetic_fasta_pattern = "../sampling_results/profam_ec_multi_seq_clustered_c70_pid_30_with_ensemble/*.fasta"
    synthetic_fasta_paths = glob.glob(synthetic_fasta_pattern)
    synthetic_ec_nums = [os.path.basename(f).split("_cluster")[0] for f in synthetic_fasta_paths]
    natural_fasta_pattern = "../data/ec/ec_validation_dataset_clustered_c70_pid_30/alignments/*_cluster_aln.filtered.fasta"
    natural_fasta_paths = glob.glob(natural_fasta_pattern)
    natural_ec_nums = [os.path.basename(f).split("_cluster")[0] for f in natural_fasta_paths]
    natural_fasta_paths = [f for f in natural_fasta_paths if os.path.basename(f).split("_cluster")[0] in synthetic_ec_nums]
    for synthetic_fasta in synthetic_fasta_paths:
        aligned_filtered_fasta = make_aligned_filtered_fasta(synthetic_fasta)
        natural_fasta = [f for f in natural_fasta_paths if os.path.basename(f).split("_cluster")[0] == os.path.basename(synthetic_fasta).split("_cluster")[0]][0]
        results = assess_correlation_preservation(natural_fasta, [aligned_filtered_fasta])
        print(results)