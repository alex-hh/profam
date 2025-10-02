import glob
import os
import shutil
import subprocess
import tempfile
from Bio import SeqIO
from Bio.Seq import Seq
import numpy as np
import random
import pandas as pd
from typing import Dict, List, Optional, Tuple
from data_creation_scripts.ec_clustered_validation_dataset import align_with_mafft, run_hhfilter

# =========================
# SCA / covariance utilities
# =========================

# 20 standard amino acids; gaps are ignored for correlation purposes
ALPHABET_20 = "ACDEFGHIKLMNPQRSTVWY"


def parse_aligned_fasta_to_indices(
    aligned_fasta_path: str,
    alphabet: str = ALPHABET_20,
) -> Tuple[np.ndarray, List[str]]:
    """
    Parse an aligned FASTA into an integer array of shape (num_sequences, L),
    using 0..19 for the 20 natural amino acids. Gaps ('-') and unknown tokens
    are encoded as -1 and excluded from correlation/counting.

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
    Compute SCA-style covariances for an aligned MSA over a 20-state alphabet.

    For each position i, j and symbols a, b in the 20-state alphabet:
      covariance[i, j, a, b] = P_ij(a, b) - P_i(a) * P_j(b)

    where probabilities are estimated by relative frequencies across sequences that
      have valid non-gap symbols at both positions (gaps/unknown residues are excluded).

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
      - S = len(alphabet) = 20 by default.
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
    min_symbol_counts: Optional[Tuple[int, int]] = (10,0),
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Compare SCA covariances between a natural MSA and one or more synthetic MSAs.

    For each synthetic MSA, returns metrics including:
      - 'global_r': Pearson correlation across all (i, j, a, b) entries passing support filters
      - 'pair_r_matrix': (L, L) matrix of Pearson r per position pair across symbol entries
      - 'mean_pair_r', 'median_pair_r': summary stats of pairwise r values (ignoring NaNs)
      - 'coverage': fraction of position pairs (i, j) passing the support filter

    Filtering:
      - Position-pair base filter: both natural and synthetic support_pair[i, j] >= min_support.
      - Non-symmetric row-wise filter: for each (i, j), keep only entries (a, b) where
        amino acid a at position i has count >= min_support in both natural and synthetic.
        No filter is applied on amino acid b at position j. If require_non_gap=True and
        the alphabet includes '-', gap rows/cols are excluded as well.
    """
    # Natural
    nat = compute_symbol_covariances(natural_aligned_fasta, return_probabilities=False)
    cov_nat = nat["covariance"].astype(np.float64)
    sup_nat = nat["support_pair"]
    cnt_single_nat = nat["counts_single"]  # (L, S)
    L = cov_nat.shape[0]
    S = cov_nat.shape[2]

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

        # Base mask by column-pair support and diagonal handling
        mask_pairs = (sup_nat >= min_support) & (sup_syn >= min_support)
        if not include_diagonal:
            diag = np.eye(L, dtype=bool)
            mask_pairs &= ~diag

        # Optional: symbol-aware constraints at each single column
        if min_symbol_counts is not None:
            min_i, min_j = int(min_symbol_counts[0]), int(min_symbol_counts[1])
            # Conservative filter: require that the maximum-residue count at i (and j)
            # is at least min_i (min_j) in both nat and syn.
            def best_count(counts_per_pos: np.ndarray) -> np.ndarray:
                return counts_per_pos.max(axis=1)

            best_i_nat = best_count(cnt_single_nat)
            best_j_nat = best_i_nat  # same array, re-index per column later
            best_i_syn = best_count(cnt_single_syn)
            best_j_syn = best_i_syn

            # Build a (L, L) mask from per-column thresholds
            col_ok_nat_i = best_i_nat[:, None] >= min_i
            col_ok_nat_j = best_j_nat[None, :] >= min_j
            col_ok_syn_i = best_i_syn[:, None] >= min_i
            col_ok_syn_j = best_j_syn[None, :] >= min_j
            mask_pairs &= col_ok_nat_i & col_ok_nat_j & col_ok_syn_i & col_ok_syn_j

        # Build non-symmetric amino-acid specific row mask at position i
        # Only include rows (amino acid a) at position i if count >= min_support in BOTH nat and syn
        row_ok_nat = (cnt_single_nat >= min_support)  # (L, S)
        row_ok_syn = (cnt_single_syn >= min_support)  # (L, S)
        row_ok = row_ok_nat & row_ok_syn  # (L, S)

        # Entry-wise mask: (L, L, S, S). Depends on i via row_ok, not on j (non-symmetric)
        # Ensure mask has full (S, S) per-pair shape to match X_full[ii, jj, :, :]
        b_ok = np.ones(S, dtype=bool)  # keep all column symbols (20-state, no gap)
        mask_entries = mask_pairs[:, :, None, None] & row_ok[:, None, :, None] & b_ok[None, None, None, :]

        # We'll compute correlations directly from the full tensors using the entry-wise mask
        X_full = cov_nat
        Y_full = cov_syn

        # Initialize r matrix with NaNs
        pair_r = np.full((L, L), np.nan, dtype=np.float64)

        # Determine which (i, j) have at least one valid (a, b) entry after masking
        has_features = np.any(mask_entries, axis=(2, 3))  # (L, L)
        valid_i, valid_j = np.where(has_features)
        if valid_i.size > 0:
            for ii, jj in zip(valid_i, valid_j):
                mask_ab = mask_entries[ii, jj, :, :]
                x_vec = X_full[ii, jj, :, :][mask_ab]
                y_vec = Y_full[ii, jj, :, :][mask_ab]
                if x_vec.size == 0 or y_vec.size == 0:
                    continue
                r_val = np.corrcoef(x_vec, y_vec)[0, 1]
                pair_r[ii, jj] = r_val

        # Global r over all entries that pass the pair mask
        if np.any(mask_entries):
            x_all = X_full[mask_entries]
            y_all = Y_full[mask_entries]
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

        # Coverage: fraction of (i, j) with at least one valid (a, b) entry
        cover_mask = has_features.copy()
        if not include_diagonal:
            # Ensure diagonal is excluded from the denominator and numerator
            diag = np.eye(L, dtype=bool)
            cover_mask &= ~diag
        denom_pairs = float((L * L) - (0 if include_diagonal else L)) if L > 0 else 0.0
        coverage = float(cover_mask.sum()) / denom_pairs if denom_pairs > 0 else 0.0

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


def make_combined_aligned_fastas(natural_fasta: str, synthetic_fasta: str) -> Tuple[str, str, str]:
    """
    Create a combined FASTA (natural followed by synthetic), run a single MAFFT alignment,
    then split the aligned combined MSA back into aligned natural and aligned synthetic FASTAs.

    Returns (natural_aligned_path, synthetic_aligned_path, combined_aligned_path).
    """
    align_dir = os.path.join(os.path.dirname(synthetic_fasta), "alignments")
    os.makedirs(align_dir, exist_ok=True)

    base = os.path.basename(synthetic_fasta).replace(".fasta", "")
    combined_fasta_path = os.path.join(align_dir, f"{base}_combined.fasta")
    combined_aln_path = os.path.join(align_dir, f"{base}_combined_aln.fasta")
    nat_aln_path = os.path.join(align_dir, f"{base}_natural_aln.fasta")
    syn_aln_path = os.path.join(align_dir, f"{base}_synthetic_aln.fasta")

    # Write combined FASTA if missing and remember split index
    if not os.path.exists(combined_fasta_path):
        nat_records = list(SeqIO.parse(natural_fasta, "fasta"))
        # Natural FASTAs may already be aligned; strip gap characters before combining
        for r in nat_records:
            # Remove common gap characters from natural sequences
            r.seq = Seq(str(r.seq).replace("-", "").replace(".", ""))

        syn_records = list(SeqIO.parse(synthetic_fasta, "fasta"))
        with open(combined_fasta_path, "w") as fout:
            SeqIO.write(nat_records + syn_records, fout, "fasta")
        nat_count = len(nat_records)
    else:
        # If combined exists, derive nat_count from natural file
        nat_count = sum(1 for _ in SeqIO.parse(natural_fasta, "fasta"))

    # Align combined once
    if not os.path.exists(combined_aln_path):
        align_with_mafft(combined_fasta_path, combined_aln_path)

    # Split aligned combined into aligned natural/synthetic
    need_split = (not os.path.exists(nat_aln_path)) or (not os.path.exists(syn_aln_path))
    if need_split:
        aligned_records = list(SeqIO.parse(combined_aln_path, "fasta"))
        nat_aligned = aligned_records[:nat_count]
        syn_aligned = aligned_records[nat_count:]
        with open(nat_aln_path, "w") as f_nat:
            SeqIO.write(nat_aligned, f_nat, "fasta")
        with open(syn_aln_path, "w") as f_syn:
            SeqIO.write(syn_aligned, f_syn, "fasta")

    return nat_aln_path, syn_aln_path, combined_aln_path


if __name__ == "__main__":
    synthetic_fasta_pattern = "../sampling_results/profam_ec_multi_seq_clustered_c70_pid_30_with_ensemble/*.fasta"
    synthetic_fasta_paths = glob.glob(synthetic_fasta_pattern)
    synthetic_ec_nums = [os.path.basename(f).split("_cluster")[0] for f in synthetic_fasta_paths]
    natural_fasta_pattern = "../data/ec/ec_validation_dataset_clustered_c70_pid_30/alignments/*filtered.fasta"
    natural_fasta_paths = glob.glob(natural_fasta_pattern)
    natural_ec_nums = [os.path.basename(f).split("_cluster")[0] for f in natural_fasta_paths]
    # Keep only ECs that have synthetic generations
    all_rows = []
    natural_fasta_paths = [f for f in natural_fasta_paths if os.path.basename(f).split("_cluster")[0] in synthetic_ec_nums]
    for synthetic_fasta in synthetic_fasta_paths:
        ec_id_from_synth = os.path.basename(synthetic_fasta).split("_cluster")[0]
        matching_natural = [f for f in natural_fasta_paths if os.path.basename(f).split("_cluster")[0] == ec_id_from_synth]
        if len(matching_natural) == 0:
            continue
        natural_eval_fasta = matching_natural[0]

        # Build combined alignment and split into aligned natural and aligned synthetic
        nat_aln_path, syn_aln_path, _combined_aln = make_combined_aligned_fastas(natural_eval_fasta, synthetic_fasta)

        # Compute covariance correlations on synchronized columns
        results = assess_correlation_preservation(nat_aln_path, [syn_aln_path])
        for k, v in results.items():
            new_row = {
                "ec_id": ec_id_from_synth,
                "natural_fasta": natural_eval_fasta,
                "synthetic_fasta": k
                }
            new_row.update(v)
        all_rows.append(new_row)
        df = pd.DataFrame(all_rows)
        df.to_csv("../sampling_results/profam_ec_multi_seq_clustered_c70_pid_30_with_ensemble/covariance_analysis.csv", index=False)