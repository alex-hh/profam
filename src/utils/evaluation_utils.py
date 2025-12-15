"""
Created by Jude Wells 2025-09-12

This module contains functions for evaluating ProFam and other models.
"""

import os
import subprocess
import sys

import logomaker
import matplotlib.pyplot as plt  # noqa: F401
import numpy as np
import pandas as pd
from Bio import AlignIO, SeqIO, pairwise2


def save_per_sequence_stats(length_ratio_stats, sequence_identity_stats, csv_path):
    """
    Save per-generated-sequence stats to CSV.

    Inputs are lists of tuples:
      length_ratio_stats: (gen_id, min_ratio, max_ratio, mean_ratio)
      sequence_identity_stats: (gen_id, min_id, max_id, mean_id)

    CSV columns:
      generated_id, coverage_min, coverage_mean, coverage_max,
      identity_min, identity_mean, identity_max
    """
    # Build mapping from generated_id to metrics
    coverage_map = {}
    for gen_id, cov_min, cov_max, cov_mean in length_ratio_stats:
        coverage_map[gen_id] = {
            "coverage_min": float(cov_min),
            "coverage_mean": float(cov_mean),
            "coverage_max": float(cov_max),
        }

    identity_map = {}
    for gen_id, id_min, id_max, id_mean in sequence_identity_stats:
        identity_map[gen_id] = {
            "identity_min": float(id_min),
            "identity_mean": float(id_mean),
            "identity_max": float(id_max),
        }

    # Merge keys
    all_ids = sorted(set(list(coverage_map.keys()) + list(identity_map.keys())))
    rows = []
    for gen_id in all_ids:
        row = {"generated_id": gen_id}
        row.update(
            {
                "coverage_min": np.nan,
                "coverage_mean": np.nan,
                "coverage_max": np.nan,
                "identity_min": np.nan,
                "identity_mean": np.nan,
                "identity_max": np.nan,
            }
        )
        if gen_id in coverage_map:
            row.update(coverage_map[gen_id])
        if gen_id in identity_map:
            row.update(identity_map[gen_id])
        rows.append(row)

    df = pd.DataFrame(
        rows,
        columns=[
            "generated_id",
            "coverage_min",
            "coverage_mean",
            "coverage_max",
            "identity_min",
            "identity_mean",
            "identity_max",
        ],
    )

    out_dir = os.path.dirname(csv_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    df.to_csv(csv_path, index=False)
    return csv_path


def run_alignment_with_mafft(fasta_input, fasta_output, threads=1):
    """
    Demonstrates how you might run an alignment with MAFFT.

    Example usage:
      mafft --thread N --auto input.fasta > output.fasta
    """
    cmd = ["mafft", "--thread", str(threads), "--auto", fasta_input]
    print(f"Running: {' '.join(cmd)}", file=sys.stderr)
    with open(fasta_output, "w") as fout:
        subprocess.run(cmd, check=True, stdout=fout)


def length_ratios(prompt_fasta, generated_fasta):
    """
    Compute the length ratio of the generated sequences to the prompt sequences.
    """
    prompt_sequences = [
        (record.id, str(record.seq).replace("-", ""))
        for record in SeqIO.parse(prompt_fasta, "fasta")
    ]
    generated_sequences = [
        (record.id, str(record.seq).replace("-", ""))
        for record in SeqIO.parse(generated_fasta, "fasta")
    ]
    results = []
    for generated_id, generated_seq in generated_sequences:
        ratios = [
            len(generated_seq) / len(prompt_seq[1]) for prompt_seq in prompt_sequences
        ]
        results.append((generated_id, min(ratios), max(ratios), np.mean(ratios)))
    return results


def pairwise_sequence_identity(seq1, seq2):
    """
    Compute the pairwise sequence identity between two sequences.

    If the inputs appear aligned (contain gap characters or equal length), compute
    identity as the fraction of matching non-gap positions over positions where
    at least one sequence has a residue. Otherwise, perform a simple global
    alignment and compute identity over the resulting alignment.
    """
    s1 = str(seq1)
    s2 = str(seq2)

    # Fast path: treat as aligned if sequences contain gaps or have equal length
    def _aligned_identity(a: str, b: str) -> float:
        max_len = max(len(a), len(b))
        if max_len == 0:
            return 1.0
        # Pad to equal length for safe zipping
        if len(a) < max_len:
            a = a + ("-" * (max_len - len(a)))
        if len(b) < max_len:
            b = b + ("-" * (max_len - len(b)))
        denom = 0
        matches = 0
        for aa, bb in zip(a, b):
            if aa == "-" and bb == "-":
                continue
            denom += 1
            if aa != "-" and bb != "-" and aa == bb:
                matches += 1
        return (matches / denom) if denom > 0 else 0.0

    if "-" in s1 or "-" in s2:
        return _aligned_identity(s1, s2)

    # Fallback: global alignment maximizing matches
    try:
        aln = pairwise2.align.globalms(s1, s2, 1, 0, -1, -1, one_alignment_only=True)
        if not aln:
            return 0.0
        a, b, _score, _start, _end = aln[0]
        return _aligned_identity(a, b)
    except Exception:
        return None


def sequence_identity_from_msa(combined_msa, generated_start_idx):
    """
    Compute identity statistics for each generated sequence against all prompt sequences
    using a pre-aligned combined MSA (FASTA with gaps).

    Returns a list of tuples: (generated_id, min_id, max_id, mean_id)
    """
    alignment = AlignIO.read(combined_msa, "fasta")
    records = list(alignment)
    prompt_records = records[:generated_start_idx]
    generated_records = records[generated_start_idx:]

    prompt_seqs = [str(r.seq) for r in prompt_records]
    results = []
    for gen in generated_records:
        gen_seq = str(gen.seq)
        similarities = [pairwise_sequence_identity(p, gen_seq) for p in prompt_seqs]
        if len(similarities) == 0:
            stats = (gen.id, 0.0, 0.0, 0.0)
        else:
            stats = (
                gen.id,
                float(np.min(similarities)),
                float(np.max(similarities)),
                float(np.mean(similarities)),
            )
        results.append(stats)
    return results


def create_logo_from_fasta(alignment_fasta, output_logo):
    """Create sequence logo from aligned FASTA."""
    try:
        import logomaker
    except ImportError:
        print("logomaker not found, skipping logo creation")
        return
    alignment = AlignIO.read(alignment_fasta, "fasta")
    sequences = [str(record.seq) for record in alignment]

    # Build logomaker counts matrix
    counts_matrix = logomaker.alignment_to_matrix(sequences)
    logo = logomaker.Logo(
        counts_matrix, color_scheme="weblogo_protein", width=0.8, figsize=(60, 2.5)
    )
    logo.fig.savefig(output_logo)
    print(f"Sequence logo saved as {output_logo}")


def make_combined_fasta(prompt_fasta, generated_fasta, combined_fasta):
    """Concatenate prompt and generated FASTAs into a single file.

    Returns the number of prompt sequences written (used as generated_start_idx).
    """
    prompt_records = list(SeqIO.parse(prompt_fasta, "fasta"))
    generated_records = list(SeqIO.parse(generated_fasta, "fasta"))

    # Ensure output directory exists
    out_dir = os.path.dirname(combined_fasta)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(combined_fasta, "w") as fout:
        SeqIO.write(prompt_records + generated_records, fout, "fasta")

    return len(prompt_records)


def _msa_column_entropies_and_depths_from_sequences(seqs):
    """
    Compute per-column Shannon entropy (natural log) and non-gap depths for an aligned
    list of sequences (strings with gap '-').
    Returns (entropy_array, depth_array).
    """
    if len(seqs) == 0:
        return np.array([]), np.array([])
    length = len(seqs[0])
    entropies = np.zeros(length, dtype=float)
    depths = np.zeros(length, dtype=int)
    for col in range(length):
        column = [s[col] for s in seqs]
        residues = [c for c in column if c != "-"]
        depth = len(residues)
        depths[col] = depth
        if depth == 0:
            entropies[col] = np.nan
            continue
        unique, counts = np.unique(residues, return_counts=True)
        probs = counts / counts.sum()
        entropies[col] = -float(np.sum(probs * np.log(probs)))
    return entropies, depths


def compute_entropy_correlation(prompt_seqs, gen_seqs, min_depth=10):
    """
    Compute the correlation between per-column entropies of two aligned
    sequence sets (prompt and generated). Sequences must be gapped to the
    same alignment length.

    Returns (corr, prompt_entropies_trimmed, gen_entropies_trimmed, mask)
    where corr is a float or None if not computable, arrays are trimmed to the
    shared min length, and mask selects columns used in the correlation (depth
    >= min_depth in both and entropies not NaN).
    """
    prompt_entropies, prompt_depths = _msa_column_entropies_and_depths_from_sequences(
        prompt_seqs
    )
    gen_entropies, gen_depths = _msa_column_entropies_and_depths_from_sequences(
        gen_seqs
    )

    min_len = min(len(prompt_entropies), len(gen_entropies))
    if min_len <= 1:
        return None, prompt_entropies[:min_len], gen_entropies[:min_len], None

    prompt_e = prompt_entropies[:min_len]
    gen_e = gen_entropies[:min_len]
    prompt_d = prompt_depths[:min_len]
    gen_d = gen_depths[:min_len]

    mask = (prompt_d >= min_depth) & (gen_d >= min_depth)
    mask &= ~np.isnan(prompt_e) & ~np.isnan(gen_e)

    if not np.any(mask):
        return None, prompt_e, gen_e, mask

    corr = float(np.corrcoef(prompt_e[mask], gen_e[mask])[0, 1])
    return corr, prompt_e, gen_e, mask


def divergences_from_combined_alignment(
    combined_msa_path,
    generated_start_idx,
    min_depth=10,
    pseudocount=1e-9,
    kl_alpha_total=2.0,
):
    """
    Compute per-position divergences between natural (prompt) and synthetic (generated)
    subsets using a pre-aligned combined MSA (FASTA with gaps).

    Definitions (all logs are natural; results are in nats):
    - Let P be the empirical amino-acid distribution per column for the natural subset
      (prompt sequences), and Q for the synthetic subset (generated sequences).
    - KL(P || Q) = sum_i P_i * log(P_i / Q_i). Returned as the non-symmetric KL from
      natural to synthetic.
    - Symmetric KL (reported here) = 0.5 * (KL(P || Q) + KL(Q || P)).
    - Jensen–Shannon divergence (JSD) = 0.5 * KL(P || M) + 0.5 * KL(Q || M), where
      M = 0.5 * (P + Q).

    Assumptions:
    - Columns are evaluated only when both subsets have non-gap depth >= min_depth.
    - Gaps ('-') are excluded from counts; all other characters are treated as distinct
      residue symbols (including ambiguous codes) and included in the per-column union
      alphabet.
    - Pseudocount smoothing with value `pseudocount` is applied symmetrically to both
      P and Q over the union alphabet of observed residues at the column, to avoid
      zeros and ensure finite KL values.

    Additional smoothing (asymmetric, affects only KL(natural || synthetic)):
    - An optional Dirichlet-style prior of total mass `kl_alpha_total` is added to the
      synthetic counts, distributed uniformly over residues that appear in the natural
      column. This only affects KL(natural || synthetic); JSD and symmetric KL use
      the baseline (pseudocount-only) smoothing.

    Returns:
      (js_mean, skl_mean, kl_nat_to_syn_mean, js_per_position, skl_per_position,
       kl_nat_to_syn_per_position, mask)
    where mask selects the positions used for averaging.
    """
    alignment = AlignIO.read(combined_msa_path, "fasta")
    records = list(alignment)
    prompt_records = records[:generated_start_idx]
    generated_records = records[generated_start_idx:]

    prompt_seqs = [str(r.seq) for r in prompt_records]
    gen_seqs = [str(r.seq) for r in generated_records]

    if len(prompt_seqs) == 0 or len(gen_seqs) == 0:
        return None, None, None, None, None

    length = len(prompt_seqs[0])
    # Safety: ensure all sequences are the same length
    if any(len(s) != length for s in prompt_seqs + gen_seqs):
        # Not aligned properly
        return None, None, None, None, None

    js_values = np.full(length, np.nan, dtype=float)
    skl_values = np.full(length, np.nan, dtype=float)
    kl_nat_to_syn_values = np.full(length, np.nan, dtype=float)
    mask = np.zeros(length, dtype=bool)

    for col in range(length):
        # Collect non-gap residues in this column
        prompt_col = [s[col] for s in prompt_seqs]
        gen_col = [s[col] for s in gen_seqs]

        prompt_residues = [c for c in prompt_col if c != "-"]
        gen_residues = [c for c in gen_col if c != "-"]

        prompt_depth = len(prompt_residues)
        gen_depth = len(gen_residues)

        if prompt_depth < min_depth or gen_depth < min_depth:
            continue

        # Build empirical distributions over the union of observed residues
        pr_unique, pr_counts = np.unique(prompt_residues, return_counts=True)
        ge_unique, ge_counts = np.unique(gen_residues, return_counts=True)

        union_residues = list(sorted(set(pr_unique.tolist()) | set(ge_unique.tolist())))
        if len(union_residues) == 0:
            continue

        pr_map = {res: cnt for res, cnt in zip(pr_unique, pr_counts)}
        ge_map = {res: cnt for res, cnt in zip(ge_unique, ge_counts)}

        # Raw count vectors over the union alphabet
        p_counts = np.array(
            [pr_map.get(res, 0.0) for res in union_residues], dtype=float
        )
        q_counts = np.array(
            [ge_map.get(res, 0.0) for res in union_residues], dtype=float
        )

        # Baseline pseudocount smoothing used for JSD and symmetric KL
        k = float(len(union_residues))
        p_probs = (p_counts + pseudocount) / (p_counts.sum() + pseudocount * k)
        q_probs = (q_counts + pseudocount) / (q_counts.sum() + pseudocount * k)

        # Helper: KL divergence in nats
        def _kl(a, b):
            return float(np.sum(a * (np.log(a) - np.log(b))))

        # JSD and symmetric KL computed on baseline-smoothed distributions
        m = 0.5 * (p_probs + q_probs)
        js = 0.5 * _kl(p_probs, m) + 0.5 * _kl(q_probs, m)
        skl = 0.5 * (_kl(p_probs, q_probs) + _kl(q_probs, p_probs))

        # KL(natural || synthetic) with optional asymmetric alpha smoothing on Q
        if kl_alpha_total is not None and kl_alpha_total > 0:
            # Distribute alpha mass uniformly over residues observed in natural (support of P)
            natural_support_mask = p_counts > 0
            support_size = int(np.sum(natural_support_mask))
            if support_size > 0:
                increment = kl_alpha_total / float(support_size)
                q_counts_alpha = q_counts.copy()
                q_counts_alpha[natural_support_mask] += increment
                q_probs_for_kl = (q_counts_alpha + pseudocount) / (
                    q_counts_alpha.sum() + pseudocount * k
                )
            else:
                # Fallback to baseline if no natural support is detected (shouldn't happen due to depth check)
                q_probs_for_kl = q_probs
        else:
            q_probs_for_kl = q_probs

        kl_pq = _kl(
            p_probs, q_probs_for_kl
        )  # KL(natural || synthetic) with asymmetric smoothing

        js_values[col] = js
        skl_values[col] = skl
        kl_nat_to_syn_values[col] = kl_pq
        mask[col] = True

    if not np.any(mask):
        return None, None, None, js_values, skl_values, kl_nat_to_syn_values, mask

    js_mean = float(np.nanmean(js_values[mask]))
    skl_mean = float(np.nanmean(skl_values[mask]))
    kl_nat_to_syn_mean = float(np.nanmean(kl_nat_to_syn_values[mask]))
    return (
        js_mean,
        skl_mean,
        kl_nat_to_syn_mean,
        js_values,
        skl_values,
        kl_nat_to_syn_values,
        mask,
    )


def plot_perplexity_series(prompt_entropies, gen_entropies, mask, output_path):
    """
    Plot per-column perplexity (exp(entropy)) for prompt and generated subsets
    over the masked positions and save to output_path. Returns the output_path.
    """
    # Convert entropy (in nats) to perplexity
    prompt_perplexities = np.exp(prompt_entropies)
    gen_perplexities = np.exp(gen_entropies)

    fig, ax = plt.subplots(figsize=(12, 3))
    ax.plot(prompt_perplexities[mask], label="Prompt perplexity", linewidth=1.5)
    ax.plot(gen_perplexities[mask], label="Generation perplexity", linewidth=1.5)
    ax.set_xlabel("Alignment column")
    ax.set_ylabel("Perplexity")
    ax.set_title("Per-column perplexity (depth ≥ 10 in both subsets)")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def sequence_only_evaluation(prompt_fasta, generated_fasta, generate_logos=True):
    """
    Uses mafft to align the promp / generations independently
    then aligns the combination.
    for each generated sequence computes relevant statistics:
    - sequence identity (min, max, mean over all prompt sequences)
    - length ratio (min, max, mean over all prompt sequences)


    for the MSA overall we compute the entropy correlation with the prompt MSA.
    we only calculate this if the prompt MSA has more than 10 seqs.
    this means we calculate the entropy of each column of the MSA for the prompt and the generated sequences.
    then we compute the correlation between the entropies of the two MSAs (only on positions that have at least depth 10).

    generate logos for the prompt and generated sequences.
    """
    alignment_directory = os.path.dirname(generated_fasta) + "/alignments"
    os.makedirs(alignment_directory, exist_ok=True)
    combined_fasta_path = os.path.join(
        alignment_directory,
        os.path.basename(generated_fasta).replace(".fasta", "combined.fasta"),
    )
    prompt_count = make_combined_fasta(
        prompt_fasta, generated_fasta, combined_fasta_path
    )
    aligned_combined_path = os.path.join(
        alignment_directory,
        os.path.basename(generated_fasta).replace(".fasta", "combined_aln.fasta"),
    )
    if not os.path.exists(aligned_combined_path):
        run_alignment_with_mafft(combined_fasta_path, aligned_combined_path)

    # Compute statistics
    length_ratio_stats = length_ratios(prompt_fasta, generated_fasta)
    seq_identity_stats = sequence_identity_from_msa(aligned_combined_path, prompt_count)

    # Entropy correlation and plot using the combined alignment to ensure shared columns
    entropy_corr = None
    perplexity_plot_path = None
    js_mean, skl_mean, kl_nat_to_syn_mean = None, None, None
    try:
        combined_alignment = AlignIO.read(aligned_combined_path, "fasta")
        records = list(combined_alignment)
        prompt_records = records[:prompt_count]
        gen_records = records[prompt_count:]
        prompt_seqs = [str(r.seq) for r in prompt_records]
        gen_seqs = [str(r.seq) for r in gen_records]

        corr, prompt_e, gen_e, mask = compute_entropy_correlation(
            prompt_seqs, gen_seqs, min_depth=10
        )
        entropy_corr = corr
        if corr is not None and mask is not None and np.any(mask):
            try:
                perplexity_plot_path = os.path.join(
                    alignment_directory,
                    os.path.basename(generated_fasta).replace(
                        ".fasta", "combined_perplexity.png"
                    ),
                )
                plot_perplexity_series(prompt_e, gen_e, mask, perplexity_plot_path)
            except ImportError:
                pass  # plotting is optional
        # Divergences (JSD, symmetric KL, and KL(natural||synthetic)) over positions with depth >= 10 in both subsets
        try:
            (
                js_mean,
                skl_mean,
                kl_nat_to_syn_mean,
                _js_vals,
                _skl_vals,
                _kl_pq_vals,
                _mask,
            ) = divergences_from_combined_alignment(
                aligned_combined_path, prompt_count, min_depth=10
            )
        except Exception:
            js_mean, skl_mean, kl_nat_to_syn_mean = None, None, None
    except Exception:
        pass  # entropy from combined alignment is best-effort

    # Save per-sequence stats CSV
    csv_path = os.path.join(
        alignment_directory,
        os.path.basename(generated_fasta).replace(".fasta", "_seq_stats.csv"),
    )
    try:
        save_per_sequence_stats(length_ratio_stats, seq_identity_stats, csv_path)
    except Exception:
        csv_path = None

    results = {
        "aligned_combined_path": aligned_combined_path,
        "entropy_correlation": round(entropy_corr, 3)
        if entropy_corr is not None
        else None,
        "perplexity_plot_path": perplexity_plot_path,
        "per_sequence_csv": csv_path,
    }
    # Create sequence logos (optional)
    if generate_logos:
        aligned_generation_path = os.path.join(
            alignment_directory,
            os.path.basename(generated_fasta).replace(".fasta", "_aln.fasta"),
        )
        aligned_prompt_path = os.path.join(
            alignment_directory,
            os.path.basename(generated_fasta).replace(".fasta", "prompt_aln.fasta"),
        )
        if not os.path.exists(aligned_generation_path):
            run_alignment_with_mafft(generated_fasta, aligned_generation_path)
        if "aligned" in prompt_fasta or "_aln.fasta" in prompt_fasta:
            aligned_prompt_path = prompt_fasta
        else:
            run_alignment_with_mafft(prompt_fasta, aligned_prompt_path)
        results["aligned_generation_path"] = aligned_generation_path
        results["aligned_prompt_path"] = aligned_prompt_path
        try:
            prompt_logo = os.path.join(
                alignment_directory,
                os.path.basename(generated_fasta).replace(".fasta", "prompt_logo.png"),
            )
            gen_logo = os.path.join(
                alignment_directory,
                os.path.basename(generated_fasta).replace(".fasta", "_logo.png"),
            )
            create_logo_from_fasta(aligned_prompt_path, prompt_logo)
            create_logo_from_fasta(aligned_generation_path, gen_logo)
        except Exception:
            # Logo creation is best-effort
            pass

    # Add averaged divergences if available
    results["js_divergence_mean"] = round(js_mean, 3) if js_mean is not None else None
    results["symmetric_kl_divergence_mean"] = (
        round(skl_mean, 3) if skl_mean is not None else None
    )
    results["kl_natural_to_synthetic_mean"] = (
        round(kl_nat_to_syn_mean, 3) if kl_nat_to_syn_mean is not None else None
    )
    for aggregation_strategy in ["min", "max", "mean"]:
        if aggregation_strategy == "mean":
            agg_func = np.mean
        elif aggregation_strategy == "min":
            agg_func = np.min
        elif aggregation_strategy == "max":
            agg_func = np.max
        for ix, inner_strategy in enumerate(["min", "max", "mean"]):
            results[f"length_ratio_{aggregation_strategy}_of_{inner_strategy}"] = round(
                agg_func([entry[ix + 1] for entry in length_ratio_stats]), 3
            )
            results[
                f"sequence_identity_{aggregation_strategy}_of_{inner_strategy}"
            ] = round(agg_func([entry[ix + 1] for entry in seq_identity_stats]), 3)
    return results
