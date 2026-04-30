"""Cluster ProteinGym DMS-substitution assays at 30% identity / 80% coverage.

Splits the 217 assays at the **cluster** level (80/20 by default) so that no two
assays sharing >30% identity straddle the train/test boundary. Writes a
``cluster_split.csv`` with columns ``DMS_id, cluster_rep, split`` plus a
summary of cluster sizes.

This is the v1 cluster-split helper described in
``plans/in_context_supervised_fitness.md``. mmseqs2 must be on ``$PATH`` (or
provided via ``--mmseqs-bin``).

Usage::

    python data_creation_scripts/cluster_proteingym_assays.py \
        --gym-dir /data/.../ProteinGym \
        --output-dir data/proteingym_icl/splits \
        --test-fraction 0.2 \
        --seed 42
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


def write_assay_fasta(df: pd.DataFrame, fasta_path: Path) -> None:
    with fasta_path.open("w") as f:
        for _, row in df.iterrows():
            seq = row["target_seq"]
            if not isinstance(seq, str) or len(seq) == 0:
                continue
            f.write(f">{row['DMS_id']}\n{seq}\n")


def run_mmseqs_easy_cluster(
    fasta_path: Path,
    out_prefix: Path,
    tmp_dir: Path,
    min_seq_id: float = 0.30,
    coverage: float = 0.80,
    cov_mode: int = 0,
    threads: int = 8,
    mmseqs_bin: str = "mmseqs",
) -> Path:
    cmd = [
        mmseqs_bin,
        "easy-cluster",
        str(fasta_path),
        str(out_prefix),
        str(tmp_dir),
        "--min-seq-id",
        str(min_seq_id),
        "-c",
        str(coverage),
        "--cov-mode",
        str(cov_mode),
        "--threads",
        str(threads),
        "-v",
        "1",
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    cluster_tsv = Path(f"{out_prefix}_cluster.tsv")
    if not cluster_tsv.exists():
        raise FileNotFoundError(f"Expected {cluster_tsv} not produced by mmseqs")
    return cluster_tsv


def assign_split(
    cluster_df: pd.DataFrame, test_fraction: float, seed: int
) -> pd.DataFrame:
    """80/20 split of unique cluster representatives, propagated to members."""
    cluster_reps = sorted(cluster_df["cluster_rep"].unique())
    rng = np.random.default_rng(seed)
    rng.shuffle(cluster_reps)
    n_test = max(1, int(round(len(cluster_reps) * test_fraction)))
    test_reps = set(cluster_reps[:n_test])
    cluster_df = cluster_df.copy()
    cluster_df["split"] = np.where(
        cluster_df["cluster_rep"].isin(test_reps), "test", "train"
    )
    return cluster_df


def summarise(cluster_df: pd.DataFrame) -> str:
    counts = cluster_df.groupby(["split", "cluster_rep"]).size().reset_index(
        name="size"
    )
    by_split = counts.groupby("split")["size"].agg(["count", "mean", "median", "max"])
    return (
        f"Per-split cluster stats:\n{by_split}\n\n"
        f"Total assays: {len(cluster_df)}\n"
        f"  Train: {(cluster_df['split'] == 'train').sum()}\n"
        f"  Test:  {(cluster_df['split'] == 'test').sum()}\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gym-dir", type=Path, required=True)
    parser.add_argument(
        "--csv-filename",
        default="DMS_substitutions.csv",
        help="Filename of the ProteinGym DMS reference CSV under --gym-dir",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-seq-id", type=float, default=0.30)
    parser.add_argument("--coverage", type=float, default=0.80)
    parser.add_argument("--cov-mode", type=int, default=0)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--mmseqs-bin", default="mmseqs")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.gym_dir / args.csv_filename)
    if "DMS_id" not in df.columns or "target_seq" not in df.columns:
        raise ValueError(
            f"{args.csv_filename} must contain DMS_id and target_seq columns"
        )
    df = df.dropna(subset=["DMS_id", "target_seq"]).reset_index(drop=True)

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        fasta_path = tmp / "assays.fasta"
        write_assay_fasta(df, fasta_path)
        out_prefix = tmp / "cluster"
        mm_tmp = tmp / "mmseqs_tmp"
        mm_tmp.mkdir(exist_ok=True)
        cluster_tsv = run_mmseqs_easy_cluster(
            fasta_path,
            out_prefix,
            mm_tmp,
            min_seq_id=args.min_seq_id,
            coverage=args.coverage,
            cov_mode=args.cov_mode,
            threads=args.threads,
            mmseqs_bin=args.mmseqs_bin,
        )
        cluster_df = pd.read_csv(
            cluster_tsv, sep="\t", header=None, names=["cluster_rep", "DMS_id"]
        )

    cluster_df = assign_split(cluster_df, args.test_fraction, args.seed)
    out_csv = args.output_dir / "cluster_split.csv"
    cluster_df.to_csv(out_csv, index=False)
    summary = summarise(cluster_df)
    print(summary)
    (args.output_dir / "cluster_split_summary.txt").write_text(summary)
    print(f"Wrote {out_csv}")


if __name__ == "__main__":
    main()
