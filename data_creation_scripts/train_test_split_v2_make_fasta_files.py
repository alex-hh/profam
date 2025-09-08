"""
Created by Jude Wells 2025-09-08
Takes the val and test holdouts that were created in:
data_creation_scripts/mmseqs_train_test_split.py
and makes fasta files of the sequences for each dataset

the following files should be used for evaluating the model:

Foldseek:
../data/foldseek/foldseek_s100_raw/train_test_split_v2/val/foldseek_s100_raw_val.parquet
../data/foldseek/foldseek_s100_raw/train_test_split_v2/test/foldseek_s100_raw_test.parquet

Funfams:
../data/funfams/s50_parquets/train_test_split_v2/val/val_000.parquet
../data/funfams/s50_parquets/train_test_split_v2/test/test_000.parquet

Pfam:
../data/pfam/train_test_split_parquets/val_clustered_split_combined.parquet
../data/pfam/train_test_split_parquets/test_clustered_split_combined.parquet

PETase:
../data/petase/combined_petase_sequences.fasta

For each parquet file:
1) Create un-aligned fasta files of the sequences (1 fasta file for each protein family: which is one row in the parquet file)
2) For each fasta file, if the number of sequences is greater than 100, run hhfilter to remove sequences with >90% identity and <50% coverage. For Foldseek families, align with MAFFT first using run_alignment_with_mafft.
"""

import os
import pandas as pd
import sys
import shutil
import subprocess
import glob
import glob

from scripts.adhoc_analysis.generate_cluster_alignments_and_logos import run_alignment_with_mafft



HHFILTER_BINARY = os.environ.get("HHFILTER_BINARY", "/mnt/disk2/msa_pairformer/hhsuite/hhfilter")


def count_fasta_records(fasta_path):
    if not os.path.exists(fasta_path):
        return 0
    count = 0
    with open(fasta_path, "r") as fh:
        for line in fh:
            if line.startswith(">"):
                count += 1
    return count


def run_hhfilter(input_path, output_path, max_seq_id=90, min_cov=None):
    """Run hhfilter on the given input alignment/FASTA and write to output_path.

    If min_cov is None, the -cov flag is omitted (any coverage).
    """
    if not os.path.exists(HHFILTER_BINARY):
        print(f"hhfilter binary not found at {HHFILTER_BINARY}; skipping filtering for {input_path}")
        return False
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cmd = [
        HHFILTER_BINARY,
        "-i", input_path,
        "-o", output_path,
        "-id", str(max_seq_id),  # remove seqs with identity > max_seq_id
    ]
    if min_cov is not None:
        cmd += ["-cov", str(min_cov)]  # require at least min_cov coverage
    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"hhfilter failed for {input_path} → {output_path}: {e}")
        return False


def make_fasta_files(parquet_file):
    df = pd.read_parquet(parquet_file)
    save_dir = os.path.join(os.path.dirname(parquet_file), "fastas")
    # Determine dataset name and split for aggregation
    # Path format example: ../data/foldseek/.../val/... or ../data/funfams/.../test/...
    try:
        ds_name = parquet_file.split("/")[2]
    except Exception:
        ds_name = "unknown"
    split = "val" if ("/val/" in parquet_file or "val_clustered" in parquet_file) else ("test" if "/test/" in parquet_file or "test_clustered" in parquet_file else "unknown")

    os.makedirs(save_dir, exist_ok=True)
    rows = []
    for i, row in df.iterrows():
        fam_id = row["fam_id"]
        sequences = row["sequences"]
        accessions = row["accessions"]
        fasta_file = os.path.join(save_dir, f"{fam_id}.fasta")
        # Aggregated output root
        aggregated_root = os.path.join("../data/val_test_v2_fastas", ds_name, split)
        os.makedirs(aggregated_root, exist_ok=True)

        # If an aggregated representative already exists, skip recomputation for this family
        aggregated_exists = False
        # Prefer strict then light naming (consider aligned variants for foldseek)
        agg_candidates = [
            os.path.join(aggregated_root, f"{fam_id}.aligned.id90_cov20.fasta"),
            os.path.join(aggregated_root, f"{fam_id}.id90_cov20.fasta"),
            os.path.join(aggregated_root, f"{fam_id}.aligned.id999.fasta"),
            os.path.join(aggregated_root, f"{fam_id}.id999.fasta"),
        ]
        for agg_path in agg_candidates:
            if os.path.exists(agg_path):
                aggregated_exists = True
                break
        if aggregated_exists and os.path.exists(fasta_file):
            continue

        new_row = {
            "parquet_file": parquet_file,
            "fam_id": fam_id,
            "n_seqs": len(accessions),
            "min_seq_length": min(len(seq) for seq in sequences),
            "max_seq_length": max(len(seq) for seq in sequences),
            "mean_seq_length": round(sum(len(seq) for seq in sequences) / len(sequences),0),
        }
        rows.append(new_row)
        # Write base FASTA only if missing
        if not os.path.exists(fasta_file):
            with open(fasta_file, "w") as f:
                for i, sequence in enumerate(sequences):
                    f.write(f">{accessions[i]}\n{sequence}\n")

        # Always align Foldseek families to produce aligned FASTAs
        is_foldseek = "foldseek" in parquet_file.lower()
        aligned_path = None
        if is_foldseek:
            aligned_path = os.path.join(save_dir, f"{fam_id}.aligned.fasta")
            if not os.path.exists(aligned_path):
                try:
                    run_alignment_with_mafft(
                        fasta_file,
                        aligned_path,
                        threads=max(1, (os.cpu_count() or 1)//2),
                    )
                except Exception as e:
                    print(f"MAFFT alignment failed for {fasta_file}: {e}")
                    aligned_path = None

        # Always run an initial light hhfilter (99.9% id, any coverage)
        input_for_filter = aligned_path if (is_foldseek and aligned_path) else fasta_file

        light_suffix = ".aligned.id999.fasta" if is_foldseek else ".id999.fasta"
        light_out = os.path.join(save_dir, f"{fam_id}{light_suffix}")

        if os.path.exists(light_out):
            ok_light = True
            before_count_light = count_fasta_records(input_for_filter)
            after_count_light = count_fasta_records(light_out)
        else:
            before_count_light = count_fasta_records(input_for_filter)
            ok_light = run_hhfilter(input_for_filter, light_out, max_seq_id=99.9, min_cov=None)
            after_count_light = count_fasta_records(light_out) if ok_light else 0
        print(
            f"hhfilter(light) fam={fam_id} before={before_count_light} after={after_count_light} "
            f"input={os.path.basename(input_for_filter)} output={os.path.basename(light_out)}"
        )

        # If sufficiently deep, apply stricter hhfilter (90% id, 20% coverage)
        if after_count_light > 100 and ok_light:
            strict_input = light_out
            strict_suffix = ".aligned.id90_cov20.fasta" if is_foldseek else ".id90_cov20.fasta"
            strict_out = os.path.join(save_dir, f"{fam_id}{strict_suffix}")

            if os.path.exists(strict_out):
                before_count_strict = count_fasta_records(strict_input)
                ok_strict = True
                after_count_strict = count_fasta_records(strict_out)
            else:
                before_count_strict = count_fasta_records(strict_input)
                ok_strict = run_hhfilter(strict_input, strict_out, max_seq_id=90, min_cov=20)
                after_count_strict = count_fasta_records(strict_out) if ok_strict else 0
            print(
                f"hhfilter(strict) fam={fam_id} before={before_count_strict} after={after_count_strict} "
                f"input={os.path.basename(strict_input)} output={os.path.basename(strict_out)}"
            )

        # Aggregate best FASTA for this family into ../data/val_test_v2_fastas/{DS_NAME}/{split}
        # Choose strict if present (>1 sequence), else light (>1 sequence). Check aligned first for foldseek
        candidates = [
            os.path.join(save_dir, f"{fam_id}.aligned.id90_cov20.fasta"),
            os.path.join(save_dir, f"{fam_id}.id90_cov20.fasta"),
            os.path.join(save_dir, f"{fam_id}.aligned.id999.fasta"),
            os.path.join(save_dir, f"{fam_id}.id999.fasta"),
        ]
        chosen = None
        for cand in candidates:
            if os.path.exists(cand) and count_fasta_records(cand) > 1:
                chosen = cand
                break
        if chosen is not None:
            # Copy with original basename to preserve details
            target_path = os.path.join(aggregated_root, os.path.basename(chosen))
            if not os.path.exists(target_path):
                shutil.copy(chosen, target_path)
        else:
            # No suitable filtered FASTA (>1 seq) found; skip aggregation for this family
            pass
    if len(rows) > 0:
        df = pd.DataFrame(rows)
        print(f"Processed {parquet_file} with {len(df)} rows")
        print("mean seqs per fam: ", df["n_seqs"].mean())
        print("min seqs per fam: ", df["n_seqs"].min())
        print("max seqs per fam: ", df["n_seqs"].max())
        print("\n\n")
    return rows


def aggregate_fasta_files(parquet_paths):
    output_dir  = "../data/val_test_v2_fastas"
    for parquet_path in parquet_paths:
        
        dirname = os.path.dirname(parquet_path)
        ds_name = parquet_path.split("/")[2]
        savedir = os.path.join(output_dir, ds_name)
        os.makedirs(savedir, exist_ok=True)
        for fasta_file in glob.glob(os.path.join(dirname, "*.fasta")):
            shutil.copy(fasta_file, savedir)


def main():
    parquet_paths = [
        "../data/foldseek/foldseek_s100_raw/train_test_split_v2/val/foldseek_s100_raw_val.parquet",
        "../data/foldseek/foldseek_s100_raw/train_test_split_v2/test/foldseek_s100_raw_test.parquet",
        "../data/funfams/s50_parquets/train_test_split_v2/val/val_000.parquet",
        "../data/funfams/s50_parquets/train_test_split_v2/test/test_000.parquet",
        "../data/pfam/train_test_split_parquets/val_clustered_split_combined.parquet",
        "../data/pfam/train_test_split_parquets/test_clustered_split_combined.parquet",
    ]
    all_rows = []
    for parquet_file in parquet_paths:
        rows = make_fasta_files(parquet_file)
        all_rows.extend(rows)
    df = pd.DataFrame(all_rows)
    df.to_csv("fasta_files_info.csv", index=False)

if __name__ == "__main__":
    main()