import glob
import os
import shutil
import subprocess
import tempfile
from Bio import SeqIO
import numpy as np
import random
from typing import Dict, List, Optional, Tuple

"""
Created by Jude Wells 2025-09-26
Purpose of the script is to create reasonably homogenous EC families
so that we can compare the alignment statistics of the family and condition on the family.

We want to keep the sequences that belong to a cov>70% and pid>30% cluster.
We always take the largest cluster.

1) generate an all against all alignment using MAFFT
2) run HHfilter to remove sequences with >98% identity and <70% coverage


We discard cases where there are fewer than 50 sequences in the cluster so that we have
robust alignment statistics.

Then for each family randomly sample sequences up to the maximum context length: this will be
the fasta file which we condition ProFam on.
The remaining sequences will be used for evaluation.
"""

HHFILTER_BINARY = os.environ.get("HHFILTER_BINARY", "/mnt/disk2/msa_pairformer/hhsuite/hhfilter")


def _cluster_with_mmseqs(seqs: List[str], min_seq_id: float = 0.3, coverage: float = 0.7, threads: int = 1) -> Dict[int, int]:
    mapping: Dict[int, int] = {}
    mmseqs_bin = shutil.which("mmseqs")
    if mmseqs_bin is None:
        return mapping  # empty indicates no clustering available
    with tempfile.TemporaryDirectory() as tmpdir:
        fasta_path = os.path.join(tmpdir, "input.fasta")
        with open(fasta_path, "w") as f:
            for i, s in enumerate(seqs):
                f.write(f">s{i}\n{s}\n")
        out_prefix = os.path.join(tmpdir, "cluster")
        cmd = [
            mmseqs_bin, "easy-cluster",
            fasta_path,
            out_prefix,
            out_prefix,
            "--min-seq-id", str(float(min_seq_id)),
            "-c", str(float(coverage)),
            "--threads", str(int(threads)),
            "--remove-tmp-files", "1",
            "--cluster-mode", "1",
        ]
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except subprocess.CalledProcessError:
            return {}
        cluster_tsv = f"{out_prefix}_cluster.tsv"
        if not os.path.exists(cluster_tsv):
            return {}
        rep_to_cid: Dict[str, int] = {}
        next_cid = 0
        with open(cluster_tsv, "r") as fr:
            for line in fr:
                parts = line.strip().split("\t")
                if len(parts) < 2:
                    continue
                rep, mem = parts[0], parts[1]
                if rep not in rep_to_cid:
                    rep_to_cid[rep] = next_cid
                    next_cid += 1
                cid = rep_to_cid[rep]
                if mem.startswith("s"):
                    try:
                        idx = int(mem[1:])
                    except Exception:
                        continue
                    mapping[idx] = cid
                # also map representative itself if present as rep id style "sX"
                if rep.startswith("s"):
                    try:
                        idx_r = int(rep[1:])
                        mapping.setdefault(idx_r, cid)
                    except Exception:
                        pass
        # Any sequences not present in mapping -> singleton clusters
        for i in range(len(seqs)):
            mapping.setdefault(i, next_cid)
            if mapping[i] == next_cid:
                next_cid += 1
        return mapping

def _run(cmd):
    print(f"[cmd] {cmd}")
    return subprocess.run(cmd, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)



def align_with_mafft(input_fasta, output_aln_fasta, threads=12):
    """
    Align all sequences with MAFFT.

    Command pattern:
      mafft --thread N input.fasta > output.fasta
    """
    records = list(SeqIO.parse(input_fasta, "fasta"))
    if len(records) == 0:
        raise ValueError(f"No sequences found in {input_fasta}")

    os.makedirs(os.path.dirname(output_aln_fasta), exist_ok=True)


    cmd = (
        f"mafft --thread {int(threads)} {input_fasta} > {output_aln_fasta}"
    )
    _run(cmd)
    return output_aln_fasta


def run_hhfilter(input_path, output_path):
    """
    Run HHfilter with default parameters on an aligned FASTA (AFA).
    We explicitly set -M afa to treat the input/output as aligned FASTA.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cmd = f"{HHFILTER_BINARY} -i {input_path} -o {output_path} -M afa"
    _run(cmd)
    return output_path


def write_one_line_fasta(record, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write(f">{record.id}\n")
        f.write(str(record.seq).replace("\n", "").replace("\r", "") + "\n")
    return output_path


def write_many_one_line_fastas(records, output_path, strip_gaps=True):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        for r in records:
            seq_str = str(r.seq)
            if strip_gaps:
                seq_str = seq_str.replace("-", "")
            seq_str = seq_str.replace("\n", "").replace("\r", "")
            f.write(f">{r.id}\n")
            f.write(seq_str + "\n")
    return output_path



if __name__ == "__main__":
    # Inputs/outputs
    ec_pattern = "../data/ec/ec_fastas/*"
    output_dir = "../data/ec/ec_validation_dataset_clustered_c70_pid_30"
    align_dir = os.path.join(output_dir, "alignments")
    conditioning_dir = os.path.join(output_dir, "conditioning")
    eval_dir = os.path.join(output_dir, "eval")
    os.makedirs(align_dir, exist_ok=True)
    os.makedirs(conditioning_dir, exist_ok=True)
    os.makedirs(eval_dir, exist_ok=True)

    # Restrict to ECs that exist in previous validation dataset
    prev_single_seq_dir = os.path.join("../data/ec/ec_validation_dataset", "ec_single_sequences")
    prev_single_seq_paths = glob.glob(os.path.join(prev_single_seq_dir, "*.fasta"))
    ec_codes = {os.path.basename(p).split(".")[0] for p in prev_single_seq_paths}

    # Params
    min_seqs_for_val = int(os.environ.get("MIN_SEQS_FOR_VAL", "50"))
    threads = int(os.environ.get("MAFFT_THREADS", "12"))
    mmseqs_min_seq_id = float(os.environ.get("MMSEQS_MIN_SEQ_ID", "0.3"))
    mmseqs_coverage = float(os.environ.get("MMSEQS_COVERAGE", "0.7"))
    mmseqs_threads = int(os.environ.get("MMSEQS_THREADS", str(max(1, threads // 2))))
    rng_seed = int(os.environ.get("RNG_SEED", "42"))
    random.seed(rng_seed)

    ec_paths = glob.glob(ec_pattern)
    for ec_path in ec_paths:
        try:
            ec_id = os.path.basename(ec_path).split(".")[0]

            if ec_codes and ec_id not in ec_codes:
                print(f"Skipping {ec_id}: not in previous validation dataset")
                continue

            all_records = list(SeqIO.parse(ec_path, "fasta"))
            if len(all_records) == 0:
                print(f"Skipping {ec_id}: no sequences found")
                continue

            # Cluster with MMseqs2 and keep largest cluster
            seqs = [str(r.seq).replace("\n", "").replace("\r", "") for r in all_records]
            cluster_map = _cluster_with_mmseqs(
                seqs, min_seq_id=mmseqs_min_seq_id, coverage=mmseqs_coverage, threads=mmseqs_threads
            )

            if not cluster_map:
                print(f"MMseqs2 not available or failed; using all sequences for {ec_id}")
                cluster_records = all_records
            else:
                counts = {}
                for idx, cid in cluster_map.items():
                    counts[cid] = counts.get(cid, 0) + 1
                largest_cid = max(counts.items(), key=lambda kv: kv[1])[0]
                cluster_records = [rec for i, rec in enumerate(all_records) if cluster_map.get(i, -1) == largest_cid]

            if len(cluster_records) < min_seqs_for_val:
                print(f"Skipping {ec_id}: largest cluster has {len(cluster_records)} sequences (< {min_seqs_for_val})")
                for p in [
                    os.path.join(align_dir, f"{ec_id}_cluster_aln.fasta"),
                    os.path.join(align_dir, f"{ec_id}_cluster_aln.filtered.fasta"),
                    os.path.join(conditioning_dir, f"{ec_id}.fasta"),
                    os.path.join(eval_dir, f"{ec_id}.fasta"),
                ]:
                    try:
                        if os.path.exists(p):
                            os.remove(p)
                    except OSError:
                        pass
                continue

            # Write cluster to temp fasta for alignment
            with tempfile.TemporaryDirectory(prefix="ec_cluster_") as tmpdir:
                cluster_fa = os.path.join(tmpdir, f"{ec_id}_cluster.fasta")
                SeqIO.write(cluster_records, cluster_fa, "fasta")

                # Align to first with MAFFT
                msa_path = os.path.join(align_dir, f"{ec_id}_cluster_aln.fasta")
                if not os.path.exists(msa_path):
                    try:
                        align_with_mafft(cluster_fa, msa_path, threads=threads)
                    except subprocess.CalledProcessError as e:
                        print(e.stdout)
                        print(e.stderr)
                        print(f"Error aligning {ec_id}")
                        continue
                    except Exception as e:
                        print(f"Error aligning {ec_id}: {e}")
                        continue

            # Run HHfilter on aligned FASTA
            filtered_msa_path = os.path.join(align_dir, f"{ec_id}_cluster_aln.filtered.fasta")
            if not os.path.exists(filtered_msa_path):
                try:
                    run_hhfilter(msa_path, filtered_msa_path)
                except subprocess.CalledProcessError as e:
                    print(e.stdout)
                    print(e.stderr)
                    print(f"Error running hhfilter for {ec_id}")
                    try:
                        if os.path.exists(msa_path):
                            os.remove(msa_path)
                        if os.path.exists(filtered_msa_path):
                            os.remove(filtered_msa_path)
                    except OSError:
                        pass
                    continue
                except Exception as e:
                    print(f"Error running hhfilter for {ec_id}: {e}")
                    try:
                        if os.path.exists(msa_path):
                            os.remove(msa_path)
                        if os.path.exists(filtered_msa_path):
                            os.remove(filtered_msa_path)
                    except OSError:
                        pass
                    continue

            # Count sequences remaining after filtering
            try:
                remaining = list(SeqIO.parse(filtered_msa_path, "fasta"))
                num_remaining = len(remaining)
            except Exception as e:
                print(f"Error reading filtered MSA for {ec_id}: {e}")
                num_remaining = 0

            if num_remaining < min_seqs_for_val:
                print(
                    f"Discarding {ec_id}: {num_remaining} sequences after hhfilter (< {min_seqs_for_val}). Deleting files."
                )
                for p in [
                    msa_path,
                    filtered_msa_path,
                    os.path.join(conditioning_dir, f"{ec_id}.fasta"),
                    os.path.join(eval_dir, f"{ec_id}.fasta"),
                ]:
                    try:
                        if os.path.exists(p):
                            os.remove(p)
                    except OSError:
                        pass
                continue

            # Split into conditioning and evaluation sets using token budgeting
            # Compute ungapped lengths for remaining sequences
            ungapped_lengths = []
            for r in remaining:
                seq_str = str(r.seq).replace("-", "").replace("\n", "").replace("\r", "")
                ungapped_lengths.append(len(seq_str))

            # Identify the longest ungapped sequence among the filtered cluster
            longest_idx = max(range(num_remaining), key=lambda i: ungapped_lengths[i])
            longest_len = ungapped_lengths[longest_idx]

            # Token budget parameters
            total_tokens = 8192
            per_token_multiplier = 1.2
            conditioning_budget = total_tokens - longest_len * per_token_multiplier

            # If no budget remains, place everything into eval
            if conditioning_budget <= 0:
                continue
            else:
                # Iteratively add conditioning sequences (excluding the longest) until budget is exhausted
                candidate_indices = np.array(range(num_remaining))
                random.shuffle(candidate_indices)
                cond_idx = set()
                used_tokens = 0.0
                for i in candidate_indices:
                    cost = ungapped_lengths[i]
                    if used_tokens + cost <= conditioning_budget:
                        cond_idx.add(i)
                        used_tokens += cost

            cond_records = [remaining[i] for i in sorted(cond_idx)]
            eval_records = [remaining[i] for i in range(num_remaining) if i not in cond_idx]

            cond_out = os.path.join(conditioning_dir, f"{ec_id}.fasta")
            eval_out = os.path.join(eval_dir, f"{ec_id}.fasta")

            try:
                write_many_one_line_fastas(cond_records, cond_out, strip_gaps=True)
                write_many_one_line_fastas(eval_records, eval_out, strip_gaps=True)
                print(
                    f"Kept {ec_id}: {num_remaining} sequences after hhfilter. Wrote {len(cond_records)} conditioning and {len(eval_records)} eval sequences."
                )
            except Exception as e:
                print(f"Error writing output FASTAs for {ec_id}: {e}")
                continue
        except Exception as e:
            print(f"Unexpected error for {ec_path}: {e}")



