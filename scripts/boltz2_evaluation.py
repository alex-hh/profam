"""
Boltz2 structure prediction and evaluation script.

This script:
 - Takes two FASTA files: one with prompt sequences and one with generated sequences
 - Writes per-sequence YAML files suitable for Boltz2 under an output directory
 - Runs Boltz2 predictions (in a specified virtual environment) over that YAML directory
 - Parses predicted structures into `Protein` objects
 - Computes TM-scores and RMSD for all generated-vs-prompt pairs (on predicted structures)
 - For each generated sequence, finds the prompt sequence with maximum TM-score, then
   computes pairwise sequence similarity and coverage (via global alignment) to that prompt
 - Saves detailed results as CSV and a scatter plot of sequence similarity (x) vs TM-score (y)
   colored by coverage
"""

import argparse
import glob
import os
import subprocess
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from Bio import pairwise2

from src.data.objects import Protein
from src.sequence.fasta import read_fasta
from src.structure.superimposition import rmsd, tm_score


def read_fasta_file(fasta_path: str) -> Tuple[List[str], List[str]]:
    names, seqs = read_fasta(
        fasta_path,
        keep_insertions=False,
        keep_gaps=False,
        to_upper=True,
    )
    return names, seqs


def write_boltz_yaml(sequence_id: str, sequence: str, yaml_path: str) -> None:
    data = {
        "sequences": [
            {
                "protein": {
                    "id": sequence_id,
                    "sequence": sequence,
                    "msa": "empty",
                }
            }
        ]
    }
    with open(yaml_path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def create_yaml_dir_from_fastas(
    prompt_fasta: str,
    generated_fasta: str,
    yaml_dir: str,
) -> Tuple[List[str], List[str], Dict[str, str]]:
    os.makedirs(yaml_dir, exist_ok=True)
    prompt_names, prompt_seqs = read_fasta_file(prompt_fasta)
    gen_names, gen_seqs = read_fasta_file(generated_fasta)

    id_to_group: Dict[str, str] = {}

    for idx, seq in enumerate(prompt_seqs):
        seq_id = f"prompt_{idx}"
        id_to_group[seq_id] = "prompt"
        write_boltz_yaml(seq_id, seq, os.path.join(yaml_dir, f"{seq_id}.yaml"))

    for idx, seq in enumerate(gen_seqs):
        seq_id = f"gen_{idx}"
        id_to_group[seq_id] = "generated"
        write_boltz_yaml(seq_id, seq, os.path.join(yaml_dir, f"{seq_id}.yaml"))

    return prompt_seqs, gen_seqs, id_to_group


def run_boltz_predictions(
    yaml_dir: str,
    out_dir: str,
    boltz_venv_dir: str,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    activate_path = os.path.join(boltz_venv_dir, "bin/activate")
    cmd = (
        f"bash -lc 'set -euo pipefail; source {activate_path} && "
        f"boltz predict {yaml_dir} --use_msa_server --out_dir {out_dir} --model boltz2'"
    )
    subprocess.run(cmd, shell=True, check=True)


def load_predicted_proteins_from_dir(pred_dir: str) -> Dict[str, Protein]:
    proteins: Dict[str, Protein] = {}
    # Collect all PDBs; Boltz2 typically writes PDBs per sequence id
    pdb_paths = glob.glob(os.path.join(pred_dir, "**", "*.pdb"), recursive=True)
    for pdb_path in pdb_paths:
        base = os.path.basename(pdb_path)
        protein_id = os.path.splitext(base)[0]
        try:
            prot = Protein.from_pdb(pdb_path, bfactor_is_plddt=True)
        except Exception:
            # Retry without assuming B-factors are pLDDT
            prot = Protein.from_pdb(pdb_path, bfactor_is_plddt=False)
        proteins[protein_id] = prot
    return proteins


def global_align(seq_a: str, seq_b: str) -> Tuple[str, str]:
    # Use identity matrix scoring; returns first best alignment
    alignments = pairwise2.align.globalxx(seq_a, seq_b)
    if len(alignments) == 0:
        return seq_a, seq_b
    a1, a2, _, _, _ = alignments[0]
    return a1, a2


def compute_seq_similarity_and_coverage(seq_a: str, seq_b: str) -> Tuple[float, float]:
    a1, a2 = global_align(seq_a, seq_b)
    # Similarity: matches over max length of ungapped sequences
    num_matches = sum(1 for x, y in zip(a1, a2) if x == y and x != "-")
    denom = max(len(seq_a), len(seq_b))
    seq_sim = (num_matches / denom) if denom > 0 else 0.0
    # Coverage: non-gap characters in a2 (prompt-aligned) over len(prompt)
    # Here, define coverage w.r.t. prompt seq (seq_b)
    cov = (sum(1 for ch in a2 if ch != "-") / len(seq_b)) if len(seq_b) > 0 else 0.0
    return seq_sim, cov


def evaluate_structures(
    predicted: Dict[str, Protein],
    prompt_seqs: List[str],
    gen_seqs: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Partition proteins by id prefix if available; else map by sequence match
    prompt_keys = [k for k in predicted.keys() if k.startswith("prompt_")]
    gen_keys = [k for k in predicted.keys() if k.startswith("gen_")]

    # Fallback mapping if IDs were not preserved in filenames
    if len(prompt_keys) == 0 or len(gen_keys) == 0:
        # Build maps by exact sequence match
        prompt_seq_set = set(prompt_seqs)
        gen_seq_set = set(gen_seqs)
        for k, prot in list(predicted.items()):
            if k in prompt_keys or k in gen_keys:
                continue
            if prot.sequence in prompt_seq_set:
                prompt_keys.append(k)
            elif prot.sequence in gen_seq_set:
                gen_keys.append(k)

    prompt_prots = [predicted[k] for k in prompt_keys]
    gen_prots = [predicted[k] for k in gen_keys]

    # Build pairwise metrics
    rows = []
    for gi, gprot in enumerate(gen_prots):
        tm_list = []
        rmsd_list = []
        for pi, pprot in enumerate(prompt_prots):
            try:
                tm = tm_score(gprot, pprot)
            except Exception:
                tm = np.nan
            try:
                r = rmsd(pprot, gprot) if len(pprot) == len(gprot) else np.nan
            except Exception:
                r = np.nan
            tm_list.append(tm)
            rmsd_list.append(r)
            rows.append(
                {
                    "gen_id": gen_keys[gi],
                    "prompt_id": prompt_keys[pi],
                    "gen_len": len(gprot),
                    "prompt_len": len(pprot),
                    "tm_score": tm,
                    "rmsd": r,
                    "gen_plddt": float(np.nanmean(gprot.plddt)) if gprot.plddt is not None else np.nan,
                    "prompt_plddt": float(np.nanmean(pprot.plddt)) if pprot.plddt is not None else np.nan,
                }
            )

    pairwise_df = pd.DataFrame(rows)

    # For each generated protein, pick best prompt by TM-score
    best_rows = []
    if len(pairwise_df) > 0:
        grouped = pairwise_df.groupby("gen_id")
        for gen_id, df in grouped:
            best_idx = df["tm_score"].astype(float).idxmax()
            best = df.loc[best_idx]
            # Compute sequence similarity and coverage between the unaligned sequences
            gen_seq = next((p.sequence for k, p in predicted.items() if k == gen_id), None)
            prompt_seq = next((p.sequence for k, p in predicted.items() if k == best["prompt_id"]), None)
            if gen_seq is None or prompt_seq is None:
                seq_sim, cov = np.nan, np.nan
            else:
                seq_sim, cov = compute_seq_similarity_and_coverage(gen_seq, prompt_seq)
            best_rows.append(
                {
                    "gen_id": gen_id,
                    "best_prompt_id": best["prompt_id"],
                    "tm_score": float(best["tm_score"]),
                    "rmsd": float(best["rmsd"]) if not pd.isna(best["rmsd"]) else np.nan,
                    "seq_similarity": float(seq_sim) if not pd.isna(seq_sim) else np.nan,
                    "coverage": float(cov) if not pd.isna(cov) else np.nan,
                    "gen_len": int(best["gen_len"]),
                    "prompt_len": int(best["prompt_len"]),
                    "gen_plddt": float(best["gen_plddt"]) if not pd.isna(best["gen_plddt"]) else np.nan,
                    "prompt_plddt": float(best["prompt_plddt"]) if not pd.isna(best["prompt_plddt"]) else np.nan,
                }
            )
    best_df = pd.DataFrame(best_rows)
    return pairwise_df, best_df


def make_scatter(best_df: pd.DataFrame, save_path: str) -> None:
    plt.figure(figsize=(7, 5))
    sc = plt.scatter(
        best_df["seq_similarity"], best_df["tm_score"], c=best_df["coverage"], cmap="viridis", edgecolors="none"
    )
    plt.colorbar(sc, label="Coverage")
    plt.xlabel("Sequence similarity to best prompt")
    plt.ylabel("TM-score vs best prompt")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def boltz2_evaluation(
    generated_sequences_fasta_path: str,
    prompt_sequences_fasta_path: str,
    out_dir: str,
    boltz_venv_dir: str = "/mnt/disk2/boltz/venvBoltz",
) -> None:
    yaml_dir = os.path.join(out_dir, "yaml")
    pred_dir = os.path.join(out_dir, "boltz2_predictions")
    os.makedirs(out_dir, exist_ok=True)

    prompt_seqs, gen_seqs, _ = create_yaml_dir_from_fastas(
        prompt_sequences_fasta_path, generated_sequences_fasta_path, yaml_dir
    )

    run_boltz_predictions(yaml_dir, pred_dir, boltz_venv_dir)

    predicted = load_predicted_proteins_from_dir(pred_dir)
    pairwise_df, best_df = evaluate_structures(predicted, prompt_seqs, gen_seqs)

    # Save outputs
    pairwise_df.to_csv(os.path.join(out_dir, "pairwise_metrics.csv"), index=False)
    best_df.to_csv(os.path.join(out_dir, "best_prompt_per_generated.csv"), index=False)
    # Scatter plot
    if len(best_df) > 0:
        make_scatter(best_df, os.path.join(out_dir, "seqsim_vs_tm_coverage.png"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Boltz2 evaluation from FASTA inputs")
    parser.add_argument("prompt_fasta", type=str, help="Path to prompt sequences FASTA")
    parser.add_argument("generated_fasta", type=str, help="Path to generated sequences FASTA")
    parser.add_argument(
        "--out-dir",
        type=str,
        default=os.path.abspath("boltz2_eval_outputs"),
        help="Directory to write YAMLs, predictions, and results",
    )
    parser.add_argument(
        "--boltz-venv-dir",
        type=str,
        default="/mnt/disk2/boltz/venvBoltz",
        help="Path to the Boltz2 virtual environment directory (containing bin/activate)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    boltz2_evaluation(
        generated_sequences_fasta_path=os.path.abspath(args.generated_fasta),
        prompt_sequences_fasta_path=os.path.abspath(args.prompt_fasta),
        out_dir=os.path.abspath(args.out_dir),
        boltz_venv_dir=os.path.abspath(args.boltz_venv_dir),
    )