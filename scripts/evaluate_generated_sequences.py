"""
Created by Jude Wells 2025-09-12

This script evaluates generated sequences from a given model.
We assume that you have already generated the sequences and they are in fasta format.
We assume that you have the target structures (representative from each cluster)
"""

import glob
import os
from Bio import SeqIO
from src.utils.evaluation_utils import sequence_only_evaluation
import pandas as pd
import numpy as np
from src.data.objects import Protein
from src.structure.superimposition import tm_score, lddt
from src.utils.evaluation_utils import pairwise_sequence_identity
from statsmodels.nonparametric.smoothers_lowess import lowess
import matplotlib.pyplot as plt

def evaluate_generated_sequences(
    generated_fasta_pattern,
    csv_save_path,
):
    all_results = []
    os.makedirs(os.path.dirname(csv_save_path), exist_ok=True)
    fasta_paths = glob.glob(generated_fasta_pattern)
    print(f"Found {len(fasta_paths)} generated fasta files")
    if len(fasta_paths) == 0:
        raise FileNotFoundError(f"No generated fasta files found for glob: {generated_fasta_pattern}")
    for generated_fasta in fasta_paths:
        split  = "val" if "val" in generated_fasta else "test"
        fname = os.path.basename(generated_fasta).replace("_generated.fasta", ".fasta")
        prompt_fasta = glob.glob(f"../data/val_test_v2_fastas/foldseek/*/{fname}")[0]
        if not os.path.exists(prompt_fasta):
            print(f"Prompt FASTA not found for {generated_fasta}")
            continue
        results = sequence_only_evaluation(prompt_fasta, generated_fasta, generate_logos=False)
        all_results.append(results)
        df = pd.DataFrame(all_results)
        df.to_csv(csv_save_path, index=False)
        print(results)

def evaluate_generated_sequences_poet():
    all_results = []
    generated_fasta_pattern = "../sampling_results/poet/poet_foldseek_*/generated_sequences_foldseek_*/*_seed42.fasta"
    csv_save_path = "../sampling_results/poet/poet_sequence_only_evaluation.csv"
    for generated_fasta in glob.glob(generated_fasta_pattern):
        split  = "val" if "val" in generated_fasta else "test"
        fname = os.path.basename(generated_fasta).replace("_samples20_seed42", "")
        prompt_fasta = f"../data/val_test_v2_fastas/foldseek/{split}/{fname}"
        if not os.path.exists(prompt_fasta):
            print(f"Prompt FASTA not found for {generated_fasta}")
            continue
        results = sequence_only_evaluation(prompt_fasta, generated_fasta, generate_logos=False)
        all_results.append(results)
        df = pd.DataFrame(all_results)
        df.to_csv(csv_save_path, index=False)
        print(results)

def evaluate_generated_sequences_poet_on_ec_single_sequence():
    all_results = []
    generated_fasta_pattern = "../sampling_results/poet/poet_ec_single_seq_synthetic_msas/*/*_samples50_seed42.fasta"
    csv_save_path = "../sampling_results/poet/poet_ec_single_seq_synthetic_msas/poet_sequence_only_evaluation_ec_single_sequence.csv"
    generated_fasta_paths = glob.glob(generated_fasta_pattern)
    print(f"Found {len(generated_fasta_paths)} generated fasta files")
    if len(generated_fasta_paths) == 0:
        raise FileNotFoundError(f"No generated fasta files found for glob: {generated_fasta_pattern}")
    for generated_fasta in glob.glob(generated_fasta_pattern):
        ec_num = os.path.basename(generated_fasta).split("_samples50_seed42")[0]
        prompt_fasta = f"../data/ec/ec_validation_dataset/alignments/{ec_num}_aln.filtered.fasta"
        if not os.path.exists(prompt_fasta):
            print(f"Prompt FASTA not found for {generated_fasta}")
            continue
        results = sequence_only_evaluation(prompt_fasta, generated_fasta, generate_logos=False)
        all_results.append(results)
        df = pd.DataFrame(all_results)
        df.to_csv(csv_save_path, index=False)
        print(results)


def evaluate_generated_sequences_profam_on_ec_single_sequence():
    all_results = []
    completed_ec_nums = []
    
    # generated_fasta_pattern = "../sampling_results/profam_ec_single_seq_synthetic_msas/*_generated.fasta"
    # csv_save_path = "../sampling_results/profam_ec_single_seq_synthetic_msas/profam_sequence_only_evaluation_ec_single_sequence.csv"

    # generated_fasta_pattern = "../sampling_results/profam_ec_multi_seq_synthetic_msas_no_ensemble/*_generated.fasta"
    # csv_save_path = "../sampling_results/profam_ec_multi_seq_synthetic_msas_no_ensemble/profam_sequence_only_evaluation_ec_multi_sequence.csv"

    generated_fasta_pattern = "../sampling_results/profam_ec_multi_seq_synthetic_msas/*_generated.fasta"
    csv_save_path = "../sampling_results/profam_ec_multi_seq_synthetic_msas/profam_sequence_only_evaluation_ec_multi_sequence_with_ensemble.csv"

    if os.path.exists(csv_save_path):
        df = pd.read_csv(csv_save_path)
        print(f"Found {len(df)} rows in {csv_save_path}")
        all_results = df.to_dict(orient="records")
        completed_ec_nums = [os.path.basename(r['aligned_generation_path']).split("_generated")[0] for r in all_results]
    generated_fasta_paths = glob.glob(generated_fasta_pattern)
    print(f"Found {len(generated_fasta_paths)} generated fasta files")
    if len(generated_fasta_paths) == 0:
        raise FileNotFoundError(f"No generated fasta files found for glob: {generated_fasta_pattern}")
    for i, generated_fasta in enumerate(generated_fasta_paths):
        print(f"Processing {i} of {len(generated_fasta_paths)}")
        ec_num = os.path.basename(generated_fasta).split("_generated")[0].replace("_aln.filtered", "")
        if ec_num in completed_ec_nums:
            print(f"Skipping {ec_num} because it already exists in {csv_save_path}")
            continue
        prompt_fasta = f"../data/ec/ec_validation_dataset/alignments/{ec_num}_aln.filtered.fasta"
        if not os.path.exists(prompt_fasta):
            print(f"Prompt FASTA not found for {generated_fasta}")
            continue
        results = sequence_only_evaluation(prompt_fasta, generated_fasta, generate_logos=False)
        all_results.append(results)
        df = pd.DataFrame(all_results)
        df.to_csv(csv_save_path, index=False)
        completed_ec_nums.append(ec_num)


def get_pdb_paths_from_fasta_path(fasta_path, gt_pdbs):
    prompt_records = list(SeqIO.parse(fasta_path, "fasta"))
    prompt_ids = [record.id for record in prompt_records]
    pdb_paths = [p for p in gt_pdbs if any(pid in p for pid in prompt_ids)]
    return pdb_paths

def make_structure_sequence_similarity_plots(csv_path):
    df = pd.read_csv(csv_path)
    structure_metrics = ['tm_max', 'lddt_max']
    for structure_metric in structure_metrics:
        x = df["seq_identity_max"].to_numpy(dtype=float)
        y = df[structure_metric].to_numpy(dtype=float)
        mask = np.isfinite(x) & np.isfinite(y)
        x_valid = x[mask]
        y_valid = y[mask]

        if x_valid.size == 0:
            continue

        plt.scatter(x_valid, y_valid, s=12, alpha=0.5, label="samples")

        try:
            smoothed = lowess(y_valid, x_valid, frac=0.5, return_sorted=True)
            plt.plot(smoothed[:, 0], smoothed[:, 1], color="crimson", linewidth=2, label="LOWESS")
        except Exception:
            pass

        plt.xlabel("Max sequence identity prompt")
        plt.ylabel(structure_metric)
        plt.legend(frameon=False)
        plt.tight_layout()
        plt.savefig(f"{csv_path.replace('.csv', f'_{structure_metric}.png')}")
        plt.close()
        



if __name__ == "__main__":
    # generated_fasta_pattern = "../sampling_results/foldseek_*/*/*.fasta",
    # sequence_only_csv_save_path = "../sampling_results/profam_sequence_only_evaluation.csv",
    # generated_fasta_pattern = "../sampling_results/foldseek_combined_val_test_2025_09_17/*.fasta"
    # evaluate_generated_sequences_poet_on_ec_single_sequence()
    evaluate_generated_sequences_profam_on_ec_single_sequence()
    # sequence_only_csv_save_path = "../sampling_results/foldseek_combined_val_test_2025_09_17/profam_sequence_only_evaluation.csv"
    # generated_pdb_pattern = "../sampling_results/colabfold_outputs/foldseek_*/gen0_unrelaxed_rank_001_alphafold2_ptm_model_1_seed_000.pdb"
    # generated_pdb_pattern = "../sampling_results/colabfold_outputs/foldseek_combined_val_test_2025_09_17_seq_sim_lt_0p5/*/*.pdb"
    # generated_pdb_pattern = "../sampling_results/poet/poet_colabfold_outputs_seq_sim_lt_0p5/*/*unrelaxed_rank_001_alphafold2_ptm_model_1_seed_000.pdb"
    # structural_csv = "/".join(generated_pdb_pattern.split("/")[:-2]) + "/structural_evaluation.csv"
    # # structural_csv = "../sampling_results/colabfold_outputs/foldseek_combined_val_test_2025_09_17_seq_sim_lt_0p5/structural_evaluation.csv"
    # # evaluate_generated_sequences(generated_fasta_pattern, sequence_only_csv_save_path)
    # # evaluate_generated_sequences_poet()
    # # generated_pdb_pattern = "../sampling_results/randomly_mutated_sequences/random_colabfold_outputs/*/*_unrelaxed_rank_001_alphafold2_ptm_model_1_seed_000.pdb"
    # # elif "/colabfold_outputs/foldseek_" in generated_pdb_pattern:
    # #     structural_csv = "../sampling_results/colabfold_outputs/profam_structural_evaluation.csv"
    # # generated_pdb_pattern = "../sampling_results/poet_colabfold_outputs/foldseek_*/generated_9_*_ptm_model_1_seed_000.pdb"
    # gt_pdb_pattern = "../data/val_test_v2_pdbs/foldseek/*.pdb"
    # generated_pdbs = glob.glob(generated_pdb_pattern)
    # print(f"Found {len(generated_pdbs)} generated pdb files")
    # if len(generated_pdbs) == 0:
    #     raise FileNotFoundError(f"No generated pdb files found for glob: {generated_pdb_pattern}")
    # gt_pdbs = glob.glob(gt_pdb_pattern)
    # rows = []
    # if os.path.exists(structural_csv):
    #     df = pd.read_csv(structural_csv)
    #     print(f"Found {len(df)} rows in {structural_csv}")
    #     rows = df.to_dict(orient="records")
    # else:
    #     df = pd.DataFrame(columns=["generated_id"])  # Ensure df always defined
    # for i, generated_pdb in enumerate(generated_pdbs):
    #     print(f"Processing {i} of {len(generated_pdbs)}")
    #     if "_test_" in generated_pdb:
    #         split = "test"
    #     elif "_val_" in generated_pdb:
    #         split = "val"
    #     elif "_random_" in generated_pdb:
    #         split = "random"
    #     else:
    #         split = None
    #     generated_id = generated_pdb.split("/")[-2].split("_")[-1]
    #     if len(df) > 0 and generated_pdb in df["generated_pdb"].values:

    #         print(f"Skipping {generated_id} because it already exists in {structural_csv}")
    #         continue
    #     prompt_fasta_path = glob.glob(f"../data/val_test_v2_fastas/foldseek/*/{generated_id}*.fasta")[0]
    #     prompt_pdb_paths = get_pdb_paths_from_fasta_path(prompt_fasta_path, gt_pdbs)
    #     # Load generated protein and compute mean pLDDT
    #     try:
    #         gen_prot = Protein.from_pdb(generated_pdb, bfactor_is_plddt=True)
    #         mean_plddt = float(np.mean(gen_prot.plddt)) if gen_prot.plddt is not None else float("nan")
    #     except Exception:
    #         gen_prot = None
    #         mean_plddt = float("nan")

    #     # Sequence identity stats (generated vs all prompt sequences)
    #     try:
    #         prompt_records = list(SeqIO.parse(prompt_fasta_path, "fasta"))
    #         prompt_seqs = [str(r.seq).replace("-", "") for r in prompt_records]
    #         gen_seq = gen_prot.sequence if gen_prot is not None else ""
    #         seq_ids = [pairwise_sequence_identity(gen_seq, pseq) for pseq in prompt_seqs] if len(prompt_seqs) > 0 and len(gen_seq) > 0 else []
    #     except Exception:
    #         seq_ids = []

    #     def _agg_stats(values):
    #         arr = np.array(values, dtype=float)
    #         if arr.size == 0:
    #             return float("nan"), float("nan"), float("nan")
    #         return float(np.nanmin(arr)), float(np.nanmax(arr)), float(np.nanmean(arr))

    #     seq_id_min, seq_id_max, seq_id_mean = _agg_stats(seq_ids)

    #     # Structural comparisons vs all prompt PDBs
    #     tm_scores = []
    #     lddt_scores = []
    #     if gen_prot is not None:
    #         for pdb_path in prompt_pdb_paths:
    #             try:
    #                 prompt_prot = Protein.from_pdb(pdb_path, bfactor_is_plddt=True)
    #             except Exception:
    #                 continue
    #             try:
    #                 tm_val = tm_score(gen_prot, prompt_prot)
    #                 tm_scores.append(float(tm_val))
    #             except Exception:
    #                 tm_scores.append(float("nan"))
    #             try:
    #                 lddt_val = lddt(gen_prot, prompt_prot)
    #                 lddt_scores.append(float(lddt_val))
    #             except Exception:
    #                 lddt_scores.append(float("nan"))

    #     tm_min, tm_max, tm_mean = _agg_stats([v for v in tm_scores if not np.isnan(v)])
    #     lddt_min, lddt_max, lddt_mean = _agg_stats([v for v in lddt_scores if not np.isnan(v)])

    #     rows.append({
    #         "generated_id": generated_id,
    #         "split": split,
    #         "generated_pdb": generated_pdb,
    #         "prompt_fasta": prompt_fasta_path,
    #         "num_prompt_pdbs": len(prompt_pdb_paths),
    #         "mean_plddt": mean_plddt,
    #         "tm_min": tm_min,
    #         "tm_mean": tm_mean,
    #         "tm_max": tm_max,
    #         "lddt_min": lddt_min,
    #         "lddt_mean": lddt_mean,
    #         "lddt_max": lddt_max,
    #         "seq_identity_min": seq_id_min,
    #         "seq_identity_mean": seq_id_mean,
    #         "seq_identity_max": seq_id_max,
    #     })

    # # Save structural evaluation CSV aggregated across all generated 

    #     os.makedirs(os.path.dirname(structural_csv), exist_ok=True)
    #     pd.DataFrame(rows).to_csv(structural_csv, index=False)
    # make_structure_sequence_similarity_plots(structural_csv)