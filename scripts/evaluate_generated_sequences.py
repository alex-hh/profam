"""
Created by Jude Wells 2025-09-12

This script evaluates generated sequences from a given model.
We assume that you have already generated the sequences and they are in fasta format.
We assume that you have the target structures (representative from each cluster)
"""

import glob
import os

from src.utils.evaluation_utils import sequence_only_evaluation
import pandas as pd

def evaluate_generated_sequences():
    all_results = []
    generated_fasta_pattern = "../sampling_results/foldseek_*/*/*.fasta"
    for generated_fasta in glob.glob(generated_fasta_pattern):
        csv_save_path = os.path.dirname(generated_fasta) + "/sequence_only_evaluation.csv"
        split  = "val" if "val" in generated_fasta else "test"
        fname = os.path.basename(generated_fasta).replace("_generated.fasta", ".fasta")
        prompt_fasta = f"../data/val_test_v2_fastas/foldseek/{split}/{fname}"
        if not os.path.exists(prompt_fasta):
            print(f"Prompt FASTA not found for {generated_fasta}")
            continue
        results = sequence_only_evaluation(prompt_fasta, generated_fasta, generate_logos=False)
        all_results.append(results)
        df = pd.DataFrame(all_results)
        df.to_csv(csv_save_path, index=False)
        print(results)

    pass

if __name__ == "__main__":
    evaluate_generated_sequences()
    generated_pdb_pattern = "../sampling_results/colabfold_outputs/foldseek_*/gen0_unrelaxed_rank_001_alphafold2_ptm_model_1_seed_000.pdb"
    gt_pdb_pattern = "../data/val_test_v2_pdbs/foldseek/*.pdb"
    generated_pdbs = glob.glob(generated_pdb_pattern)
    gt_pdbs = glob.glob(gt_pdb_pattern)
    for generated_pdb in generated_pdbs:
        generated_id = generated_pdb.split("/")[-2].split("_")[-1]
        gt_pdb = next((p for p in gt_pdbs if generated_id in p), None)
        if gt_pdb is None:
            print(f"No GT PDB found for {generated_id}")
            continue
        print(f"Evaluating {generated_id}")
        print(f"GT PDB: {gt_pdb}")
        print(f"Generated PDB: {generated_pdb}")