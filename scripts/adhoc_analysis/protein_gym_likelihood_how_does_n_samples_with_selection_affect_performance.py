import pandas as pd
import numpy as np
import glob
import os
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

"""
Created by Jude Wells 2025-08-30
Answers a different question to:
scripts/adhoc_analysis/protein_gym_likelihood_n_samples_plot copy.py

In the former script we always selected the top-scoring from all 200 forward passes
In this script we ask the question: what if we had n < 200 forward passes to select from?
This script should inform us how much benefit we get from increasing the number of forward passes.
"""

def load_dms_scores(csv_path, seed=42, max_mutated_sequences=3000):
    dms_df = pd.read_csv(csv_path)
    if max_mutated_sequences is not None and max_mutated_sequences < len(dms_df):
        dms_df = dms_df.sample(n=max_mutated_sequences, random_state=seed)
    return dms_df

def threshold_lls(lls, lls_mean, threshold, top_k):
    mask = lls_mean < threshold
    if np.sum(mask) == 0:
        lowest_10_indices = np.argsort(lls_mean)[:10]
        selected = lls[lowest_10_indices]
        selected_means = lls_mean[lowest_10_indices]
    else:
        selected = lls[mask]
        selected_means = lls_mean[mask]
    if top_k < len(selected):
        top_k_indices = np.argsort(-selected_means)[:top_k]
        selected = selected[top_k_indices]
    return selected


if __name__ == "__main__":
    # npz_files = glob.glob("logs/abyoeovl_openfold_fs50_ur90_memmap_251m_copied_2025-06-23_22-18/20250726_173620/*.npz")
    # npz_files = glob.glob("logs/abyoeovl_openfold_fs50_ur90_memmap_251m_copied_2025-06-23_22-18/20250730_183304_100_reps/*.npz")
    # npz_files = glob.glob("logs/proteingym_eval_results/20250808_000300_PoET_MSAs_BO/*.npz")
    # npz_files = glob.glob("logs/proteingym_eval_results/20250810_135739_poet_msas_random_sampling_v4/*.npz")
    # npz_files = glob.glob("logs/abyoeovl_openfold_fs50_ur90_memmap_251m_copied_2025-06-23_22-18/20250810_135739_full_gym/*.npz")
    npz_files = glob.glob("logs/abyoeovl_openfold_fs50_ur90_memmap_251m_copied_2025-06-23_22-18/20250810_135739_v6_full_gym/*v6_lls.npz")
    # npz_files = glob.glob("logs/abyoeovl_openfold_fs50_ur90_memmap_251m_copied_2025-06-23_22-18/20250828_v6_gym_msas/*v6_lls.npz")
    # npz_files = glob.glob("logs/abyoeovl_openfold_fs50_ur90_memmap_251m_copied_2025-06-23_22-18/20250829_v6_gym_msas_filtered/*v6_lls.npz")
    save_dir = "../v6_gym_msas_evaluating_n_forward"
    os.makedirs(save_dir, exist_ok=True)
    results_rows = []
    target_score = -1.3
    for n_forward in [10, 20, 50, 100, 200]:
        for top_pct in [1.0, 0.7, 0.5, 0.3, 0.2]:
            spearman_scores = []
            print(f"Found {len(npz_files)} npz files")
            n_select = int(n_forward * top_pct)
            print(f"Processing {n_select} samples")
            one_value_spearman_corrs = []
            one_value_spearman_corrs_sorted = []
            for npz_file in npz_files:
                data = np.load(npz_file)
                dms_scores = data["dms_scores"]
                assert len(dms_scores) == data["lls"].shape[1], "Number of lls and dms scores must match"
                if data["lls"].shape[0] < 200:
                    print(f"Skipping {npz_file} because it has {data['lls'].shape[0]} reps")
                    continue
                lls_to_use = data["lls"][:n_forward]
                # sort the lls by the mean likelihood
                mutant_means = lls_to_use.mean(axis=1)
                lls_distances = np.abs(mutant_means - target_score)
                lls_mean_sorted_indices = np.argsort(lls_distances)
                lls_sorted = lls_to_use[lls_mean_sorted_indices]
                lls_to_use_sorted = lls_sorted[:n_select]
                lls_mean_sorted = lls_to_use_sorted.mean(axis=0)
                spearman_score = spearmanr(lls_mean_sorted, dms_scores)[0]
                spearman_scores.append(spearman_score)
            mean_spearman_score = np.mean(spearman_scores)
            new_row = {
                "n_forward": n_forward,
                "top_pct": top_pct,
                "n_select": n_select,
                "target_score": target_score,
                "spearman_score": mean_spearman_score,
            }
            results_rows.append(new_row)
    results_df = pd.DataFrame(results_rows)
    print(results_df.groupby("n_forward").max())
    # results_df.to_csv(f"{save_dir}/protein_gym_likelihood_how_does_n_samples_with_selection_affect_performance.csv", index=False)
                



