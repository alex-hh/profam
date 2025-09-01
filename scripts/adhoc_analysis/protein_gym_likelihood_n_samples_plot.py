import pandas as pd
import numpy as np
import glob
import os
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

"""
Created by Jude Wells 2025-08-04
Generates plots showing the number of ensembed predictions and 
the spearman correlation.
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
    # npz_files = glob.glob("logs/abyoeovl_openfold_fs50_ur90_memmap_251m_copied_2025-06-23_22-18/20250810_135739_v6_full_gym/*v6_lls.npz")
    # npz_files = glob.glob("logs/abyoeovl_openfold_fs50_ur90_memmap_251m_copied_2025-06-23_22-18/20250828_v6_gym_msas/*v6_lls.npz")
    # npz_files = glob.glob("logs/abyoeovl_openfold_fs50_ur90_memmap_251m_copied_2025-06-23_22-18/20250829_v6_gym_msas_filtered/*v6_lls.npz")
    npz_files = glob.glob("logs/abyoeovl_openfold_fs50_ur90_memmap_251m_copied_2025-06-23_22-18/20250829_msa_pariformer_exp_2_gym_msas_unfiltered/*v7_lls.npz")
    npz_files = glob.glob("filtered_poet_msas_no_weighting_v9/20250831_154342/*.npz")
    npz_files = glob.glob("filtered_poet_msas_with_weighting_v9/20250831_154342/*.npz")
    npz_files = glob.glob("filtered_poet_msas_with_no_weighting_v9/20250831_211318/*.npz")
    npz_files = glob.glob("filtered_poet_msas_with_only_diversity_weighting_v9/20250831_221004/*.npz")
    save_dir = "gym_spearman_likelihood_plots_subsample_filtered_poet_msas_diversity_weighting_abyoeovl"
    os.makedirs(save_dir, exist_ok=True)
    target_scores = [-1.1, -1.2, -1.3, -1.4, -1.5, -1.6, -1.7]
    for target_score in target_scores:
        print(f"Found {len(npz_files)} npz files")
        results_rows = []
        completed_dms_ids = []
        samps_used = []
        spearman_corrs = []
        sorted_spearman_corrs = []
        # for i in [1, 2, 5, 10, 20, 30, 50, 80, 120, 160, 200]:
        for i in [1, 2, 5, 10, 20, 30, 50]:
            print(f"Processing {i} samples")
            one_value_spearman_corrs = []
            one_value_spearman_corrs_sorted = []
            for npz_file in npz_files:
                # npz_file = npz_file.replace("20250829_msa_pariformer_exp_2_gym_msas_unfiltered", "20250828_v6_gym_msas").replace("v7_lls.npz", "v6_lls.npz")
                data = np.load(npz_file)
                dms_scores = data["dms_scores"]
                assert len(dms_scores) == data["lls"].shape[1], "Number of lls and dms scores must match"
                if data["lls"].shape[0] < 50:
                    print(f"Skipping {npz_file} because it has {data['lls'].shape[0]} reps")
                    continue
                lls_to_use = data["lls"][:i]
                lls_mean = lls_to_use.mean(axis=0)
                one_value_spearman_corrs.append(spearmanr(lls_mean, dms_scores)[0])
                # sort the lls by the mean likelihood
                mutant_means = data["lls"].mean(axis=1)
                lls_distances = np.abs(mutant_means - target_score)
                lls_mean_sorted_indices = np.argsort(lls_distances)
                lls_sorted = data["lls"][lls_mean_sorted_indices]
                lls_to_use_sorted = lls_sorted[:i]

                lls_mean_sorted = lls_to_use_sorted.mean(axis=0)
                one_value_spearman_corrs_sorted.append(spearmanr(lls_mean_sorted, dms_scores)[0])

            samps_used.append(i)
            spearman_corrs.append(np.mean(one_value_spearman_corrs))
            sorted_spearman_corrs.append(np.mean(one_value_spearman_corrs_sorted))

        plt.plot(samps_used, sorted_spearman_corrs, label=f"Sorted {target_score}")
        if target_score == target_scores[0]:
            plt.plot(samps_used, spearman_corrs, label="Unsorted", color="blue")
        plt.xlabel("Number of samples used")
        plt.ylabel("Spearman correlation")
        plt.title("Spearman correlation vs. number of samples used")
        plt.legend()
        plt.savefig(f"{save_dir}/protein_gym_likelihood_n_samples_plot_{target_score}.png")



