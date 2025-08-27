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
    npz_files = glob.glob("logs/abyoeovl_openfold_fs50_ur90_memmap_251m_copied_2025-06-23_22-18/20250810_135739_full_gym/*.npz")
    target_score = -1.5
    print(f"Found {len(npz_files)} npz files")
    results_rows = []
    others = pd.read_csv("/Users/judewells/Documents/dataScienceProgramming/ProteinGym/benchmarks/DMS_zero_shot/substitutions/Spearman/DMS_substitutions_Spearman_DMS_level.csv")
    completed_dms_ids = []
    samps_used = []
    spearman_corrs = []
    sorted_spearman_corrs = []
    for i in range(1, 200 + 1, 4):
        print(f"Processing {i} samples")
        one_value_spearman_corrs = []
        one_value_spearman_corrs_sorted = []
        for npz_file in npz_files:
            # npz_file = npz_file.replace("20250810_135739_poet_msas_random_sampling_v4", "20250808_000300_PoET_MSAs_BO").replace("_v4_lls.npz", "_v5.npz")
            # print(npz_file)
            # csv_path = npz_file.replace("_lls.npz", ".csv").replace(".npz", ".csv")
            # df = pd.read_csv(csv_path)
            # dms_id = df["DMS_id"].iloc[0]
            # completed_dms_ids.append(dms_id)
            # dms_scores_path = f"../data/ProteinGym/DMS_ProteinGym_substitutions/{dms_id}.csv"
            # dms_scores = load_dms_scores(dms_scores_path).DMS_score.values
            data = np.load(npz_file)
            dms_scores = data["dms_scores"]
            assert len(dms_scores) == data["lls"].shape[1], "Number of lls and dms scores must match"
            if data["lls"].shape[0] < 200:
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

    plt.plot(samps_used, sorted_spearman_corrs, label="Sorted")

    plt.plot(samps_used, spearman_corrs, label="Unsorted")
    plt.xlabel("Number of samples used")
    plt.ylabel("Spearman correlation")
    plt.title("Spearman correlation vs. number of samples used")
    plt.legend()
    plt.savefig("protein_gym_likelihood_n_samples_plot.png")



