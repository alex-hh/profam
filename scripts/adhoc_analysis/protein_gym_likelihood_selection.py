import pandas as pd
import numpy as np
import glob
import os
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

"""
For each .npz file we have lls with shape:
(replicates, n_completion_seqs)
our goal is to determine the optimal strategy

strategies:
1) average all likelihoods
2) maximum likelihood only
3) average top 10 likelihoods
4) average top 20 likelihoods
5) average top 50 likelihoods
6) average top 5 likelihoods

for each of the above named strategies:
also try removing likelihoods with a score > THRESHOLD
where threshold is -0.5, -1.0, -1.1, -1.2, -1.3, -1.4, -1.5, -1.6, -1.7

If no likelihoods are less than threshold, then take the average of the lowest 10 likelihoods
"""

strategies = [
    "average_all",
    "max_only",
    "average_top_5",
    "average_top_10",
    "average_top_20",
    "average_top_50",
]

top_ks = [99999, 100, 50, 20, 10, 5, 1]

thresholds = [0, -0.5, -1.0, -1.1, -1.2, -1.3, -1.4, -1.5, -1.6, -1.7]

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
    npz_files = glob.glob("logs/abyoeovl_openfold_fs50_ur90_memmap_251m_copied_2025-06-23_22-18/20250730_183304_100_reps/*.npz")
    print(f"Found {len(npz_files)} npz files")
    results_rows = []
    others = pd.read_csv("/Users/judewells/Documents/dataScienceProgramming/ProteinGym/benchmarks/DMS_zero_shot/substitutions/Spearman/DMS_substitutions_Spearman_DMS_level.csv")
    completed_dms_ids = []
    for npz_file in npz_files:
        print(npz_file)
        csv_path = npz_file.replace("_lls.npz", ".csv")
        df = pd.read_csv(csv_path)
        dms_id = df["DMS_id"].iloc[0]
        completed_dms_ids.append(dms_id)
        dms_scores_path = f"../data/ProteinGym/DMS_ProteinGym_substitutions/{dms_id}.csv"
        dms_scores = load_dms_scores(dms_scores_path).DMS_score.values
        data = np.load(npz_file)
        assert len(dms_scores) == data["lls"].shape[1], "Number of lls and dms scores must match"
        
        lls = data["lls"]
        lls_mean = lls.mean(axis=1)
        for top_k in top_ks:
            for threshold in thresholds:
                lls_to_use = lls.copy()
                lls_to_use = threshold_lls(lls_to_use, lls_mean, threshold, top_k)
                ensemble_likelihoods = lls_to_use.mean(axis=0)
                spearman_corr, _ = spearmanr(ensemble_likelihoods, dms_scores)
                new_row = {
                    "dms_id": dms_id,
                    "top_k": top_k,
                    "threshold": threshold,
                    "strategy": f"top_{top_k}_threshold_{threshold}",
                    "spearman_corr": spearman_corr,
                }
                results_rows.append(new_row)
        results_df = pd.DataFrame(results_rows)
        results_df.to_csv("protein_gym_likelihood_selection_results.csv", index=False)
    
    grouped_results_df = results_df[["strategy", "spearman_corr"]].groupby( "strategy").mean().reset_index()
    # sort by spearman_corr
    grouped_results_df = grouped_results_df.sort_values(by="spearman_corr", ascending=False)
    for i, row in grouped_results_df.iterrows():
        print(row.strategy, round(row.spearman_corr, 3))

