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



if __name__ == "__main__":
    # npz_files = glob.glob("logs/abyoeovl_openfold_fs50_ur90_memmap_251m_copied_2025-06-23_22-18/20250726_173620/*.npz")
    # npz_files = glob.glob("logs/abyoeovl_openfold_fs50_ur90_memmap_251m_copied_2025-06-23_22-18/20250730_183304_100_reps/*.npz")
    # npz_files = glob.glob("logs/proteingym_eval_results/20250808_000300_PoET_MSAs_BO/*.npz")
    # npz_files = glob.glob("logs/proteingym_eval_results/20250810_135739_poet_msas_random_sampling_v4/*.npz")
    # npz_files = glob.glob("logs/abyoeovl_openfold_fs50_ur90_memmap_251m_copied_2025-06-23_22-18/20250810_135739_full_gym/*.npz")
    # npz_files = glob.glob("logs/abyoeovl_openfold_fs50_ur90_memmap_251m_copied_2025-06-23_22-18/20250810_135739_v6_full_gym/*v6_lls.npz")
    # npz_files = glob.glob("logs/abyoeovl_openfold_fs50_ur90_memmap_251m_copied_2025-06-23_22-18/filtered_poet_msas_with_all_weighting_v9/*v9_lls.npz")
    # npz_files = glob.glob("logs/abyoeovl_openfold_fs50_ur90_memmap_251m_copied_2025-06-23_22-18/20250828_v6_gym_msas/*v6_lls.npz")
    # npz_files = glob.glob("logs/abyoeovl_openfold_fs50_ur90_memmap_251m_copied_2025-06-23_22-18/20250829_v6_gym_msas_filtered/*v6_lls.npz")
    # save_dir = "../v6_gym_msas_evaluating_n_forward"
    npz_files = glob.glob("logs/saturn_cloud_good_runs/abyoeovl_openfold_fs50_ur90_memmap_251m/copied_2025-06-23_22-18/2025-06-10_22-48-14-455325/unfiltered_poet_msas_with_only_diversity_weighting_v9/*.npz")
    save_dir = "plots_for_paper/gym_spearman_by_n_samples_and_likelihood_abyoeovl_unfiltered_poet_msas_with_only_diversity_weighting_v9"
    os.makedirs(save_dir, exist_ok=True)
    results_rows = []
    target_score = -1.3
    for n_forward in [1, 2, 5, 10, 15, 20, 35, 50, 65, 80, 100, 110, 120]: # , 150, 180, 200
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
                lls_to_use = data["lls"][:n_forward]
                # sort the lls by the mean likelihood
                mutant_means = lls_to_use.mean(axis=1)
                lls_distances = np.abs(mutant_means - target_score)
                lls_mean_sorted_indices = np.argsort(lls_distances)
                lls_sorted = lls_to_use[lls_mean_sorted_indices]
                lls_to_use_sorted = lls_sorted[:n_select]
                lls_mean_sorted = lls_to_use_sorted.mean(axis=0)
                spearman_score = spearmanr(lls_mean_sorted, dms_scores)[0]
                if not np.isfinite(spearman_score):
                    bp=1
                else:
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
    
    
    print(f"Results dataframe shape: {results_df.shape}")
    print("Sample data:")
    print(results_df.head())
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    grouped_data = results_df.groupby("n_forward").max()['spearman_score']
    print(f"Grouped data shape: {grouped_data.shape}")
    print("Grouped data:")
    print(grouped_data)
    
    ax = grouped_data.plot(marker='o', linewidth=2, markersize=6)
    ax.set_ylabel("Spearman correlation")
    ax.set_xlabel("Number of forward passes")
    ax.set_title("How does the number of forward passes affect performance?")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/protein_gym_likelihood_how_does_n_samples_with_selection_affect_performance.png", 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Summary statistics by n_forward:")
    print(results_df.groupby("n_forward").max())
    results_df.to_csv(f"{save_dir}/protein_gym_likelihood_how_does_n_samples_with_selection_affect_performance.csv", index=False)
                



