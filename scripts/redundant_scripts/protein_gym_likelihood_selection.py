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

# Selection config
top_pcts = [1.0, 0.9, 0.8, 0.7, 0.4, 0.3, 0.2]
min_seq_sims = [0.0, 0.1, 0.15]
max_seq_sims = [1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65]
ll_thresholds = [0, -0.5, -1.0, -1.1, -1.2, -1.3, -1.4, -1.5]
min_coverages = [0, 0.6, 0.7, 0.8]

def load_dms_scores(csv_path, seed=42, max_mutated_sequences=3000):
    dms_df = pd.read_csv(csv_path)
    if max_mutated_sequences is not None and max_mutated_sequences < len(dms_df):
        dms_df = dms_df.sample(n=max_mutated_sequences, random_state=seed)
    return dms_df

def select_replicates_with_thresholds(
    lls: np.ndarray,
    lls_mean: np.ndarray,
    ll_threshold: float,
    top_pct: float,
    min_seq_sim_arr: np.ndarray,
    max_seq_sim_arr: np.ndarray,
    coverage_arr: np.ndarray,
    min_sim_threshold: float,
    max_sim_threshold: float,
    min_coverage: float,
):
    """Select replicates based on LL threshold and sequence similarity thresholds.

    - Keep replicates where:
      lls_mean < ll_threshold AND
      min_seq_sim_arr >= min_sim_threshold AND
      max_seq_sim_arr <= max_sim_threshold
    - If fewer than 5 replicates satisfy these, add replicates with the smallest
      total violation distance to reach 5 replicates.
    - From the resulting set, keep the top percentage of replicates (by lls_mean, descending).
    """
    num_replicates = lls.shape[0]

    # Compute per-condition violations (>= 0); zero means condition satisfied
    ll_violation = np.maximum(0.0, lls_mean - ll_threshold)
    min_sim_violation = np.maximum(0.0, min_sim_threshold - min_seq_sim_arr)
    max_sim_violation = np.maximum(0.0, max_seq_sim_arr - max_sim_threshold)
    coverage_violation = np.maximum(0.0, min_coverage - coverage_arr)
    total_violation = ll_violation + min_sim_violation + max_sim_violation + coverage_violation

    valid_mask = total_violation == 0.0
    valid_indices = np.where(valid_mask)[0]

    # Ensure at least 5 replicates by including the closest violators
    if valid_indices.size < 5:
        # Sort all replicates by total violation ascending (valid ones first)
        sorted_by_violation = np.argsort(total_violation)
        chosen_indices = sorted_by_violation[: max(10, valid_indices.size)]
    else:
        chosen_indices = valid_indices

    chosen_means = lls_mean[chosen_indices]

    # Determine top-k from percentage of TOTAL replicates
    top_k = int(np.ceil(max(0.0, min(1.0, top_pct)) * num_replicates))
    top_k = max(1, top_k)

    if top_k < len(chosen_indices):
        # Higher lls_mean is better
        top_order = np.argsort(-chosen_means)[:top_k]
        final_indices = chosen_indices[top_order]
    else:
        final_indices = chosen_indices

    return lls[final_indices], final_indices


if __name__ == "__main__":
    # npz_files = glob.glob("logs/abyoeovl_openfold_fs50_ur90_memmap_251m_copied_2025-06-23_22-18/20250726_173620/*.npz")
    # npz_files = glob.glob("logs/abyoeovl_openfold_fs50_ur90_memmap_251m_copied_2025-06-23_22-18/20250730_183304_100_reps/*.npz")
    # npz_files = glob.glob("debug_gym_results/20250807_231522/*.npz")
    npz_files = glob.glob("logs/proteingym_eval_results/20250808_000300_PoET_MSAs_BO/*.npz")
    print(f"Found {len(npz_files)} npz files")
    results_rows = []
    completed_dms_ids = []
    for npz_file in npz_files:
        print(npz_file)
        csv_path = npz_file.replace(".npz", ".csv")
        df = pd.read_csv(csv_path)
        dms_id = df["DMS_id"].iloc[0]
        completed_dms_ids.append(dms_id)
        data = np.load(npz_file)
        dms_scores = data["dms_scores"]
        assert len(dms_scores) == data["lls"].shape[1], "Number of lls and dms scores must match"
        min_seq_sim = data["min_sequence_similarity_list"]
        max_seq_sim = data["max_sequence_similarity_list"]
        coverage_arr = data["min_coverage_list"]
        assert len(min_seq_sim) == len(max_seq_sim) == len(coverage_arr) == data["lls"].shape[0], "Number of min_seq_sim, max_seq_sim, coverage, and lls must match"
        
        lls = data["lls"]
        lls_mean = lls.mean(axis=1)
        min_seq_sim_arr = min_seq_sim
        max_seq_sim_arr = max_seq_sim

        for top_pct in top_pcts:
            for threshold in ll_thresholds:
                for min_sim_threshold in min_seq_sims:
                    for max_sim_threshold in max_seq_sims:
                        for min_coverage in min_coverages:
                            # Skip inconsistent threshold combos
                            if min_sim_threshold > max_sim_threshold:
                                continue
                            lls_to_use, chosen_indices = select_replicates_with_thresholds(
                                lls=lls,
                                lls_mean=lls_mean,
                                ll_threshold=threshold,
                                top_pct=top_pct,
                                min_seq_sim_arr=min_seq_sim_arr,
                                max_seq_sim_arr=max_seq_sim_arr,
                                coverage_arr=coverage_arr,
                                min_sim_threshold=min_sim_threshold,
                                max_sim_threshold=max_sim_threshold,
                                min_coverage=min_coverage,
                            )
                            ensemble_likelihoods = lls_to_use.mean(axis=0)
                            spearman_corr, _ = spearmanr(ensemble_likelihoods, dms_scores)

                            # Derived values for logging
                            derived_top_k = len(chosen_indices)

                            new_row = {
                                "dms_id": dms_id,
                                "top_pct": top_pct,
                                "ll_threshold": threshold,
                                "min_sim_threshold": min_sim_threshold,
                                "max_sim_threshold": max_sim_threshold,
                                "strategy": (
                                    f"pct_{top_pct}_llth_{threshold}_minsim_{min_sim_threshold}_maxsim_{max_sim_threshold}"
                                ),
                                "spearman_corr": spearman_corr,
                                "n_in_ensemble": len(lls_to_use),
                                "replicate_indices": ",".join(map(str, chosen_indices.tolist())),
                                "derived_top_k": derived_top_k,
                            }
                            results_rows.append(new_row)
        results_df = pd.DataFrame(results_rows)
        # results_df.to_csv(f"{os.path.dirname(npz_file)}/protein_gym_likelihood_selection_results.csv", index=False)
    
    grouped_results_df = results_df[["strategy", "spearman_corr"]].groupby( "strategy").mean().reset_index()
    # sort by spearman_corr
    grouped_results_df = grouped_results_df.sort_values(by="spearman_corr", ascending=False)
    grouped_results_df.to_csv(f"{os.path.dirname(npz_file)}/protein_gym_likelihood_selection_results_grouped_by_strategy.csv", index=False)
    for i, row in grouped_results_df.iterrows().iloc[:20]:
        print(row.strategy, round(row.spearman_corr, 3))

