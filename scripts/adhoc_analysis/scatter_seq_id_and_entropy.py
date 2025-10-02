
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.nonparametric.smoothers_lowess import lowess

# Set Times New Roman font
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif']
plt.rcParams['font.size'] = 10


def make_info_theory_scatter_overlay(csv_paths, labels, colors, output_prefix):
    """
    Create overlaid scatter plots of sequence identity vs information theoretic metrics
    for multiple datasets.
    
    Args:
        csv_paths: List of paths to CSV files containing the data
        labels: List of labels for each dataset
        colors: List of colors for each dataset
        output_prefix: Prefix for output plot files
    """
    information_theoretic_cols = [
        "entropy_correlation",
        "js_divergence_mean",
        "symmetric_kl_divergence_mean",
        "kl_natural_to_synthetic_mean"
    ]
    x_col = "sequence_identity_mean_of_max"
    
    # Load all dataframes
    dfs = [pd.read_csv(path) for path in csv_paths]
    
    for metric in information_theoretic_cols:
        plt.figure(figsize=(7, 5))
        
        # Plot each dataset
        for df, label, color in zip(dfs, labels, colors):
            if metric not in df.columns:
                print(f"Warning: {metric} not found for {label}")
                continue
                
            x = df[x_col].to_numpy(dtype=float)
            y = df[metric].to_numpy(dtype=float)
            
            # Filter out non-finite values
            mask = np.isfinite(x) & np.isfinite(y)
            x_valid = x[mask]
            y_valid = y[mask]
            
            if x_valid.size == 0:
                print(f"No valid data for {metric} in {label}")
                continue
            
            # Create scatter plot
            plt.scatter(x_valid, y_valid, s=12, alpha=0.4, color=color, label=label)
            
            # Add LOWESS smoothed line
            try:
                smoothed = lowess(y_valid, x_valid, frac=0.5, return_sorted=True)
                plt.plot(smoothed[:, 0], smoothed[:, 1], color=color, linewidth=2.5, alpha=0.9)
            except Exception as e:
                print(f"Could not fit LOWESS for {metric} ({label}): {e}")
        
        # Format plot
        plt.xlabel("Sequence identity (mean of max)")
        plt.ylabel(metric.replace("_", " ").title())
        plt.legend(frameon=False, loc='best')
        plt.tight_layout()
        
        # Save plot
        output_path = f"{output_prefix}_{metric}.png"
        plt.savefig(output_path, dpi=300)
        print(f"Saved plot to {output_path}")
        plt.close()


if __name__ == "__main__":
    profam_ensemble_path = "../sampling_results/profam_ec_multi_seq_clustered_c70_pid_30_with_ensemble/profam_sequence_only_evaluation_ec_multi_sequence_with_ensemble.csv"
    profam_no_ensemble_path = "../sampling_results/profam_ec_multi_seq_clustered_c70_pid_30_no_ensemble/profam_sequence_only_evaluation_ec_multi_sequence_no_ensemble.csv"
    poet_path = "../sampling_results/poet/ec_multi_clustered_c70_pid_30_poet/poet_sequence_only_evaluation_ec_multi_sequence.csv"
    
    # Generate overlaid plots for all three datasets
    make_info_theory_scatter_overlay(
        csv_paths=[profam_ensemble_path, profam_no_ensemble_path, poet_path],
        labels=["ProFaM ensemble", "ProFaM single", "POET"],
        colors=["#1f77b4", "#ff7f0e", "#2ca02c"],  # Blue, orange, green
        output_prefix="plots_for_paper/seq_id_vs_info_theory_comparison"
    )