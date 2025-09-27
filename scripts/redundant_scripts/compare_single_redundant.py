import pandas as pd
import os
import math
import numpy as np
import matplotlib.pyplot as plt


df_single = pd.read_csv('../sampling_results/profam_ec_multi_seq_synthetic_msas_no_ensemble/profam_sequence_only_evaluation_ec_multi_sequence_no_ensemble.csv')
df_ensemble  = pd.read_csv('../sampling_results/profam_ec_multi_seq_synthetic_msas/profam_sequence_only_evaluation_ec_multi_sequence_with_ensemble.csv')
df_single['ec_num'] = df_single['aligned_prompt_path'].apply(lambda x: x.split("/")[-1].split("_aln.filtered")[0])
df_ensemble['ec_num'] = df_ensemble['aligned_prompt_path'].apply(lambda x: x.split("/")[-1].split("_aln.filtered")[0])
print(f"df_single: {df_single.shape}")
print(f"df_ensemble: {df_ensemble.shape}")
df_single = df_single[(df_single.length_ratio_max_of_min < 1.15)&(df_single.length_ratio_min_of_max > 0.85)]
df_ensemble = df_ensemble[(df_ensemble.length_ratio_max_of_min < 1.15)&(df_ensemble.length_ratio_min_of_max > 0.85)]
df_single = df_single[df_single.ec_num.isin(df_ensemble.ec_num)]
df_ensemble = df_ensemble[df_ensemble.ec_num.isin(df_single.ec_num)]
print(f"AFTER FILTERING df_single: {df_single.shape}")
print(f"AFTER FILTERING df_ensemble: {df_ensemble.shape}")


columns_for_histogram = ["entropy_correlation", "length_ratio_min_of_max", "length_ratio_max_of_min", "sequence_identity_min_of_max", "sequence_identity_mean_of_max", "js_divergence_mean"]

save_dir = "../plots_compare_single_ensemble_on_ec_filtered_v3"
os.makedirs(save_dir, exist_ok=True)

scatter_plot_pairs = [
    ("entropy_correlation", "js_divergence_mean"),
    ("entropy_correlation", "symmetric_kl_divergence_mean"),
    ("entropy_correlation", "kl_natural_to_synthetic_mean"),
    
    ("sequence_identity_mean_of_max", "js_divergence_mean"),
    ("sequence_identity_mean_of_max", "symmetric_kl_divergence_mean"),
    ("sequence_identity_mean_of_max", "kl_natural_to_synthetic_mean"),

    ("js_divergence_mean", "symmetric_kl_divergence_mean"),
    ("js_divergence_mean", "kl_natural_to_synthetic_mean"),

    ("symmetric_kl_divergence_mean", "kl_natural_to_synthetic_mean"),
]


def _compute_common_bins(values_a: np.ndarray, values_b: np.ndarray, default_bins: int = 50) -> np.ndarray:
    if values_a.size == 0 and values_b.size == 0:
        return np.linspace(0.0, 1.0, default_bins + 1)
    if values_a.size == 0:
        vmin = float(np.nanmin(values_b))
        vmax = float(np.nanmax(values_b))
    elif values_b.size == 0:
        vmin = float(np.nanmin(values_a))
        vmax = float(np.nanmax(values_a))
    else:
        vmin = float(min(np.nanmin(values_a), np.nanmin(values_b)))
        vmax = float(max(np.nanmax(values_a), np.nanmax(values_b)))
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        vmin, vmax = 0.0, 1.0
    if vmin == vmax:
        eps = 1e-6 if vmin == 0 else abs(vmin) * 1e-3
        vmin -= eps
        vmax += eps
    return np.linspace(vmin, vmax, default_bins + 1)


def save_histograms_subplots(df_a: pd.DataFrame, df_b: pd.DataFrame, columns: list, out_path: str) -> None:
    num_metrics = len(columns)
    cols_per_row = min(3, num_metrics)
    rows = int(math.ceil(num_metrics / cols_per_row))
    fig, axes = plt.subplots(rows, cols_per_row, figsize=(cols_per_row * 5.5, rows * 4.5), squeeze=False)

    for idx, metric in enumerate(columns):
        r = idx // cols_per_row
        c = idx % cols_per_row
        ax = axes[r][c]

        if metric not in df_a.columns or metric not in df_b.columns:
            ax.text(0.5, 0.5, f"Missing column: {metric}", ha='center', va='center')
            ax.set_axis_off()
            continue

        a_vals = df_a[metric].to_numpy(dtype=float)
        b_vals = df_b[metric].to_numpy(dtype=float)
        a_vals = a_vals[~np.isnan(a_vals)]
        b_vals = b_vals[~np.isnan(b_vals)]

        bins = _compute_common_bins(a_vals, b_vals, default_bins=50)

        if a_vals.size == 0 and b_vals.size == 0:
            ax.text(0.5, 0.5, f"No data for {metric}", ha='center', va='center')
            ax.set_axis_off()
            continue

        ax.hist(a_vals, bins=bins, density=True, alpha=0.6, label='single', color='tab:blue', edgecolor='white', linewidth=0.5)
        ax.hist(b_vals, bins=bins, density=True, alpha=0.6, label='ensemble', color='tab:orange', edgecolor='white', linewidth=0.5)
        ax.set_title(metric)
        ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)
        ax.legend(frameon=False, fontsize=9)

    # Hide any unused subplots
    total_axes = rows * cols_per_row
    for empty_idx in range(num_metrics, total_axes):
        r = empty_idx // cols_per_row
        c = empty_idx % cols_per_row
        axes[r][c].set_visible(False)

    fig.suptitle('Single vs Ensemble: Metric Distributions', fontsize=14)
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def save_scatter_both(df_single: pd.DataFrame, df_ensemble: pd.DataFrame, x_col: str, y_col: str, out_dir: str) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 5.5))

    any_plotted = False

    # Plot single
    if x_col in df_single.columns and y_col in df_single.columns:
        sub_single = df_single[[x_col, y_col]].dropna()
        if len(sub_single) > 0:
            ax.scatter(sub_single[x_col], sub_single[y_col], s=12, alpha=0.5, color='tab:blue', label='single', edgecolors='none')
            any_plotted = True

    # Plot ensemble
    if x_col in df_ensemble.columns and y_col in df_ensemble.columns:
        sub_ensemble = df_ensemble[[x_col, y_col]].dropna()
        if len(sub_ensemble) > 0:
            ax.scatter(sub_ensemble[x_col], sub_ensemble[y_col], s=12, alpha=0.5, color='tab:orange', label='ensemble', edgecolors='none')
            any_plotted = True

    if not any_plotted:
        ax.text(0.5, 0.5, f"No data or missing columns: {x_col}/{y_col}", ha='center', va='center')
        ax.set_axis_off()
    else:
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.set_title(f"{x_col} vs {y_col} (single vs ensemble)")
        ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)
        ax.legend(frameon=False, fontsize=9)

    fname = f"scatter_single_vs_ensemble_{x_col}_vs_{y_col}.png"
    out_path = os.path.join(out_dir, fname)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    # Histograms in a single subplot figure
    hist_out_path = os.path.join(save_dir, "hist_subplots_single_vs_ensemble.png")
    save_histograms_subplots(df_single, df_ensemble, columns_for_histogram, hist_out_path)

    # Scatter plots: single figure per pair with both datasets overlaid
    for x_col, y_col in scatter_plot_pairs:
        save_scatter_both(df_single, df_ensemble, x_col, y_col, out_dir=save_dir)