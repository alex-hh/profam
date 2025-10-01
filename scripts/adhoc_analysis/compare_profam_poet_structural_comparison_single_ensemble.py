import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import seaborn as sns
import numpy as np
from statsmodels.nonparametric.smoothers_lowess import lowess
import os

# Set Times New Roman as the default font
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif', 'serif']
mpl.rcParams['mathtext.fontset'] = 'stix'  # For math text compatibility with Times

colors = {"ProFam": "#1f77b4", "PoET": "#2ca02c", "Random": "#808080"}



def make_structure_sequence_similarity_plots(
    profam_ensemble_df: pd.DataFrame,
    profam_single_df: pd.DataFrame,
    poet_df: pd.DataFrame, 
    random_df: pd.DataFrame,
    plot_ensemble: bool = True,
    plot_single: bool = True,
    plot_poet: bool = True,
    plot_random: bool = True,
    save_name: str = "",
):
    structure_metrics = ["tm_max", "lddt_max", "mean_plddt"]

    colors = {
        "profam_ensemble": "#1f77b4",  # blue
        "profam_single": "#ff7f0e",    # orange
        "poet": "#2ca02c",            # green
        "random": "#808080",          # gray
    }

    for structure_metric in structure_metrics:
        plt.figure()

        # ProFam ensemble
        if plot_ensemble:
            x = profam_ensemble_df["seq_identity_max"].to_numpy(dtype=float)
            y = profam_ensemble_df[structure_metric].to_numpy(dtype=float)
            mask = np.isfinite(x) & np.isfinite(y)
            print(f"ProFam ensemble: {len(x[mask])} points")
            x_valid, y_valid = x[mask], y[mask]
            if x_valid.size > 0:
                plt.scatter(x_valid, y_valid, s=12, alpha=0.5, label="ProFam ensemble", color=colors["profam_ensemble"], edgecolors="none")
                try:
                    smoothed = lowess(y_valid, x_valid, frac=0.6, return_sorted=True)
                    plt.plot(smoothed[:, 0], smoothed[:, 1], color=colors["profam_ensemble"], linewidth=2)
                except Exception:
                    pass

        # ProFam single
        if plot_single:
            x = profam_single_df["seq_identity_max"].to_numpy(dtype=float)
            y = profam_single_df[structure_metric].to_numpy(dtype=float)
            mask = np.isfinite(x) & np.isfinite(y)
            print(f"ProFam single: {len(x[mask])} points")
            x_valid, y_valid = x[mask], y[mask]
            if x_valid.size > 0:
                plt.scatter(x_valid, y_valid, s=12, alpha=0.5, label="ProFam single", color=colors["profam_single"], edgecolors="none")
                try:
                    smoothed = lowess(y_valid, x_valid, frac=0.6, return_sorted=True)
                    plt.plot(smoothed[:, 0], smoothed[:, 1], color=colors["profam_single"], linewidth=2)
                except Exception:
                    pass

        # PoET (no mode subset)
        if plot_poet:
            x = poet_df["seq_identity_max"].to_numpy(dtype=float)
            y = poet_df[structure_metric].to_numpy(dtype=float)
            mask = np.isfinite(x) & np.isfinite(y)
            print(f"PoET: {len(x[mask])} points")
            x_valid, y_valid = x[mask], y[mask]
            if x_valid.size > 0:
                plt.scatter(x_valid, y_valid, s=12, alpha=0.5, label="PoET", color=colors["poet"], edgecolors="none")
                try:
                    smoothed = lowess(y_valid, x_valid, frac=0.6, return_sorted=True)
                    plt.plot(smoothed[:, 0], smoothed[:, 1], color=colors["poet"], linewidth=2)
                except Exception:
                    pass

        # Random
        if plot_random:
            x = random_df["seq_identity_max"].to_numpy(dtype=float)
            y = random_df[structure_metric].to_numpy(dtype=float)
            mask = np.isfinite(x) & np.isfinite(y)
            print(f"Random: {len(x[mask])} points")
            x_valid, y_valid = x[mask], y[mask]
            if x_valid.size > 0:
                plt.scatter(x_valid, y_valid, s=12, alpha=0.5, label="Random", color=colors["random"], edgecolors="none")
                try:
                    smoothed = lowess(y_valid, x_valid, frac=0.6, return_sorted=True)
                    plt.plot(smoothed[:, 0], smoothed[:, 1], color=colors["random"], linewidth=2)
                except Exception:
                    pass

        plt.xlabel("Max sequence identity prompt")
        plt.ylabel(structure_metric)
        plt.legend(frameon=False)
        plt.tight_layout()
        figname = "structure_sequence_similarity"
        if plot_ensemble:
            figname += "_ensemble"
        if plot_single:
            figname += "_single"
        if plot_poet:
            figname += "_poet"
        if plot_random:
            figname += "_random"
        plt.savefig(f"plots_for_paper/{figname}_{structure_metric}_{save_name}.png")
        plt.close()


if __name__ == "__main__":
    profam_csv = "../sampling_results/colabfold_outputs/profam_structural_evaluation.csv"
    poet_csv = "../sampling_results/colabfold_outputs/poet_structural_eval/poet_structural_evaluation.csv"
    random_csv = "../sampling_results/randomly_mutated_sequences/random_colabfold_outputs/random_structural_evaluation.csv"
    profam_csv = "/mnt/disk2/cath_plm/sampling_results/colabfold_outputs/profam_structural_evaluation.csv"

    profam_df = pd.read_csv(profam_csv)
    print(f"ProFam df: {profam_df.shape}")
    profam_df_single = profam_df[profam_df.generated_pdb.str.contains("single")]
    profam_df_ensemble = profam_df[profam_df.generated_pdb.str.contains("ensemble")]
    print(f"ProFam single df: {profam_df_single.shape}")
    print(f"ProFam ensemble df: {profam_df_ensemble.shape}")
    poet_df = pd.read_csv(poet_csv)
    print(f"PoET df: {poet_df.shape}")
    random_df = pd.read_csv(random_csv)
    print(f"Random df: {random_df.shape}")
    make_structure_sequence_similarity_plots(
        profam_ensemble_df = profam_df_ensemble, 
        profam_single_df = profam_df_single, 
        poet_df = poet_df, 
        random_df = random_df, 
        plot_ensemble=True, 
        plot_single=True, 
        plot_poet=True, 
        plot_random=True,
        save_name="foldseek_combined_val_test_single_ensemble_poet_random",
    )