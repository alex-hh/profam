import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from statsmodels.nonparametric.smoothers_lowess import lowess

profam_csv = "../sampling_results/colabfold_outputs/profam_structural_evaluation.csv"
poet_csv = "../sampling_results/colabfold_outputs/poet_structural_eval/poet_structural_evaluation.csv"
random_csv = "../sampling_results/randomly_mutated_sequences/random_colabfold_outputs/random_structural_evaluation.csv"

profam_seq_csv = "../sampling_results/foldseek_val/ensemble_8_tp0p95_ns20/sequence_only_evaluation.csv"
poet_seq_csv = "../sampling_results/poet/poet_sequence_only_evaluation.csv"

profam_df = pd.read_csv(profam_csv)
poet_df = pd.read_csv(poet_csv)
profam_seq_df = pd.read_csv(profam_seq_csv)
poet_seq_df = pd.read_csv(poet_seq_csv)

def make_structure_sequence_similarity_plots(profam_df: pd.DataFrame, poet_df: pd.DataFrame):
    structure_metrics = ["tm_max", "lddt_max", "mean_plddt"]

    colors = {
        "profam_ensemble": "#1f77b4",  # blue
        "profam_single": "#ff7f0e",    # orange
        "poet": "#2ca02c",            # green
    }

    for structure_metric in structure_metrics:
        plt.figure()

        # ProFam ensemble
        df_mode = profam_df[profam_df["generated_pdb"].astype(str).str.contains("ensemble", na=False)]
        x = df_mode["seq_identity_max"].to_numpy(dtype=float)
        y = df_mode[structure_metric].to_numpy(dtype=float)
        mask = np.isfinite(x) & np.isfinite(y)
        x_valid, y_valid = x[mask], y[mask]
        if x_valid.size > 0:
            plt.scatter(x_valid, y_valid, s=12, alpha=0.5, label="ProFam ensemble", color=colors["profam_ensemble"], edgecolors="none")
            try:
                smoothed = lowess(y_valid, x_valid, frac=0.6, return_sorted=True)
                plt.plot(smoothed[:, 0], smoothed[:, 1], color=colors["profam_ensemble"], linewidth=2)
            except Exception:
                pass

        # ProFam single
        df_mode = profam_df[profam_df["generated_pdb"].astype(str).str.contains("single", na=False)]
        x = df_mode["seq_identity_max"].to_numpy(dtype=float)
        y = df_mode[structure_metric].to_numpy(dtype=float)
        mask = np.isfinite(x) & np.isfinite(y)
        x_valid, y_valid = x[mask], y[mask]
        if x_valid.size > 0:
            plt.scatter(x_valid, y_valid, s=12, alpha=0.5, label="ProFam single", color=colors["profam_single"], edgecolors="none")
            try:
                smoothed = lowess(y_valid, x_valid, frac=0.6, return_sorted=True)
                plt.plot(smoothed[:, 0], smoothed[:, 1], color=colors["profam_single"], linewidth=2)
            except Exception:
                pass

        # PoET (no mode subset)
        x = poet_df["seq_identity_max"].to_numpy(dtype=float)
        y = poet_df[structure_metric].to_numpy(dtype=float)
        mask = np.isfinite(x) & np.isfinite(y)
        x_valid, y_valid = x[mask], y[mask]
        if x_valid.size > 0:
            plt.scatter(x_valid, y_valid, s=12, alpha=0.5, label="PoET", color=colors["poet"], edgecolors="none")
            try:
                smoothed = lowess(y_valid, x_valid, frac=0.6, return_sorted=True)
                plt.plot(smoothed[:, 0], smoothed[:, 1], color=colors["poet"], linewidth=2)
            except Exception:
                pass

        plt.xlabel("Max sequence identity prompt")
        plt.ylabel(structure_metric)
        plt.legend(frameon=False)
        plt.tight_layout()
        plt.savefig(f"profam_poet_{structure_metric}.png")
        plt.close()




make_structure_sequence_similarity_plots(profam_df, poet_df)