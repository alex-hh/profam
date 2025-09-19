import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from statsmodels.nonparametric.smoothers_lowess import lowess

# profam_csv = "../sampling_results/colabfold_outputs/profam_structural_evaluation.csv"
# poet_csv = "../sampling_results/colabfold_outputs/poet_structural_eval/poet_structural_evaluation.csv"
profam_csv = "../sampling_results/colabfold_outputs/foldseek_combined_val_test_2025_09_17_seq_sim_lt_0p5/structural_evaluation.csv"
poet_csv = "../sampling_results/poet/poet_colabfold_outputs_seq_sim_lt_0p5/structural_evaluation.csv"
random_csv = "../sampling_results/randomly_mutated_sequences/random_colabfold_outputs/random_structural_evaluation.csv"



profam_df = pd.read_csv(profam_csv)
print(f"ProFam df: {profam_df.shape}")
poet_df = pd.read_csv(poet_csv)
print(f"PoET df: {poet_df.shape}")
random_df = pd.read_csv(random_csv)
print(f"Random df: {random_df.shape}")

def barplot_prop_with_good_tm_score(
    profam_df,
    poet_df,
    random_df,
    seq_threshold=0.5,
    tm_threshold=0.5,
    include_ci=True,
    save_path=None,
):
    def wilson_ci(num_success, num_total, z=1.96):
        if num_total == 0:
            return np.nan, np.nan
        p_hat = num_success / num_total
        denom = 1.0 + (z ** 2) / num_total
        center = (p_hat + (z ** 2) / (2 * num_total)) / denom
        half_width = (
            z
            * np.sqrt(
                (p_hat * (1.0 - p_hat)) / num_total + (z ** 2) / (4.0 * (num_total ** 2))
            )
            / denom
        )
        low = max(0.0, center - half_width)
        high = min(1.0, center + half_width)
        return low, high

    results = []
    for name, df in [("ProFam", profam_df), ("PoET", poet_df), ("Random", random_df)]:
        df_filt = df.copy()
        # Filter on finite values and thresholds
        df_filt = df_filt[np.isfinite(df_filt["seq_identity_max"]) & np.isfinite(df_filt["tm_max"])]
        df_filt = df_filt[df_filt.seq_identity_max < seq_threshold]
        n = len(df_filt)
        k = int((df_filt.tm_max >= tm_threshold).sum())
        prop = (k / n) if n > 0 else np.nan
        print(f"{name}: {prop} (k={k}, n={n})")
        low, high = (np.nan, np.nan)
        if include_ci and n > 0:
            low, high = wilson_ci(k, n)
        results.append({"name": name, "prop": prop, "low": low, "high": high, "n": n, "k": k})

    # Prepare barplot
    labels = [r["name"] for r in results]
    props = [r["prop"] for r in results]
    colors = {"ProFam": "#1f77b4", "PoET": "#2ca02c", "Random": "#808080"}
    bar_colors = [colors.get(lbl, "#333333") for lbl in labels]

    if include_ci:
        # yerr expects shape (2, N) for asymmetric errors
        lower_err = [
            (p - l) if (np.isfinite(p) and np.isfinite(l)) else 0.0
            for p, l in [(r["prop"], r["low"]) for r in results]
        ]
        upper_err = [
            (h - p) if (np.isfinite(p) and np.isfinite(h)) else 0.0
            for p, h in [(r["prop"], r["high"]) for r in results]
        ]
        yerr = np.array([lower_err, upper_err])
    else:
        yerr = None

    plt.figure()
    plt.bar(labels, props, color=bar_colors, yerr=yerr, capsize=4)
    plt.ylim(0, 0.55)
    plt.ylabel(f"Proportion with TM-score ≥ {tm_threshold}")
    plt.title(f"Proportion of good TM-score (seq id < {seq_threshold})")
    plt.tight_layout()

    if save_path is None:
        save_path = f"plots_for_paper/prop_good_tm_seq_lt_{seq_threshold}_tm_ge_{tm_threshold}.png"
    plt.savefig(save_path)
    plt.close()

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




# make_structure_sequence_similarity_plots(
#     profam_ensemble_df = profam_df, 
#     profam_single_df = None, 
#     poet_df = poet_df, 
#     random_df = random_df, 
#     plot_ensemble=True, 
#     plot_single=False, 
#     plot_poet=True, 
#     plot_random=True,
#     save_name="foldseek_combined_val_test_2025_09_17_seq_sim_lt_0p5",
# )
barplot_prop_with_good_tm_score(profam_df, poet_df, random_df, seq_threshold=0.4, tm_threshold=0.5, include_ci=True, save_path="plots_for_paper/prop_good_tm_seq_lt_0.4_tm_ge_0.5.png")