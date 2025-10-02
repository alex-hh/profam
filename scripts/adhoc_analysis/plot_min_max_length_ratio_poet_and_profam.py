
import pandas as pd
import glob
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import os

# Match styling used in compare_profam_poet_structual_comparison.py
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif', 'serif']
mpl.rcParams['mathtext.fontset'] = 'stix'

colors = {"ProFam": "#1f77b4", "PoET": "#2ca02c", "Random": "#808080"}

# profam_df = pd.read_csv("/mnt/disk2/cath_plm/sampling_results/profam_sequence_only_evaluation.csv")
# profam_df = profam_df[profam_df['aligned_generation_path'].str.contains("ensemble")]
# poet_df = pd.read_csv("/mnt/disk2/cath_plm/sampling_results/poet/poet_sequence_only_evaluation.csv")
# min_ratios_profam = profam_df["length_ratio_min_of_max"]
# min_ratios_poet = poet_df["length_ratio_min_of_max"]
# max_ratios_profam = profam_df["length_ratio_max_of_min"]
# max_ratios_poet = poet_df["length_ratio_max_of_min"]
# plt.hist(min_ratios_profam, bins=30, label="ProFam", alpha=0.5)
# plt.hist(min_ratios_poet, bins=30, label="PoET", alpha=0.5)
# plt.xlabel("min[generated length / min(prompt_lengths)]")
# plt.legend()
# plt.savefig("min_length_ratio_poet_and_profam.png")
# plt.clf()
# plt.hist(max_ratios_profam, bins=30, label="ProFam", alpha=0.5)
# plt.hist(max_ratios_poet, bins=30, label="PoET", alpha=0.5)
# plt.xlabel("max[generated length / max(prompt_lengths)]")
# plt.legend()
# plt.savefig("max_length_ratio_poet_and_profam.png")
# plt.clf()
# bp=1

def make_barplot_proportion_within_length_range(
    profam_df,
    poet_df,
    min_proportion=0.66,
    max_proportion=1.5,
    include_ci=True,
    save_path=None,
):
    """
    Plot the proportion of generated sequences whose lengths fall within
    [min_proportion * min(prompt lengths), max_proportion * max(prompt lengths)].

    Using available columns:
      - coverage_min = generated_len / max(prompt_len)
      - coverage_max = generated_len / min(prompt_len)
    Condition: (coverage_min <= max_proportion) & (coverage_max >= min_proportion)
    """

    def wilson_ci(num_success, num_total, z=1.96):
        if num_total <= 0:
            return np.nan, np.nan
        p_hat = num_success / num_total
        denom = 1.0 + (z ** 2) / num_total
        center = (p_hat + (z ** 2) / (2.0 * num_total)) / denom
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
    for name, df in [("ProFam", profam_df), ("PoET", poet_df)]:
        df_filt = df.copy()
        # Ensure finite values
        df_filt = df_filt[np.isfinite(df_filt.get("coverage_min", np.nan)) & np.isfinite(df_filt.get("coverage_max", np.nan))]

        within = (df_filt["coverage_min"] <= max_proportion) & (df_filt["coverage_max"] >= min_proportion)
        n = int(len(df_filt))
        k = int(within.sum())
        prop = (k / n) if n > 0 else np.nan
        low, high = (np.nan, np.nan)
        if include_ci and n > 0:
            low, high = wilson_ci(k, n)
        results.append({"name": name, "prop": prop, "low": low, "high": high, "n": n, "k": k})

    labels = [r["name"] for r in results]
    props = [r["prop"] for r in results]
    bar_colors = [colors.get(lbl, "#333333") for lbl in labels]

    if include_ci:
        lower_err = [(p - l) if (np.isfinite(p) and np.isfinite(l)) else 0.0 for p, l in [(r["prop"], r["low"]) for r in results]]
        upper_err = [(h - p) if (np.isfinite(p) and np.isfinite(h)) else 0.0 for p, h in [(r["prop"], r["high"]) for r in results]]
        yerr = np.array([lower_err, upper_err])
    else:
        yerr = None

    plt.figure(figsize=(6, 4))
    plt.bar(labels, props, color=bar_colors, yerr=yerr, capsize=4)
    plt.ylim(0, 1.0)
    plt.ylabel(f"Proportion within length range [{min_proportion}, {max_proportion}]")
    plt.title("Proportion of sequences within length range", pad=12)
    plt.tight_layout()

    if save_path is None:
        min_str = str(min_proportion).replace(".", "p")
        max_str = str(max_proportion).replace(".", "p")
        save_path = f"plots_for_paper/proportion_within_length_range_{min_str}_{max_str}_no_ensemble.png"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

pattern = "../sampling_results/foldseek_*/*ensemble*/alignments/*generated_seq_stats.csv"
csv_paths = glob.glob(pattern)
df = pd.concat([pd.read_csv(csv_path) for csv_path in csv_paths])
min_ratio_profam = np.clip((df.coverage_max ** -1).to_numpy(), 0, 4)
poet_pattern = "../sampling_results/poet/poet_foldseek_*/generated_sequences_foldseek_*/alignments/*_seq_stats.csv"
poet_csv_paths = glob.glob(poet_pattern)
poet_df = pd.concat([pd.read_csv(csv_path) for csv_path in poet_csv_paths])

# Barplot of proportion within specified length range (with error bars)
make_barplot_proportion_within_length_range(
    profam_df=df,
    poet_df=poet_df,
    min_proportion=0.66,
    max_proportion=1.5,
    include_ci=True,
)

min_ratio_poet = np.clip((poet_df.coverage_max ** -1).to_numpy(), 0, 4)
edges_min = np.linspace(
    min(min_ratio_profam.min(), min_ratio_poet.min()),
    max(min_ratio_profam.max(), min_ratio_poet.max()),
    101,
)
plt.hist(min_ratio_profam, bins=edges_min, label="ProFam", alpha=0.5, density=True)
plt.hist(min_ratio_poet, bins=edges_min, label="PoET", alpha=0.5, density=True)
plt.xlabel("min(prompt len) / generated seq len")
plt.legend()
plt.savefig("length_ratio_min_prompt_len_over_generated_len_poet_and_profam.png")
plt.clf()
bp=1
len_ratio_profam = np.clip(df.coverage_min.to_numpy(), 0, 4)
len_ratio_poet = np.clip(poet_df.coverage_min.to_numpy(), 0, 4)
edges_max = np.linspace(
    min(len_ratio_profam.min(), len_ratio_poet.min()),
    max(len_ratio_profam.max(), len_ratio_poet.max()),
    101,
)
plt.hist(len_ratio_profam, bins=edges_max, label="ProFam", alpha=0.5, density=True)
plt.hist(len_ratio_poet, bins=edges_max, label="PoET", alpha=0.5, density=True)
plt.xlabel("generated seq len / max(prompt len)")
plt.legend()
plt.savefig("length_ratio_generated_over_max_prompt_len_poet_and_profam.png")
plt.clf()


