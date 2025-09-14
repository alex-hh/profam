
import pandas as pd
import glob
import matplotlib.pyplot as plt
import numpy as np

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

pattern = "../sampling_results/foldseek_*/*ensemble*/alignments/*generated_seq_stats.csv"
csv_paths = glob.glob(pattern)
df = pd.concat([pd.read_csv(csv_path) for csv_path in csv_paths])
min_ratio_profam = np.clip((df.coverage_max ** -1).to_numpy(), 0, 4)
poet_pattern = "../sampling_results/poet/poet_foldseek_*/generated_sequences_foldseek_*/alignments/*_seq_stats.csv"
poet_csv_paths = glob.glob(poet_pattern)
poet_df = pd.concat([pd.read_csv(csv_path) for csv_path in poet_csv_paths])
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
