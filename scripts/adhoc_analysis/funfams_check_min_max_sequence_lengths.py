import glob
import os
import numpy as np
import matplotlib.pyplot as plt


def extract_sequences_fasta(fasta_path):
    with open(fasta_path, "r") as f:
        sequences = []
        sequence_fragments = []
        for line in f:
            if line.startswith(">"):
                if len(sequence_fragments) > 0:
                    sequences.append("".join(sequence_fragments))
                sequence_fragments = []
            else:
                sequence_fragments.append(line.strip().replace("-", ""))
        if len(sequence_fragments) > 0:
            sequences.append("".join(sequence_fragments))
    return sequences

fasta_paths = glob.glob("/mnt/disk2/cath_plm/data/funfams/original_funfams_fastas/funfams_s50_ali/*/*.faa")
min_max_ratios = []
prop_50pc_over_median = []
prop_30pc_under_median = []
for fasta_path in fasta_paths:
    sequences = extract_sequences_fasta(fasta_path)
    if len(sequences) < 5:
        continue
    lengths = [len(seq) for seq in sequences]
    median_length = np.median(lengths)
    n_seqs_50pc_over_median = sum(1 for l in lengths if l > median_length * 1.5)
    n_seqs_30pc_under_median = sum(1 for l in lengths if l < median_length * 0.7)
    prop_50pc_over_median.append(n_seqs_50pc_over_median / len(lengths))
    prop_30pc_under_median.append(n_seqs_30pc_under_median / len(lengths))
    max_over_min = max(lengths) / min(lengths)
    min_max_ratios.append(max_over_min)

plt.hist(min_max_ratios, bins=100)
plt.title("Min/max ratio of sequence lengths")
plt.xlabel("Min/max ratio")
plt.ylabel("Count")
plt.savefig("min_max_ratio.png")
plt.show()
plt.clf()

plt.hist(prop_50pc_over_median, bins=100)
plt.title("Proportion of sequences over 150% median length")
plt.xlabel("Proportion of sequences over 150% median length")
plt.ylabel("Count")
plt.savefig("prop_50pc_over_median.png")
plt.show()
plt.clf()

plt.hist(prop_30pc_under_median, bins=100)
plt.title("Proportion of sequences under 70% median length")
plt.xlabel("Proportion of sequences under 70% median length")
plt.ylabel("Count")
plt.savefig("prop_30pc_under_median.png")
plt.show()
plt.clf()