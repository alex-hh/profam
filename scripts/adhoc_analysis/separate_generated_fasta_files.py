"""
Takes the generated fasta file for all of the funfams and foldseek families
and takes the first generated sequence for each family
and places it in a new fasta file with format:
{funfams/foldseek}_{val/test}_{ensemble/single}_{fam_id}_gen0.fasta
all files should be placed in the same directory
there should also be a file called fasta_file_list.txt

which has one filename per line

current paths to fastas:
  "../sampling_results/foldseek_val/sampler=ensemble_tp=0.95_ns=20_nv=8_red=mean_probs/*.fasta"
  "../sampling_results/foldseek_val/sampler=single_tp=0.95_ns=20_nv=8_red=mean_probs/*.fasta"
    "../sampling_results/funfams_val/sampler=ensemble_tp=0.95_ns=20_nv=8_red=mean_probs/*.fasta"
  "../sampling_results/funfams_val/sampler=single_tp=0.95_ns=20_nv=8_red=mean_probs/*.fasta"

    "../sampling_results/foldseek_test/sampler=ensemble_tp=0.95_ns=20_nv=8_red=mean_probs/*.fasta"
  "../sampling_results/foldseek_test/sampler=single_tp=0.95_ns=20_nv=8_red=mean_probs/*.fasta"
    "../sampling_results/funfams_test/sampler=ensemble_tp=0.95_ns=20_nv=8_red=mean_probs/*.fasta"
  "../sampling_results/funfams_test/sampler=single_tp=0.95_ns=20_nv=8_red=mean_probs/*.fasta"

ensure that the ordering in the fasta file list adds each file in the order above
"""

import argparse
import glob
import os
import re
import sys
from typing import List, Tuple
from Bio import SeqIO
import pandas as pd


# pattern = "../sampling_results/foldseek_combined_val_test_2025_09_17/*.fasta"
# output_dir = "../sampling_results/foldseek_combined_val_test_ensemble8_single_colabfold_fastas_seq_sim_lt_0p5"
# pattern = "../sampling_results/poet/poet_foldseek_*/generated_sequences_foldseek_*/*_samples20_seed42.fasta"
# output_dir = "../sampling_results/poet/poet_foldseek_combined_val_test_single_colabfold_fastas_seq_sim_lt_0p5"
# pattern = "../sampling_results/foldseek_combined_val_test_no_ensemble_2025_10_01/*.fasta"
pattern = "../sampling_results/foldseek_combined_val_test_poet_exact_prompts_no_ensemble_2025_10_01/*.fasta"
output_dir = f"{os.path.dirname(pattern)}/median_only"

seq_sim_upper_bound = None
os.makedirs(output_dir, exist_ok=True)
fasta_paths = glob.glob(pattern)
print(f"Found {len(fasta_paths)} fasta files")
for fasta_path in fasta_paths:
    fasta_name = os.path.basename(fasta_path).replace(".fasta", "")
    if seq_sim_upper_bound is not None:
      per_seq_sim_csv = os.path.join(os.path.dirname(fasta_path), f"alignments/{fasta_name}_seq_stats.csv")
      if not os.path.exists(per_seq_sim_csv):
          print(f"Per-sequence similarity CSV not found for {fasta_name}")
          continue
      per_seq_sim_df = pd.read_csv(per_seq_sim_csv)
      per_seq_sim_df = per_seq_sim_df[per_seq_sim_df["identity_max"] < seq_sim_upper_bound]
      if len(per_seq_sim_df) == 0:
          print(f"No sequences with similarity < {seq_sim_upper_bound} found for {fasta_name}")
          continue
    records = SeqIO.parse(fasta_path, "fasta")
    sorted_records = sorted(records, key=lambda x: len(x.seq))
    if seq_sim_upper_bound is not None:
        sorted_filtered_records = [r for r in sorted_records if r.id in per_seq_sim_df.generated_id.values]
    else:
        sorted_filtered_records = sorted_records
    print(f"Found {len(sorted_filtered_records)} sequences with similarity < {seq_sim_upper_bound} for {fasta_name}")
    n_seqs = len(sorted_filtered_records)
    median_ix = n_seqs // 2
    median_record = sorted_filtered_records[median_ix]
    median_record.id = f"{fasta_name}_median"
    median_record.name = f"{fasta_name}_median"
    median_record.description = f"{fasta_name}_median"
    output_path = os.path.join(output_dir, f"{fasta_name}_median.fasta")
    SeqIO.write(median_record, output_path, "fasta")