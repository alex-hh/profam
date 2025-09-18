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


pattern = "../sampling_results/foldseek_combined_val_test_2025_09_17/*.fasta"
output_dir = "../sampling_results/foldseek_combined_val_test_ensemble8_single_colabfold_fastas"
os.makedirs(output_dir, exist_ok=True)
fasta_paths = glob.glob(pattern)
for fasta_path in fasta_paths:
    fasta_name = os.path.basename(fasta_path).replace(".fasta", "")
    records = SeqIO.parse(fasta_path, "fasta")
    sorted_records = sorted(records, key=lambda x: len(x.seq))
    n_seqs = len(sorted_records)
    median_ix = n_seqs // 2
    median_record = sorted_records[median_ix]
    median_record.id = f"{fasta_name}_median"
    median_record.name = f"{fasta_name}_median"
    median_record.description = f"{fasta_name}_median"
    output_path = os.path.join(output_dir, f"{fasta_name}_median.fasta")
    SeqIO.write(median_record, output_path, "fasta")