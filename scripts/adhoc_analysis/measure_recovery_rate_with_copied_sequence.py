"""
Checks the recovery rate when just sampling a random sequence in the prompt
as the answer, using MAFFT family alignment and evaluating against most common AA.
"""
import glob
import subprocess
import sys
import tempfile
from collections import Counter

import numpy as np
import pandas as pd
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

dataset_dir = "../data/foldseek/foldseek_s50_seq_only/train_val_test_split/val"
parquets = glob.glob(f"{dataset_dir}/*.parquet")


def run_alignment_with_mafft(fasta_input, fasta_output, threads=12):
    """
    Runs an alignment with MAFFT.

    Example usage:
      mafft --thread N --auto input.fasta > output.fasta
    """
    cmd = ["mafft", "--thread", str(threads), "--auto", fasta_input]
    print(f"Running: {' '.join(cmd)}", file=sys.stderr)
    with open(fasta_output, "w") as fout:
        subprocess.run(cmd, check=True, stdout=fout)


def evaluate_sequence_recovery_rate(sequences, eval_sequence):
    try:
        # Create temporary files for MAFFT
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".fasta", delete=False
        ) as temp_in, tempfile.NamedTemporaryFile(
            mode="w", suffix=".fasta", delete=False
        ) as temp_out:

            # Write sequences to input FASTA
            for i, seq in enumerate(sequences):
                record = SeqRecord(Seq(seq), id=f"seq_{i}", description="")
                SeqIO.write(record, temp_in, "fasta")
            temp_in.flush()

            # Run MAFFT alignment
            run_alignment_with_mafft(temp_in.name, temp_out.name)

            # Read aligned sequences
            aligned_seqs = list(SeqIO.parse(temp_out.name, "fasta"))

            # Get the evaluation sequence (last one)
            eval_aligned = str(aligned_seqs[-1].seq)
            other_aligned = [str(seq.seq) for seq in aligned_seqs[:-1]]

            # For each position, find most common AA in other sequences
            matches = 0
            total_positions = 0

            for pos in range(len(eval_aligned)):
                # Skip positions where eval sequence has a gap
                if eval_aligned[pos] == "-":
                    continue

                # Get AAs at this position from other sequences
                pos_aas = [seq[pos] for seq in other_aligned if seq[pos] != "-"]
                if not pos_aas:  # Skip if no valid AAs at this position
                    continue

                # Find most common AA
                most_common_aa = Counter(pos_aas).most_common(1)[0][0]

                # Count as match if eval sequence matches most common AA
                if eval_aligned[pos] == most_common_aa:
                    matches += 1
                total_positions += 1

            return matches / total_positions if total_positions > 0 else 0

    except Exception as e:
        print(f"Error in alignment: {e}")
        return None
    finally:
        # Clean up temporary files
        try:
            import os

            os.unlink(temp_in.name)
            os.unlink(temp_out.name)
        except:
            pass


max_fams = 100
recovery_rates = []
for i, parquet_file in enumerate(parquets):
    print(f"Processing {i} / {len(parquets)}")
    df = pd.read_parquet(parquet_file)
    df = df.sample(min(max_fams, len(df)))
    for _, row in df.iterrows():
        sequences = row["sequences"]
        if len(sequences[0]) > 500:
            continue
        # shuffle the sequences which are in a numpy array
        sequences = np.random.permutation(sequences)
        evaluation_sequence = sequences[-1]
        ranked_sequences = sequences[:-1]

        recovery_rate = evaluate_sequence_recovery_rate(
            ranked_sequences, evaluation_sequence
        )
        if recovery_rate is not None:
            recovery_rates.append(recovery_rate)

print(f"Mean recovery rate: {np.mean(recovery_rates)}")
print(f"Median recovery rate: {np.median(recovery_rates)}")
print(f"Std recovery rate: {np.std(recovery_rates)}")
