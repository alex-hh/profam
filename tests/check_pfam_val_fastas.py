import os
import pandas as pd
from Bio import SeqIO

def read_fasta(file_path):
    return {record.id: str(record.seq) for record in SeqIO.parse(file_path, "fasta")}

def read_parquet(file_path):
    df = pd.read_parquet(file_path)
    return dict(zip(df['sequence_name'], df['sequence']))

def compare_sequences(fasta_dir, parquet_dir, index_file):
    index_df = pd.read_csv(index_file)
    
    for fasta_file in os.listdir(fasta_dir):
        if fasta_file.endswith("_test.fasta"):
            fam_id = fasta_file.split("_")[0]
            fasta_sequences = read_fasta(os.path.join(fasta_dir, fasta_file))
            
            parquet_file = index_df[index_df['fam_id'] == fam_id]['parquet_file'].values[0]
            parquet_sequences = read_parquet(os.path.join(parquet_dir, parquet_file))
            
            for seq_id, fasta_seq in fasta_sequences.items():
                if seq_id in parquet_sequences:
                    if fasta_seq != parquet_sequences[seq_id]:
                        print(f"Discrepancy found for {seq_id} in {fam_id}")
                        print(f"FASTA: {fasta_seq}")
                        print(f"Parquet: {parquet_sequences[seq_id]}")
                        print()
                else:
                    print(f"Sequence {seq_id} not found in Parquet file for {fam_id}")

# Usage
fasta_dir = "data/val_test/pfam/val/clustered_split_fastas"
parquet_dir = "../data/pfam/train_test_split_parquets/val"
index_file = "../data/pfam/train_test_split_parquets/new_index.csv"

compare_sequences(fasta_dir, parquet_dir, index_file)