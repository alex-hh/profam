import os

import pandas as pd
from Bio import SeqIO, pairwise2
from Bio.pairwise2 import format_alignment


def read_fasta(file_path):
    return {record.id: str(record.seq) for record in SeqIO.parse(file_path, "fasta")}


def read_parquet(file_path, fam_id):
    df = pd.read_parquet(file_path)
    df = df[df["fam_id"] == fam_id]
    df = df.iloc[0].to_dict()
    accessions = [x.split("/")[0] for x in df["accessions"]]
    if len(df) == 0:
        print(f"Index incorrect for {fam_id}")
        return {}
    return dict(zip(accessions, df["sequences"]))


def make_seq_id_to_accession():
    df = pd.read_csv("data/val_test/pfam/pfam_val_test_accessions_w_unip_accs.csv")
    return dict(zip(df["accession"], df["Entry"]))


def print_pairwise_alignment(seq1, seq2):
    alignments = pairwise2.align.globalxx(seq1, seq2)
    for a in alignments:
        print(format_alignment(*a))
        break
    return alignments[0].score


def compare_sequences(fasta_dir, parquet_dir, index_file):
    index_df = pd.read_csv(index_file)
    seq_id_to_accession = make_seq_id_to_accession()
    match_count = 0
    above_98_counter = 0
    seq_counter = 0
    fasta_longer = 0
    parquet_longer = 0
    same_length = 0
    for fasta_file in os.listdir(fasta_dir):
        if fasta_file.endswith("_test.fasta"):
            fam_id = fasta_file.split("_")[0].split(".")[0]
            fasta_sequences = read_fasta(os.path.join(fasta_dir, fasta_file))

            parquet_file = index_df[index_df["fam_id"] == fam_id]["parquet_file"]
            if len(parquet_file) == 0:
                # print(f"No Parquet file found for {fam_id}")
                continue
            else:
                parquet_file = parquet_file.values[0]
            parquet_sequences = read_parquet(
                os.path.join(parquet_dir, parquet_file), fam_id
            )

            for seq_id, fasta_seq in fasta_sequences.items():
                seq_id = seq_id.split("/")[0]
                up_id = seq_id_to_accession.get(seq_id, None)
                if up_id is None:
                    pass
                if seq_id in parquet_sequences:
                    seq_counter += 1
                    parq_seq = parquet_sequences[seq_id]
                    raw_parq = parq_seq.replace("-", "").replace(".", "").upper()
                    raw_fasta = fasta_seq.replace("-", "").replace(".", "")
                    if len(raw_parq) > len(raw_fasta):
                        parquet_longer += 1
                    if len(raw_fasta) > len(raw_parq):
                        fasta_longer += 1
                    if len(raw_fasta) == len(raw_parq):
                        same_length += 1
                    if raw_parq == raw_fasta:
                        match_count += 1
                        continue
                    else:
                        score = print_pairwise_alignment(raw_parq, raw_fasta)
                        score = score / max(len(raw_parq), len(raw_fasta))
                        if score > 0.98:
                            above_98_counter += 1

                else:
                    pass
                    # print(f"Sequence {seq_id} not found in Parquet file for {fam_id}")
    print(f"Match count: {match_count}")
    print(f"Above 98%: {above_98_counter}")
    print(f"Total sequences: {seq_counter}")
    print(f"Parquet longer: {parquet_longer}")
    print(f"Fasta longer: {fasta_longer}")
    print(f"Same length: {same_length}")


# Usage
fasta_dir = "data/val_test/pfam/val/clustered_split_fastas"
parquet_dir = "../data/pfam/train_test_split_parquets/val"
index_file = "../data/pfam/train_test_split_parquets/new_index.csv"

compare_sequences(fasta_dir, parquet_dir, index_file)
