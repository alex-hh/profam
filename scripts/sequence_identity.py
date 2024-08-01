import sys

from Bio import SeqIO
from Bio import pairwise2 as pw2


def compute_pairwise_identity(seq1, seq2):
    """
    Computes the pairwise sequence identity between two sequences.
    """
    # https://www.biostars.org/p/208540/
    global_align = pw2.align.globalxx(seq1, seq2)
    seq_length = min(len(seq1), len(seq2))
    matches = global_align[0][2]
    percent_match = (matches / seq_length) * 100
    # print(global_align[0])
    return percent_match


def compute_average_pairwise_identity(fasta_file):
    """
    Computes the average pairwise sequence identity for sequences in a FASTA file.
    """
    sequences = list(SeqIO.parse(fasta_file, "fasta"))
    num_sequences = len(sequences)
    if num_sequences < 2:
        raise ValueError("Need at least two sequences to compute pairwise identities.")

    total_identity = 0.0
    count = 0

    for i in range(num_sequences):
        for j in range(i + 1, num_sequences):
            seq1 = sequences[i].seq
            seq2 = sequences[j].seq
            identity = compute_pairwise_identity(seq1, seq2)
            total_identity += identity
            count += 1

    average_identity = total_identity / count
    return average_identity


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python average_pairwise_identity.py <fasta_file>")
        sys.exit(1)

    fasta_file = sys.argv[1]
    try:
        average_identity = compute_average_pairwise_identity(fasta_file)
        print(f"Average pairwise sequence identity: {average_identity:.4f}")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
