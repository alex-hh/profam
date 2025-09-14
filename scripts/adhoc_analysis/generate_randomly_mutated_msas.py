"""
Created by Jude Wells 2025-09-13

Generates randomly mutated MSAs for the foldseek families.
so that we can then predict the structures of the randomly
mutated sequences and see what is the baseline performance 
on metrics such as TM score and LDDT.

Protocol:
Sample a random sequence from the prompt fasta file.
Sample a random mutation rate (p_mut) uniformly from 0.01 to 0.99
at each position apply a mutation with probability p_mut
the mutation is chosen uniformly from the 20 standard amino acids
save the mutated sequence to a fasta file.
"""

import glob
import random
import os
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

def generate_randomly_mutated_seqs(prompt_fasta, output_dir):
    """
    For a given input FASTA, pick 2 sequences at random (with replacement),
    independently sample mutation rates, mutate, and write two output FASTAs.
    Returns a list of basenames for the created FASTA files.
    """
    records = list(SeqIO.parse(prompt_fasta, "fasta"))
    if not records:
        return []

    base_name = os.path.splitext(os.path.basename(prompt_fasta))[0]
    created_basenames = []

    standard_amino_acids = list("ACDEFGHIKLMNPQRSTVWY")
    allowed_set = set(standard_amino_acids)

    for i in range(1):
        original_record = random.choice(records)
        original_sequence = str(original_record.seq).upper().replace("-", "")

        # Sample mutation rate uniformly from 0.01 to 0.99, independently for each pick
        p_mut = random.uniform(0.01, 0.99)

        mutated_chars = []
        for residue in original_sequence:
            if residue in allowed_set and random.random() < p_mut:
                candidates = [aa for aa in standard_amino_acids if aa != residue]
                mutated_chars.append(random.choice(candidates))
            else:
                mutated_chars.append(residue)

        mutated_sequence = "".join(mutated_chars)

        mutated_record = SeqRecord(
            Seq(mutated_sequence),
            id=f"{original_record.id}|random_mutation|p_mut={p_mut:.3f}",
            description="",
        )

        output_fasta = os.path.join(output_dir, f"{base_name}.fasta")
        SeqIO.write([mutated_record], output_fasta, "fasta")
        created_basenames.append(os.path.basename(output_fasta))

    return created_basenames

if __name__=="__main__":
    prompt_pattern = glob.glob(f"../data/val_test_v2_fastas/foldseek/*/*.fasta")
    output_dir = "../sampling_results/randomly_mutated_sequences"
    os.makedirs(output_dir, exist_ok=True)

    all_created_basenames = []
    for prompt_fasta in prompt_pattern:
        created = generate_randomly_mutated_seqs(prompt_fasta, output_dir)
        all_created_basenames.extend(created)

    # Write list of created FASTA basenames
    list_path = os.path.join(output_dir, "fasta_file_list.txt")
    with open(list_path, "w") as f:
        for name in all_created_basenames:
            f.write(name + "\n")