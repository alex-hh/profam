from Bio import SeqIO
import os
import glob
from src.utils.evaluation_utils import sequence_only_evaluation


def filter_bad_length_sequences(prompt_fasta, generated_fasta):
    prompt_sequences = list(SeqIO.parse(prompt_fasta, "fasta"))
    min_prompt_length = min(len(seq) for seq in prompt_sequences) * 0.9
    max_prompt_length = max(len(seq) for seq in prompt_sequences) * 1.15
    generated_sequences = list(SeqIO.parse(generated_fasta, "fasta"))
    new_generated_path = generated_fasta.replace(".fasta", "_filtered_by_length.fasta")
    with open(new_generated_path, "w") as fh:
        for seq in generated_sequences:
            if len(seq) >= min_prompt_length and len(seq) <= max_prompt_length:
                fh.write(f">{seq.id}\n{seq.seq}\n")
    return new_generated_path


def combine_beta_lactamase(input_dir, output_path):
    if os.path.exists(output_path):
        return output_path
    input_paths = sorted(glob.glob(os.path.join(input_dir, "*.fasta")))
    all_records = []
    all_sequences = set()
    for input_path in input_paths:
        records = list(SeqIO.parse(input_path, "fasta"))
        for record in records:
            seq_str = str(record.seq)
            if seq_str in all_sequences:
                continue
            all_sequences.add(seq_str)
            all_records.append(record)

    # Rename records to have consecutive, unique identifiers to avoid clashes
    for idx, record in enumerate(all_records):
        ll = record.id.split("_log_likelihood_")[1]
        new_id = f"Betalactamase_sample_{idx}_log_likelihood_{ll}"
        record.id = new_id
        record.name = new_id
        record.description = new_id
    with open(output_path, "w") as fh:
        SeqIO.write(all_records, fh, "fasta")
    return output_path


combined_path = "../data/beta_lactamase/Betalactamases31_generated_ensemble/Betalactamases31_generated_ensemble_combined.fasta"

combine_beta_lactamase(
    input_dir="../data/beta_lactamase/Betalactamases31_generated_ensemble",
    output_path=combined_path)

new_generated_path = filter_bad_length_sequences(
    prompt_fasta="../data/beta_lactamase/Betalactamases31_0.fasta",
    generated_fasta=combined_path,
)

sequence_only_evaluation(
    prompt_fasta="../data/beta_lactamase/Betalactamases31_0.fasta",
    generated_fasta=new_generated_path,
    generate_logos=True
)