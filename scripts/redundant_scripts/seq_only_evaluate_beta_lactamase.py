from src.utils.evaluation_utils import sequence_only_evaluation
from Bio import SeqIO

from data_creation_scripts.train_test_split_v2_make_fasta_files import run_hhfilter

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

# new_generated_path = filter_bad_length_sequences(
#     prompt_fasta="../data/beta_lactamase/Betalactamases31_0.fasta",
#     generated_fasta="../data/beta_lactamase/Betalactamase_4BLM_generated_single/Betalactamase_4BLM_generated.fasta",
# )

# sequence_only_evaluation(
#     prompt_fasta="../data/beta_lactamase/Betalactamases31_0.fasta",
#     generated_fasta=new_generated_path,
#     generate_logos=True
# )