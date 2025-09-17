from src.utils.evaluation_utils import sequence_only_evaluation


sequence_only_evaluation(
    "../data/beta_lactamase/Betalactamases31_0.fasta",
    "../data/beta_lactamase/Betalactamases31_generated/Betalactamases31_0_generated.fasta",
    generate_logos=True
)