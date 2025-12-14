from src.data.objects import ProteinDocument
from src.data.processors.transforms import (
    preprocess_raw_sequences_sampling_to_max_tokens,
)
from src.sequence.fasta import read_fasta_sequences


def test_sequence_of_sequence_tokenization(profam_tokenizer):
    example_sequences = ["ARNDC", "QEGHIL", "KMFPST", "WYV"]
    concatenated_sequence = (
        "[RAW]" + profam_tokenizer.bos_token + "[SEP]".join(example_sequences) + "[SEP]"
    )
    tokenized = profam_tokenizer(
        concatenated_sequence,
        return_tensors="pt",
        truncation=False,
        max_length=100,
        padding="max_length",
        add_special_tokens=False,
    )
    # TODO: extend...
    assert tokenized.input_ids[0, 0] == profam_tokenizer.convert_tokens_to_ids("[RAW]")
    assert not (
        tokenized["input_ids"] == profam_tokenizer.convert_tokens_to_ids("[UNK]")
    ).any()
