from src.data.fasta import read_fasta_sequences
from src.data.utils import sample_to_max_tokens


def test_encode_decode(profam_tokenizer_seqpos, pfam_fasta_text):
    lines = pfam_fasta_text.split("\n")
    sequence_iterator = read_fasta_sequences(
        lines,
        # preserve original sequences before getting positions
        keep_gaps=False,
        keep_insertions=True,
        to_upper=True,
    )
    sequences = sample_to_max_tokens(
        list(sequence_iterator), max_tokens=profam_tokenizer_seqpos.max_tokens
    )
    # n.b. encode_sequences encodes as a sequence of sequences
    encoded = profam_tokenizer_seqpos.encode_sequences(sequences).input_ids
    decoded = profam_tokenizer_seqpos.decode_tokens(encoded.unsqueeze(0))[
        0
    ]  # decode_tokens returns a list of lists
    for input_seq, decoded_seq in zip(sequences, decoded):
        assert input_seq == decoded_seq


def test_sequence_of_sequence_tokenization(profam_tokenizer_seqpos):
    example_sequences = ["ARNDC", "QEGHIL", "KMFPST", "WYV"]
    concatenated_sequence = (
        "[RAW]"
        + profam_tokenizer_seqpos.bos_token
        + "[SEP]".join(example_sequences)
        + "[SEP]"
    )
    tokenized = profam_tokenizer_seqpos(
        concatenated_sequence,
        return_tensors="pt",
        truncation=False,
        max_length=100,
        padding="max_length",
        add_special_tokens=False,
    )
    # TODO: extend...
    assert tokenized.input_ids[0, 0] == profam_tokenizer_seqpos.convert_tokens_to_ids(
        "[RAW]"
    )
    assert not (
        tokenized["input_ids"] == profam_tokenizer_seqpos.convert_tokens_to_ids("[UNK]")
    ).any()
