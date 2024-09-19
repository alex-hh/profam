from src.sequence.fasta import read_fasta_sequences
from src.data.objects import ProteinDocument
from src.data.transforms import sample_to_max_tokens


def test_encode_decode(profam_tokenizer, pfam_fasta_text):
    lines = pfam_fasta_text.split("\n")
    sequence_iterator = read_fasta_sequences(
        lines,
        # preserve original sequences before getting positions
        keep_gaps=False,
        keep_insertions=True,
        to_upper=True,
    )
    proteins = sample_to_max_tokens(
        ProteinDocument(sequences=list(sequence_iterator)),
        max_tokens=profam_tokenizer.max_tokens,
        extra_tokens_per_document=2,
    )
    # n.b. encode_sequences encodes as a sequence of sequences
    encoded = profam_tokenizer.encode(proteins).input_ids
    decoded = profam_tokenizer.decode_tokens(encoded.unsqueeze(0))[
        0
    ]  # decode_tokens returns a list of lists
    for input_seq, decoded_seq in zip(proteins.sequences, decoded):
        assert input_seq == decoded_seq


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


def test_interleaved_sequence_structure_tokenization(profam_tokenizer):
    # TODO: make this use encode sequences and test encode decode
    example_sequences = ["ARNDC", "QEGHIL", "KMFPST", "WYV"]
    example_3dis = [s.lower() for s in example_sequences]
    sequences = [
        seq_3d + profam_tokenizer.seq_struct_sep_token + seq
        for seq_3d, seq in zip(example_sequences, example_3dis)
    ]
    concatenated_sequence = (
        "[RAW]" + profam_tokenizer.bos_token + "[SEP]".join(sequences) + "[SEP]"
    )
    tokenized = profam_tokenizer(
        concatenated_sequence,
        return_tensors="pt",
        truncation=False,
        max_length=100,
        padding="max_length",
        add_special_tokens=False,
    )
    assert (
        tokenized.input_ids == profam_tokenizer.seq_struct_sep_token_id
    ).sum() == len(example_sequences)
    # TODO: test aa mask
