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
        seq_3d + "[SEQ-STRUCT-SEP]" + seq
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
        tokenized.input_ids
        == profam_tokenizer.convert_tokens_to_ids("[SEQ-STRUCT-SEP]")
    ).sum() == len(example_sequences)
