import pandas as pd
import pytest

from src.data.fasta import convert_sequence_with_positions, read_fasta_sequences


@pytest.fixture
def pfam_example_text():
    """Fixture to load the sample MSA data."""
    df = pd.read_parquet("data/example_data/pfam/Domain_60429258_61033370.parquet")
    return df.iloc[0]["text"]


def get_sequence_match_positions(sequence):
    sequence_index = 0  # relative to raw sequence
    raw_seq_match_positions = []
    for aa in sequence:
        if aa.isupper() or aa == "-":
            raw_seq_match_positions.append(sequence_index)
        sequence_index += 1
    return raw_seq_match_positions


def test_match_state_positions(pfam_example_text):
    """Check that all sequences have same position ids in all match state positions."""
    sequences = list(
        read_fasta_sequences(
            pfam_example_text.split("\n"),
            keep_gaps=True,
            keep_insertions=True,
            to_upper=False,
        )
    )
    _, positions, is_match = convert_sequence_with_positions(
        sequences[0], keep_gaps=True, keep_insertions=True, to_upper=True
    )
    # n.b. these start from 1 because 0 represents pre-match inserts
    match_positions = [pos for pos, is_match in zip(positions, is_match) if is_match]
    for raw_seq in sequences:
        # we need keep_gaps=True to test in this way (i.e. via a match mask)
        _, positions, is_match = convert_sequence_with_positions(
            raw_seq, keep_gaps=True, keep_insertions=True, to_upper=True
        )
        _match_positions = [
            pos for pos, is_match in zip(positions, is_match) if is_match
        ]
        assert tuple(_match_positions) == tuple(match_positions)


class TestSequencePositions:
    def test_raw_positions(self):
        sequences = ["ABC", "DEFD", "GHIJMEKJF"]
        positions = [[1, 2, 3], [1, 2, 3, 4], [1, 2, 3, 4, 5, 6, 7, 8, 9]]
        for seq, pos in zip(sequences, positions):
            inferred_pos = convert_sequence_with_positions(
                seq, keep_gaps=False, keep_insertions=True, to_upper=True
            )[1]
            inferred_pos_nomsa = convert_sequence_with_positions(
                seq,
                keep_gaps=False,
                keep_insertions=True,
                to_upper=True,
                use_msa_pos=False,
            )[1]
            assert tuple(inferred_pos) == tuple(pos)
            assert tuple(inferred_pos_nomsa) == tuple(pos)

    def test_msa_positions_no_gaps(self):
        sequences = ["aB..-C", "DEF", "GdfkHIJm--F"]
        positions = [[0, 1, 3], [1, 2, 3], [1, 1, 1, 1, 2, 3, 4, 4, 7]]
        for seq, pos in zip(sequences, positions):
            inferred_pos = convert_sequence_with_positions(
                seq,
                keep_gaps=False,
                keep_insertions=True,
                to_upper=True,
                use_msa_pos=True,
            )[1]
            assert tuple(inferred_pos) == tuple(pos)

    def test_msa_positions_with_gaps(self):
        sequences = ["aB..-C", "DEF", "GdfkHIJm--F"]
        positions = [[1, 2, 3], [1, 2, 3], [1, 2, 3, 4, 5, 6, 7]]
        for seq, pos in zip(sequences, positions):
            inferred_pos = convert_sequence_with_positions(
                seq,
                keep_gaps=True,
                keep_insertions=False,
                to_upper=True,
            )[1]
            assert tuple(inferred_pos) == tuple(pos)

    def test_msa_positions_not_msa_relative(self):
        sequences = ["aB..-C", "DEF", "GdfkHIJm--F"]
        positions = [[1, 2, 3], [1, 2, 3], [1, 2, 3, 4, 5, 6, 7, 8, 9]]
        for seq, pos in zip(sequences, positions):
            inferred_pos = convert_sequence_with_positions(
                seq,
                keep_gaps=False,
                keep_insertions=True,
                to_upper=True,
                use_msa_pos=False,
            )[1]
            assert tuple(inferred_pos) == tuple(pos)

    def test_msa_positions_with_gaps_not_msa_relative(self):
        sequences = ["aB..-C", "DEF", "GdfkHIJm--F"]
        positions = [[1, 2, 3], [1, 2, 3], [1, 2, 3, 4, 5, 6, 7]]
        for seq, pos in zip(sequences, positions):
            inferred_pos = convert_sequence_with_positions(
                seq,
                keep_gaps=True,
                keep_insertions=False,
                to_upper=True,
                use_msa_pos=False,
            )[1]
            assert tuple(inferred_pos) == tuple(pos)
