import pandas as pd
import pytest

from src.data.processors.transforms import (
    convert_aligned_sequence_adding_positions,
    convert_raw_sequence_adding_positions,
)
from src.sequence.fasta import read_fasta_sequences


def get_sequence_match_positions(sequence):
    sequence_index = 0  # relative to raw sequence
    raw_seq_match_positions = []
    for aa in sequence:
        if aa.isupper() or aa == "-":
            raw_seq_match_positions.append(sequence_index)
        sequence_index += 1
    return raw_seq_match_positions


class TestSequencePositions:
    def test_raw_positions(self):
        sequences = ["ABC", "DEFD", "GHIJMEKJF"]
        positions = [[1, 2, 3], [1, 2, 3, 4], [1, 2, 3, 4, 5, 6, 7, 8, 9]]
        for seq, pos in zip(sequences, positions):
            inferred_pos = convert_raw_sequence_adding_positions(seq)[1]
            inferred_pos_nomsa = convert_raw_sequence_adding_positions(seq)[1]
            assert tuple(inferred_pos) == tuple(pos)
            assert tuple(inferred_pos_nomsa) == tuple(pos)

    def test_msa_positions_no_gaps(self):
        sequences = ["aB..-C", "DEF", "GdfkHIJm--F"]
        positions = [[0, 1, 3], [1, 2, 3], [1, 1, 1, 1, 2, 3, 4, 4, 7]]
        for seq, pos in zip(sequences, positions):
            inferred_pos = convert_aligned_sequence_adding_positions(
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
            inferred_pos = convert_aligned_sequence_adding_positions(
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
            inferred_pos = convert_aligned_sequence_adding_positions(
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
            inferred_pos = convert_aligned_sequence_adding_positions(
                seq,
                keep_gaps=True,
                keep_insertions=False,
                to_upper=True,
                use_msa_pos=False,
            )[1]
            assert tuple(inferred_pos) == tuple(pos)
