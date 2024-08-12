import pytest

from src.data.fasta import convert_sequence_with_positions, read_fasta_sequences


@pytest.fixture
def sample_msa():
    """Fixture to load the sample MSA data."""
    with open("test_data/sample_msa.txt") as f:
        msa_data = f.read()
    return msa_data


def test_match_state_positions(test_openfold_msa):
    """Check that all sequences have same position ids in all match state positions."""
    sequences = list(
        read_fasta_sequences(
            test_openfold_msa, keep_gaps=True, keep_insertions=True, to_upper=False
        )
    )
    match_positions = [i for i, c in enumerate(sequences[0]) if c.isupper() or c == "-"]
    for seq in sequences:
        _, positions = convert_sequence_with_positions(
            seq, keep_gaps=False, keep_insertions=True, to_upper=True
        )
        _match_states = [i for i, c in enumerate(seq) if c.isupper()]
        _match_positions = [positions[i] for i in _match_states]
        assert tuple(match_positions) == tuple(_match_positions)
