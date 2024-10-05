from src.data.objects import ProteinDocument
from src.data.processors.transforms import filter_by_length


# TODO: test filtering
def test_length_filter():
    sequences = ["ARG", "PMMPMM", "RP"]
    proteins = ProteinDocument(sequences=sequences)
    filtered_proteins = filter_by_length(proteins, min_length=3, max_length=5)
    assert len(filtered_proteins) == 1
    assert filtered_proteins[0].sequence == "ARG"

    filtered_proteins = filter_by_length(proteins, max_length=3)
    assert len(filtered_proteins) == 2
    assert filtered_proteins[0].sequence == "ARG"
    assert filtered_proteins[1].sequence == "RP"
