import pytest

from src.data.fasta import read_fasta_sequences
from src.data.utils import sample_to_max_tokens
from src.utils.tokenizers import ProFamTokenizer


@pytest.fixture
def tokenizer():
    _tok = ProFamTokenizer(
        tokenizer_file="src/data/components/profam_tokenizer.json",
        unk_token="[UNK]",
        pad_token="[PAD]",
        bos_token="[start-of-document]",
        sep_token="[SEP]",
        mask_token="[MASK]",
        add_special_tokens=True,
        add_final_sep=True,
        add_bos_token=True,
        add_document_type_token=True,
        use_seq_pos=True,
        max_seq_pos=1024,
        max_tokens=5000,
    )
    return _tok


def test_encode_decode(tokenizer, pfam_fasta_text):
    lines = pfam_fasta_text.split("\n")
    sequence_iterator = read_fasta_sequences(
        lines,
        # preserve original sequences before getting positions
        keep_gaps=False,
        keep_insertions=True,
        to_upper=True,
    )
    sequences = sample_to_max_tokens(
        list(sequence_iterator), max_tokens=tokenizer.max_tokens
    )
    # n.b. encode_sequences encodes as a sequence of sequences
    encoded = tokenizer.encode_sequences(sequences).input_ids
    decoded = tokenizer.decode_tokens(encoded.unsqueeze(0))[
        0
    ]  # decode_tokens returns a list of lists
    for input_seq, decoded_seq in zip(sequences, decoded):
        assert input_seq == decoded_seq
