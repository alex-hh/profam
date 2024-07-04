import pytest
import torch
from src.data.utils import get_seq_pos
from transformers import PreTrainedTokenizerFast


# Mock tokenizer for testing
@pytest.fixture
def mock_tokenizer():
    return PreTrainedTokenizerFast(
        tokenizer_file="path/to/tokenizer.json",
        unk_token="[UNK]",
        pad_token="[PAD]",
        bos_token="[start-of-document]",
        sep_token="[SEP]",
        mask_token="[MASK]",
    )


# Test get_seq_pos function
class TestGetSeqPos:
    def test_basic_functionality(self):
        input_ids = torch.tensor(      [0, 1, 2, 3, 27, 4, 5, 6, 27, 7, 8])
        expected_output = torch.tensor([0, 0, 1, 2, 0,  0, 1, 2, 0,  0, 1])
        assert torch.all(get_seq_pos(input_ids, sep_token_id=27) == expected_output)

    def test_max_seq_pos(self):
        input_ids = torch.tensor([0] + list(range(1, 25)) + [27, 3, 4, 5])
        expected_output = torch.tensor([0] + [0,1,2,3] + [3] * 20 + [0, 0, 1, 2])
        assert torch.all(get_seq_pos(input_ids, sep_token_id=27, max_seq_pos=4) == expected_output)

    def test_multiple_sequences(self):
        input_ids = torch.tensor(      [0, 1, 2, 27, 3, 4, 5, 27, 6, 7])
        expected_output = torch.tensor([0, 0, 1, 0,  0, 1, 2, 0,  0, 1])
        assert torch.all(get_seq_pos(input_ids, sep_token_id=27) == expected_output)

    def test_no_sep_token(self):
        input_ids = torch.tensor(      [0, 1, 2, 3, 4, 5])
        expected_output = torch.tensor([0, 0, 1, 2, 3, 4])
        assert torch.all(get_seq_pos(input_ids, sep_token_id=27) == expected_output)

    def test_all_sep_tokens(self):
        input_ids = torch.tensor([27, 27, 27, 27])
        expected_output = torch.tensor([0, 0, 0, 0])
        assert torch.all(get_seq_pos(input_ids, sep_token_id=27) == expected_output)

