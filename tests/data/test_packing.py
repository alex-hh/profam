import numpy as np
import pytest

from src.data.processors.batch_transforms import pack_batches

@pytest.mark.parametrize("allow_split_packed_documents", [False, True])
@pytest.mark.parametrize("max_tokens_per_batch", [10, 20])
def test_pack_batches_without_overhangs(profam_tokenizer, allow_split_packed_documents, max_tokens_per_batch):
    # trivial case where we pack each example into itself
    arr = np.zeros(10)
    arr[0] = profam_tokenizer.bos_token_id
    examples = [
        {"input_ids": arr}
    ] * 100
    packed_examples = pack_batches(
        examples, max_tokens_per_batch=max_tokens_per_batch, tokenizer=profam_tokenizer, allow_split_packed_documents=allow_split_packed_documents
    )
    assert all(len(inp) == max_tokens_per_batch for inp in packed_examples["input_ids"])
    assert len(packed_examples["input_ids"]) == len(examples) // (max_tokens_per_batch // 10)
