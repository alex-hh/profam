import numpy as np
import pytest

from src.data.processors.batch_transforms import pack_batches, split_example


@pytest.mark.parametrize("allow_split_packed_documents", [False, True])
@pytest.mark.parametrize("max_tokens_per_batch", [10, 20])
def test_pack_batches_without_overhangs(
    profam_tokenizer, allow_split_packed_documents, max_tokens_per_batch
):
    # trivial case where we pack each example into itself
    tokens_per_doc = 10
    arr = np.zeros(tokens_per_doc)
    arr[0] = profam_tokenizer.bos_token_id
    examples = [{"input_ids": arr.copy()} for _ in range(100)]
    packed_examples = pack_batches(
        examples,
        max_tokens_per_batch=max_tokens_per_batch,
        tokenizer=profam_tokenizer,
        allow_split_packed_documents=allow_split_packed_documents,
    )
    print(len(packed_examples["input_ids"]))
    print(packed_examples["input_ids"])
    assert len(packed_examples["input_ids"]) == len(examples) // (
        max_tokens_per_batch // tokens_per_doc
    )
    assert all(len(inp) == max_tokens_per_batch for inp in packed_examples["input_ids"])


def test_split_example(profam_tokenizer):
    arr = np.zeros(10)
    arr[0] = profam_tokenizer.bos_token_id
    example = {"input_ids": arr}
    truncated_example, example = split_example(
        example, split_at_num_tokens=5, tokenizer=profam_tokenizer
    )
    assert len(truncated_example["input_ids"]) == 5  # 5
    assert len(example["input_ids"]) == 6  # 5 + 1 for added bos


@pytest.mark.parametrize("allow_split_packed_documents", [False, True])
@pytest.mark.parametrize("max_tokens_per_batch", [5, 15])
def test_pack_batches_with_overhangs(
    profam_tokenizer, allow_split_packed_documents, max_tokens_per_batch
):
    tokens_per_doc = 9  # we add a bos token to new example when splitting, so we get 10 total per example
    if tokens_per_doc > max_tokens_per_batch and not allow_split_packed_documents:
        return None  # we can't fit any example in the packed batch
    arr = np.zeros(tokens_per_doc)
    arr[0] = profam_tokenizer.bos_token_id
    examples = [{"input_ids": arr.copy()} for _ in range(100)]
    packed_examples = pack_batches(
        examples,
        max_tokens_per_batch=max_tokens_per_batch,
        tokenizer=profam_tokenizer,
        allow_split_packed_documents=allow_split_packed_documents,
        minimum_tokens_to_split_document=1,
    )
    print(len(packed_examples["input_ids"]))
    print(packed_examples["input_ids"])
    if allow_split_packed_documents:
        expected_batch_len = max_tokens_per_batch
    else:
        expected_batch_len = tokens_per_doc
    assert all(
        len(inp) == expected_batch_len for inp in packed_examples["input_ids"][:-1]
    )  # final one might be a problem

    if max_tokens_per_batch == 5:
        # TODO: think about case for 15
        assert all(
            all(inp == arr[:5]) for inp in packed_examples["input_ids"][:-1]
        )  # final one might be a problem
