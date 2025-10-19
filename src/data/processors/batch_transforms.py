import warnings
from typing import Dict, List, Union

import numpy as np
import torch

from src.constants import ARRAY_TYPES, TOKENIZED_FEATURE_TYPES
from src.data.tokenizers import ProFamTokenizer
from src.data.utils import examples_list_to_dict, examples_to_list_of_dicts


def pack_examples(examples: List[Dict]):
    keys = examples[0].keys()
    packed_example = {}
    batch_size = len(examples)
    for example in examples:
        for k in keys:
            if isinstance(example[k], torch.Tensor):
                if k in packed_example:
                    packed_example[k] = torch.cat(
                        [packed_example[k], example[k]], dim=0
                    )
                else:
                    packed_example[k] = example[k].clone()
            elif isinstance(example[k], np.ndarray):
                if k in packed_example:
                    packed_example[k] = np.concatenate(
                        [packed_example[k], example[k]], axis=0
                    )
                else:
                    packed_example[k] = example[k].copy()
            elif isinstance(example[k], list):
                if k in packed_example:
                    packed_example[k] += example[k]
                else:
                    packed_example[k] = example[k][:]
            elif isinstance(example[k], str):
                # n.b. this will break document metrics based on these strings
                if k in packed_example:
                    packed_example[k] += "-" + example[k]
                else:
                    packed_example[k] = str(example[k])
            elif k in ["original_size"]:
                if k in packed_example:
                    packed_example[k].append(example[k])
                else:
                    packed_example[k] = [example[k]]
            else:
                raise ValueError(f"Unsupported type: {type(example[k])}")
    if "original_size" in packed_example:
        packed_example["original_size"] = np.mean(packed_example["original_size"])
    packed_example["batch_size"] = batch_size
    return packed_example


def split_example(example, split_at_num_tokens, tokenizer):
    split_example_pre = {}
    assert example["input_ids"][0] == tokenizer.bos_token_id
    for k, v in example.items():
        if k in TOKENIZED_FEATURE_TYPES and (
            isinstance(TOKENIZED_FEATURE_TYPES[k], ARRAY_TYPES)
        ):
            bos_val = v[:1]
            split_example_pre[k] = v[:split_at_num_tokens]
            example[k] = np.concatenate([bos_val, v[split_at_num_tokens:]])
        else:
            split_example_pre[k] = v
    return split_example_pre, example


# TODO: accept a batch_sampler (see below)
def pack_batches(
    batch_examples: Union[Dict[str, List], List[Dict]],
    max_tokens_per_batch: int,
    tokenizer: ProFamTokenizer,
    allow_split_packed_documents: bool = False,
    minimum_tokens_to_split_document: int = 10,
):
    """Designed to be last step in batched map.
    Documents must start with a bos token.
    Returns a dict of lists, since this is required by datasets batched map.

    allow_split_packed_documents: if False, only pack examples that are full (i.e. don't split documents across batches)
    if True, split documents across batches to create packed examples with exactly max_tokens_per_batch tokens
    minimum_tokens_to_split_document: if allow_split_packed_documents is True, split documents
    if there is at least minimum_tokens_to_split_document tokens in the overhang
    """
    if allow_split_packed_documents:
        warnings.warn("allow_split_packed_documents is not thoroughly tested")
    bos_token_id = tokenizer.bos_token_id
    packed_examples = []
    examples_to_pack = []
    if isinstance(batch_examples, dict):
        examples = examples_to_list_of_dicts(batch_examples)
    else:
        assert isinstance(batch_examples, list) and isinstance(batch_examples[0], dict)
        examples = batch_examples
    total_packed_tokens = 0
    for example in examples:
        if example["input_ids"][0] != bos_token_id:
            raise ValueError("Documents must start with a bos token")

        if total_packed_tokens + example["input_ids"].shape[-1] > max_tokens_per_batch:
            # we can't fit the example in the packed batch
            # if we allow splitting documents, split the document if there is enough space for
            # minimum_tokens_to_split_document tokens
            overhang_tokens = max_tokens_per_batch - total_packed_tokens
            if (
                allow_split_packed_documents
                and overhang_tokens >= minimum_tokens_to_split_document
            ):
                truncated_example, example = split_example(
                    example, split_at_num_tokens=overhang_tokens, tokenizer=tokenizer
                )
                examples_to_pack.append(truncated_example)

            # add the packed batch to the list of packed batches
            packed_examples.append(pack_examples(examples_to_pack))

            # initialise the next packed batch with whatever of the current example couldn't fit in the previous packed batch
            examples_to_pack = []
            total_packed_tokens = 0
            if (
                example["input_ids"].shape[-1] <= max_tokens_per_batch
            ):  # should be always true
                examples_to_pack.append(example)
                total_packed_tokens += example["input_ids"].shape[-1]
        else:
            # we can fit the example in the packed batch -
            # add the example to the list of examples to pack and continue
            examples_to_pack.append(example)
            total_packed_tokens += example["input_ids"].shape[-1]
    if examples_to_pack:
        packed_examples.append(pack_examples(examples_to_pack))
    return examples_list_to_dict(packed_examples)
