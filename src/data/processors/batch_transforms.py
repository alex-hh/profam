from typing import Dict, List

import numpy as np
import torch

from src.constants import ARRAY_TYPES, TOKENIZED_FEATURE_TYPES
from src.data.tokenizers import ProFamTokenizer
from src.data.utils import examples_list_to_dict, examples_to_list_of_dicts


def pack_examples(examples: List[Dict]):
    keys = examples[0].keys()
    packed_example = {}
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
    packed_example["original_size"] = np.mean(packed_example["original_size"])
    return packed_example


def split_example(example, split_at_num_tokens):
    split_example = {}
    for k, v in example.items():
        if k in TOKENIZED_FEATURE_TYPES and (
            isinstance(TOKENIZED_FEATURE_TYPES[k], ARRAY_TYPES)
        ):
            split_example[k] = v[:split_at_num_tokens]
            example[k] = v[split_at_num_tokens:]
        else:
            split_example[k] = v
    return split_example, example


# TODO: accept a batch_sampler (see below)
def pack_batches(
    batch_examples: Dict[str, List],
    max_tokens_per_batch: int,
    tokenizer: ProFamTokenizer,
    allow_split_packed_documents: bool = False,
):
    """Designed to be last step in batched map.
    Documents must start with a bos token.

    full_examples_only: if True, only pack examples that are full (i.e. don't split documents across batches)
    if False, split documents across batches to create packed examples with exactly max_tokens_per_batch tokens
    """
    bos_token_id = tokenizer.bos_token_id
    packed_examples = []
    examples_to_pack = []
    examples = examples_to_list_of_dicts(batch_examples)
    total_packed_tokens = 0
    for example in examples:
        if example["input_ids"][0] != bos_token_id:
            raise ValueError("Documents must start with a bos token")
        if total_packed_tokens + example["input_ids"].shape[-1] > max_tokens_per_batch:
            if allow_split_packed_documents:
                overhang_tokens = max_tokens_per_batch - total_packed_tokens
                truncated_example, example = split_example(
                    example, split_at_num_tokens=overhang_tokens
                )
                examples_to_pack.append(truncated_example)
            packed_examples.append(pack_examples(examples_to_pack))
            examples_to_pack = []
            total_packed_tokens = 0

        examples_to_pack.append(example)
        total_packed_tokens += example["input_ids"].shape[-1]
    if examples_to_pack:
        packed_examples.append(pack_examples(examples_to_pack))
    return examples_list_to_dict(packed_examples)


# def naive_concatenated_document_batch_sampler(lengths, max_tokens_per_example):
#     concatenated_length = 0
#     batch_indices = []
#     for i, l in enumerate(lengths):
#         if concatenated_length > max_tokens_per_example:
#             yield batch_indices
#             batch_indices = []
#             concatenated_length = 0
#         concatenated_length += l
#         batch_indices.append(i)
#     yield batch_indices


# def concatenate_short_documents(
#     examples,
#     batch_sampler,
#     feature_names: List[str],
#     max_tokens_per_example: Optional[int] = None,
# ):
#     """Concatenate short documents into a single example.

#     This is ultimately a bin-packing problem, if we handle it via a fixed set of examples (i.e. via batched map).
#     So batch_sampler is a bin-packing sampler.
#     An approximate bin packing solution:
#     https://github.com/imoneoi/multipack_sampler/blob/master/multipack_sampler.py
#     """
#     # TODO: use logic from DataCollatorWithFlattening
#     # advantage of doing processing here rather than in collator is that we can
#     # determine 'batch size' (number of documents) dynamically
#     additional_features_to_flatten = [
#         f
#         for f in feature_names
#         if f not in ["input_ids", "labels"] and f not in STRING_FEATURE_NAMES
#     ]

#     flattening_collator = DataCollatorWithFlattening(
#         separator_id=-100,
#         additional_features_to_flatten=additional_features_to_flatten,
#         return_position_ids=False,
#     )
#     lengths = [input_ids.shape[0] for input_ids in examples["input_ids"]]
#     examples_dicts = examples_to_list_of_dicts(examples)
#     if max_tokens_per_example is None:
#         batch_indices = [list(range(len(lengths)))]
#     else:
#         batch_indices = batch_sampler(lengths, max_tokens_per_example)
#     concatenated_examples = []
#     for batch_indices in batch_indices:
#         batch_examples = [examples_dicts[i] for i in batch_indices]
#         concatenated_example = flattening_collator.numpy_flatten(batch_examples)
#         for f in STRING_FEATURE_NAMES:
#             concatenated_example[f] = "-".join([ex[f] for ex in batch_examples])
#         concatenated_examples.append(concatenated_example)
#     return examples_list_to_dict(concatenated_examples)
