import bisect
import glob
import itertools
import os
import random
from typing import Any, List, Optional

import numpy as np
import torch
from transformers import DataCollatorForLanguageModeling, PreTrainedTokenizerFast


class StringObject:
    """
    Custom class to allow for
    non-tensor elements in batch
    """

    text: List[str]

    def to(self, device):
        return self


class CustomDataCollator:
    """
    Wraps DataCollatorForLanguageModeling
    allows us to include elements which are not
    tensors with seq_len dimension, eg. dataset names
    """

    def __init__(self, tokenizer, mlm=False):
        self.base_collator = DataCollatorForLanguageModeling(tokenizer, mlm=mlm)

    def __call__(self, examples):
        non_string_data = [
            {k: v for k, v in e.items() if not isinstance(v, str)} for e in examples
        ]
        string_data = [
            {k: v for k, v in e.items() if isinstance(v, str)} for e in examples
        ]
        string_data_keys = set(k for obs in string_data for k in obs.keys())
        batch = self.base_collator(non_string_data)
        for str_key in string_data_keys:
            str_vals = [obs.get(str_key, "") for obs in string_data]
            str_obj = StringObject()
            str_obj.text = str_vals
            batch[str_key] = str_obj
        return batch


def get_flat_seq_pos_from_positions(
    positions,
    max_seq_pos: int = 1024,
    prepend_index=0,
    append_index=0,
    sep_index=0,
    num_start_tokens=1,
    num_end_tokens=1,
):
    # TODO: maybe raise exception if max_seq_pos exceeded rather than duplicating...
    if len(positions) > 0:
        flat_positions = [prepend_index] * num_start_tokens
        for sequence_positions in positions[:-1]:
            # add 1 so that sep doesnt have same position index
            flat_positions += [min(p + 1, max_seq_pos) for p in sequence_positions]
            flat_positions.append(sep_index)
        flat_positions += [min(p + 1, max_seq_pos) for p in positions[-1]]
        flat_positions += [append_index] * num_end_tokens
        return flat_positions
    else:
        return []


def get_seq_pos_from_positions(
    input_ids,
    positions,
    pad_token_id,
    max_seq_pos: int = 1024,
    num_start_tokens=1,
    num_end_tokens=1,
):
    assert input_ids.ndim == 1
    seq_pos = torch.zeros_like(input_ids)
    flat_pos = get_flat_seq_pos_from_positions(
        positions,
        max_seq_pos=max_seq_pos,
        prepend_index=0,
        append_index=0,
        sep_index=0,
        num_start_tokens=num_start_tokens,  # TODO: handle better
        num_end_tokens=num_end_tokens,
    )
    pad_any = torch.argwhere(input_ids == pad_token_id)
    if pad_any.any():
        pad_start = pad_any.min()
    else:
        pad_start = input_ids.shape[0]
    seq_pos[:pad_start] = torch.tensor(flat_pos)
    return seq_pos


def sample_to_max_tokens(
    sequences,
    extra_arrays: Optional[List[List[Any] | np.ndarray]] = None,
    max_tokens: Optional[int] = None,
    shuffle=True,
    seed: Optional[int] = None,
    drop_first: bool = False,
):
    rng = np.random.default_rng(seed)
    # TODO: implement keep first, drop first
    if drop_first:
        sequences = sequences[1:]
        if extra_arrays is not None:
            extra_arrays = [arr[1:] for arr in extra_arrays]

    if shuffle:
        perm = rng.permutation(len(sequences))
        sequences = [sequences[i] for i in perm]
        if extra_arrays is not None:
            extra_arrays = [[arr[i] for i in perm] for arr in extra_arrays]

    if max_tokens is not None:
        cumulative_lengths = list(
            itertools.accumulate([len(s) + 1 for s in sequences])
        )  # +1 for separator
        insertion_point = bisect.bisect_left(
            cumulative_lengths,
            max_tokens - 2,
        )  # -2 for doc start and end tokens
    else:
        insertion_point = len(sequences)
    if extra_arrays is None:
        return sequences[:insertion_point]
    else:
        return sequences[:insertion_point], [
            arr[:insertion_point] if arr is not None else None for arr in extra_arrays
        ]
