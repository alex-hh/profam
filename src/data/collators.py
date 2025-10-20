from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import numpy as np
from torch.utils.data import default_collate
from transformers.data.data_collator import DefaultDataCollator, default_data_collator

from src.data.objects import StringObject
from src.data.processors.batch_transforms import pack_batches


def np_flatten(
    current_feature_val, new_feature_val, separator_id=None, is_labels=False
):
    assert isinstance(
        new_feature_val, (list, np.ndarray)
    ), f"Invalid feature type: {type(new_feature_val)}"

    if is_labels:
        if isinstance(new_feature_val, list):
            new_feature_val = [separator_id] + new_feature_val[1:]
        else:
            new_feature_val = np.concatenate(
                [np.array([separator_id]), new_feature_val[1:]], axis=0
            )

    if current_feature_val is None:
        if isinstance(new_feature_val, list):
            return list(new_feature_val)
        else:
            return new_feature_val.copy()
    elif isinstance(new_feature_val, list):
        return current_feature_val + new_feature_val
    else:
        return np.concatenate([current_feature_val, new_feature_val], axis=0)


@dataclass
class DataCollatorWithFlattening(DefaultDataCollator):
    """
    Data collator used for padding free approach.

    Does the following:

    - concatate the entire mini batch into single long sequence [1, total_tokens]
    - uses `separator_id` to separate sequences within the concatenated `labels`, default value is -100
    - no padding will be added, returns `input_ids`, `labels` and `position_ids, as well
    as flattened copies of any additional features with names in `additional_features_to_flatten`.

    We assume that concatenation occurs along the first dimension for any additional features.

    Returning position ids (`return_position_ids=True`) is a good idea because the flash attention
    packing implementation relies on position ids to determine the boundaries of datapoints. If
    return_position_ids is False, a downstream model might automatically generate position ids which
    do not respect the boundaries of the concatenated sequences.
    """

    def __init__(
        self,
        *args,
        return_position_ids=True,
        additional_features_to_flatten: Optional[List[str]] = None,
        separator_id=-100,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.return_position_ids = return_position_ids
        self.additional_features_to_flatten = additional_features_to_flatten
        self.separator_id = separator_id

    @staticmethod
    def append_flattened_features(
        ret: Dict[str, Any],
        single_example_features: Dict[str, Any],
        feature_names_to_flatten: List[str],
        flatten_fn: Callable = np_flatten,
        separator_id: int = -100,
        append_position_ids: bool = True,
    ):
        for feature_name in feature_names_to_flatten:
            feature_val = single_example_features[feature_name]
            ret[feature_name] = flatten_fn(
                ret.get(feature_name, None),
                feature_val,
                separator_id=separator_id,
                is_labels=feature_name == "labels",
            )

        if "labels" not in feature_names_to_flatten:
            # add labels if not provided
            ret["labels"] = flatten_fn(
                ret.get("labels", None),
                single_example_features["input_ids"],
                separator_id=separator_id,
                is_labels=True,
            )

        if append_position_ids:
            assert "position_ids" not in feature_names_to_flatten
            position_ids = list(
                range(len(single_example_features["input_ids"]))
            )  # len compatible with all types
            if "position_ids" not in ret:
                ret["position_ids"] = position_ids
            else:
                ret["position_ids"] += position_ids
        return ret

    def _flatten_features(
        self,
        features: List[Dict[str, Any]],
        feature_names_to_flatten: List[str],
        flatten_fn: Callable,
    ):
        ret = {}
        for idx in range(0, len(features)):
            single_example_features = features[idx]
            self.append_flattened_features(
                ret=ret,
                single_example_features=single_example_features,
                flatten_fn=flatten_fn,
                feature_names_to_flatten=feature_names_to_flatten,
                separator_id=self.separator_id,
                append_position_ids=self.return_position_ids,
            )

        return ret

    def torch_flatten(self, features):
        import torch

        def flatten_single_feature(
            current_feature_val, new_feature_val, separator_id=None, is_labels=False
        ):
            assert isinstance(
                new_feature_val, (list, np.ndarray, torch.Tensor)
            ), f"Invalid feature type: {type(new_feature_val)}"
            if is_labels:
                if isinstance(new_feature_val, list):
                    new_feature_val = [separator_id] + new_feature_val[1:]
                elif isinstance(new_feature_val, np.ndarray):
                    new_feature_val = np.concatenate(
                        [np.array([separator_id]), new_feature_val[1:]], axis=0
                    )
                else:
                    new_feature_val = torch.cat(
                        [
                            torch.full((1,), separator_id).to(new_feature_val),
                            new_feature_val[1:],
                        ],
                        dim=0,
                    )

            if current_feature_val is None:
                if isinstance(new_feature_val, list):
                    return list(new_feature_val)
                elif isinstance(new_feature_val, np.ndarray):
                    return new_feature_val.copy()
                else:
                    return new_feature_val.clone()
            elif isinstance(new_feature_val, list):
                return current_feature_val + new_feature_val
            elif isinstance(new_feature_val, np.ndarray):
                return np.concatenate([current_feature_val, new_feature_val], axis=0)
            else:
                return torch.cat([current_feature_val, new_feature_val], dim=0)

        is_labels_provided = "labels" in features[0]
        feature_names_to_flatten = ["input_ids"]
        if is_labels_provided:
            feature_names_to_flatten.append("labels")
        feature_names_to_flatten += self.additional_features_to_flatten or []
        ret = self._flatten_features(
            features,
            feature_names_to_flatten,
            flatten_single_feature,
        )
        return ret

    def numpy_flatten(self, features):
        is_labels_provided = "labels" in features[0]
        feature_names_to_flatten = ["input_ids"]
        if is_labels_provided:
            feature_names_to_flatten.append("labels")
        feature_names_to_flatten += self.additional_features_to_flatten or []
        ret = self._flatten_features(
            features,
            feature_names_to_flatten,
            np_flatten,
        )
        return ret

    def tf_flatten(self, features):
        import tensorflow as tf

        def flatten_single_feature(
            current_feature_val, new_feature_val, separator_id=None, is_labels=False
        ):
            assert isinstance(
                new_feature_val, (list, np.ndarray, tf.Tensor)
            ), f"Invalid feature type: {type(new_feature_val)}"

            if is_labels:
                if isinstance(new_feature_val, list):
                    new_feature_val = [separator_id] + new_feature_val[1:]
                elif isinstance(new_feature_val, np.ndarray):
                    new_feature_val = np.concatenate(
                        [np.array([separator_id]), new_feature_val[1:]], axis=0
                    )
                else:
                    new_feature_val = tf.concat(
                        [
                            tf.fill([1], tf.cast(separator_id, new_feature_val.dtype)),
                            new_feature_val[1:],
                        ],
                        axis=0,
                    )

            if current_feature_val is None:
                if isinstance(new_feature_val, list):
                    return list(new_feature_val)
                elif isinstance(new_feature_val, np.ndarray):
                    return new_feature_val.copy()
                else:
                    return tf.identity(new_feature_val)
            elif isinstance(new_feature_val, list):
                return current_feature_val + new_feature_val
            elif isinstance(new_feature_val, np.ndarray):
                return np.concatenate([current_feature_val, new_feature_val], axis=0)
            else:
                return tf.concat([current_feature_val, new_feature_val], axis=0)

        is_labels_provided = "labels" in features[0]
        feature_names_to_flatten = ["input_ids"]
        if is_labels_provided:
            feature_names_to_flatten.append("labels")
        feature_names_to_flatten += self.additional_features_to_flatten or []
        ret = self._flatten_features(
            features,
            feature_names_to_flatten,
            flatten_single_feature,
        )
        return ret

    def torch_call(self, features):
        ret = self.torch_flatten(features)
        return default_data_collator([ret], "pt")

    def numpy_call(self, features):
        ret = self.numpy_flatten(features)
        return default_data_collator([ret], "np")

    def tf_call(self, features):
        return self.tf_flatten(features)


class DocumentBatchCollator:
    """
    N.B. HF collator was very slow for some reason (calling tolist on numpy arrays...)
    """

    def __init__(
        self,
        tokenizer,
        ignore_gaps: bool = False,
        feature_names: Optional[List[str]] = None,
        pack_to_max_tokens: Optional[int] = None,
        allow_split_packed_documents: bool = False,
    ):

        self.tokenizer = tokenizer
        self.ignore_gaps = ignore_gaps
        self.feature_names = feature_names
        self.pack_to_max_tokens = pack_to_max_tokens
        self.allow_split_packed_documents = allow_split_packed_documents

    def __call__(self, examples):
        # TODO: maybe I have an issue with blending data with different keys?
        # need to handle either in collator or by standardising in tokenizer.
        def keep_feature(feature_name):
            return self.feature_names is None or feature_name in self.feature_names

        # If packing enabled, greedily fill up to pack_to_max_tokens
        if self.pack_to_max_tokens is not None:
            chosen, remainder = [], []
            current_tokens = 0
            for ex in examples:
                n_tokens = len(ex["input_ids"])
                if (
                    current_tokens + n_tokens <= self.pack_to_max_tokens
                    or len(chosen) == 0
                ):
                    chosen.append(ex)
                    current_tokens += n_tokens
                else:
                    # print("Warning too many tokens in batch")
                    break

            combined_examples = chosen
        else:
            combined_examples = examples

        non_string_data = [
            {k: v for k, v in e.items() if (not isinstance(v, str)) and keep_feature(k)}
            for e in combined_examples
        ]

        # If packing was performed and resulted in multiple chosen examples,
        # actually pack them now into a single example dict.
        if self.pack_to_max_tokens is not None and len(non_string_data) > 1:
            # pack_batches expects a dict of lists, convert non_string_data (list of dicts)
            # It returns a dict of lists, where each list has 1 element (the packed data)
            packed_dict_of_lists = pack_batches(
                non_string_data,  # This should be a list of dicts
                max_tokens_per_batch=self.pack_to_max_tokens,
                tokenizer=self.tokenizer,
                allow_split_packed_documents=self.allow_split_packed_documents,
            )
            # Convert the dict of lists (with one item per list) to a single dict
            single_packed_example = {k: v[0] for k, v in packed_dict_of_lists.items()}
            non_string_data = [
                single_packed_example
            ]  # Now a list containing one packed dict
        elif len(non_string_data) == 0:
            # This can happen if the ring buffer was empty and no new examples came in.
            # Or if all examples were filtered out (e.g. by keep_feature)
            # Return an empty dict or handle as appropriate for downstream.
            # For now, let default_collate handle it, it might raise an error or return empty tensors.
            pass

        string_data = [
            {k: v for k, v in e.items() if isinstance(v, str) and keep_feature(k)}
            for e in combined_examples
        ]

        # Adjust string_data to match the (potentially) packed non_string_data
        if self.pack_to_max_tokens is not None and len(chosen) > 1:
            # If packing occurred, string_data needs to be reconstructed for the single packed item
            # The original string fields from all chosen examples are concatenated.
            packed_string_data = {}
            if combined_examples:  # Ensure combined_examples (chosen) is not empty
                for key in [
                    k
                    for k in combined_examples[0]
                    if isinstance(combined_examples[0][k], str)
                ]:
                    if keep_feature(key):
                        parts = [e[key] for e in combined_examples]
                        packed_string_data[key] = "$".join(parts)
            string_data = (
                [packed_string_data] if packed_string_data else []
            )  # list containing one dict of packed strings
        elif len(non_string_data) == 0:
            string_data = []

        string_data_keys = set(k for obs in string_data for k in obs.keys())

        # Pad for specified keys to handle variable lengths ofr batch_size > 1
        if len(non_string_data) > 1:
            keys_to_pad = [
                "input_ids",
                "attention_mask",
                "aa_mask",
            ]
            for key in keys_to_pad:
                existing = [d for d in non_string_data if key in d]
                if existing:
                    max_len = max(len(d[key]) for d in existing)
                    for d in non_string_data:
                        if key in d:
                            val = d[key]
                            if isinstance(val, list):
                                curr_len = len(val)
                                if curr_len < max_len:
                                    d[key] = val + [0] * (max_len - curr_len)
                            elif isinstance(val, np.ndarray):
                                curr_len = val.shape[0]
                                if curr_len < max_len:
                                    pad_width = [(0, max_len - curr_len)]
                                    pad_width += [(0, 0)] * (val.ndim - 1)
                                    d[key] = np.pad(val, pad_width, mode="constant")
        try:
            batch = default_collate(non_string_data)
        except Exception as e:
            print("Error in collator")
            print(string_data)
            # print(non_string_data)
            raise e
        labels = batch["input_ids"].clone()
        if self.tokenizer.pad_token_id is not None:
            labels[labels == self.tokenizer.pad_token_id] = -100
        if self.ignore_gaps:
            labels[labels == self.tokenizer.convert_tokens_to_ids("-")] = -100
        # dont predict mask tokens.
        labels[labels == self.tokenizer.mask_token_id] = -100
        batch["labels"] = labels
        # n.b. padding tokens should already be -100 due to base collator.
        for str_key in string_data_keys:
            str_vals = [obs.get(str_key, "") for obs in string_data]
            str_obj = StringObject()
            str_obj.text = str_vals
            batch[str_key] = str_obj

        if "batch_size" not in batch:
            batch["batch_size"] = len(combined_examples)
        # if 'train' in examples[0]['ds_name']:
        #     proportion_uniref_90 = sum(1 for ex in combined_examples if 'uniref90' in ex['ds_name']) / len(combined_examples)
        #     proportion_funfam_50 = sum(1 for ex in combined_examples if 'funfam' in ex['ds_name']) / len(combined_examples)
        #     print('buffer_len:', ring_buffer_len, batch['input_ids'].shape, 'uniref:', proportion_uniref_90, 'funfam:', proportion_funfam_50)
        # elif 'val' in examples[0]['ds_name']:
        #     print('val collate_buffer_len:', ring_buffer_len)
        return batch
