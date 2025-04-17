from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import numpy as np
from torch.utils.data import default_collate
from transformers.data.data_collator import DefaultDataCollator, default_data_collator

from src.data.objects import StringObject


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
    ):
        self.tokenizer = tokenizer
        self.ignore_gaps = ignore_gaps
        self.feature_names = feature_names

    def __call__(self, examples):
        # TODO: maybe I have an issue with blending data with different keys?
        # need to handle either in collator or by standardising in tokenizer.
        def keep_feature(feature_name):
            return self.feature_names is None or feature_name in self.feature_names

        non_string_data = [
            {k: v for k, v in e.items() if (not isinstance(v, str)) and keep_feature(k)}
            for e in examples
        ]
        # TODO: handle Nones
        string_data = [
            {k: v for k, v in e.items() if isinstance(v, str) and keep_feature(k)}
            for e in examples
        ]
        string_data_keys = set(k for obs in string_data for k in obs.keys())
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
        return batch
