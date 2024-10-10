from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from torch.utils.data import default_collate
from transformers import DefaultDataCollator
from transformers.data.data_collator import default_data_collator

from src.data.objects import StringObject


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

    def _torch_flatten(self, ret, feature_name, orig_feature_val, separator_id=None):
        import torch

        assert isinstance(
            orig_feature_val, (list, np.ndarray, torch.Tensor)
        ), f"Invalid feature type: {type(orig_feature_val)}"
        feature_val = orig_feature_val
        if feature_name == "labels":
            if isinstance(orig_feature_val, list):
                feature_val = [separator_id] + orig_feature_val[1:]
            elif isinstance(orig_feature_val, np.ndarray):
                feature_val = np.concatenate(
                    [np.array([separator_id]), orig_feature_val[1:]], axis=0
                )
            else:
                feature_val = torch.cat(
                    [
                        torch.full((1,), separator_id).to(orig_feature_val),
                        orig_feature_val[1:],
                    ],
                    dim=0,
                )

        if feature_name not in ret:
            if isinstance(feature_val, list):
                ret[feature_name] = list(feature_val)
            elif isinstance(feature_val, np.ndarray):
                ret[feature_name] = feature_val.copy()
            else:
                ret[feature_name] = feature_val.clone()
        elif isinstance(feature_val, list):
            ret[feature_name] += feature_val
        elif isinstance(feature_val, np.ndarray):
            ret[feature_name] = np.concatenate([ret[feature_name], feature_val], axis=0)
        else:
            ret[feature_name] = torch.cat([ret[feature_name], feature_val], dim=0)
        return ret

    def _np_flatten(self, ret, feature_name, orig_feature_val, separator_id=None):
        assert isinstance(
            orig_feature_val, (list, np.ndarray)
        ), f"Invalid feature type: {type(orig_feature_val)}"

        feature_val = orig_feature_val
        if feature_name == "labels":
            if isinstance(orig_feature_val, list):
                feature_val = [separator_id] + orig_feature_val[1:]
            else:
                feature_val = np.concatenate(
                    [np.array([separator_id]), orig_feature_val[1:]], axis=0
                )

        if feature_name not in ret:
            if isinstance(feature_val, list):
                ret[feature_name] = list(feature_val)
            else:
                ret[feature_name] = feature_val.copy()
        elif isinstance(feature_val, list):
            ret[feature_name] += feature_val
        else:
            ret[feature_name] = np.concatenate([ret[feature_name], feature_val], axis=0)
        return ret

    def _tf_flatten(self, ret, feature_name, orig_feature_val, separator_id=None):
        import tensorflow as tf

        assert isinstance(
            orig_feature_val, (list, np.ndarray, tf.Tensor)
        ), f"Invalid feature type: {type(orig_feature_val)}"

        feature_val = orig_feature_val
        if feature_name == "labels":
            if isinstance(orig_feature_val, list):
                feature_val = [separator_id] + orig_feature_val[1:]
            elif isinstance(orig_feature_val, np.ndarray):
                feature_val = np.concatenate(
                    [[separator_id], orig_feature_val[1:]], axis=0
                )
            else:
                feature_val = tf.concat(
                    [
                        tf.fill([1], tf.cast(separator_id, orig_feature_val.dtype)),
                        orig_feature_val[1:],
                    ],
                    axis=0,
                )

        if feature_name not in ret:
            if isinstance(feature_val, list):
                ret[feature_name] = list(feature_val)
            elif isinstance(feature_val, np.ndarray):
                ret[feature_name] = feature_val.copy()
            else:
                ret[feature_name] = tf.identity(feature_val)
        elif isinstance(feature_val, list):
            ret[feature_name] += feature_val
        elif isinstance(feature_val, np.ndarray):
            ret[feature_name] = np.concatenate([ret[feature_name], feature_val], axis=0)
        else:
            ret[feature_name] = tf.concat([ret[feature_name], feature_val], axis=0)
        return ret

    def __call__(self, features, return_tensors=None, separator_id=None):
        if return_tensors is None:
            return_tensors = self.return_tensors

        if separator_id is None:
            separator_id = self.separator_id
        is_labels_provided = "labels" in features[0]
        feature_names_to_flatten = ["input_ids"]
        if is_labels_provided:
            feature_names_to_flatten.append("labels")
        feature_names_to_flatten += self.additional_features_to_flatten or []
        # use return tensors to determine what input types to check for
        ret = {}
        for idx in range(0, len(features)):
            for feature_name in feature_names_to_flatten:
                feature_val = features[idx][feature_name]
                if return_tensors == "pt":
                    self._torch_flatten(ret, feature_name, feature_val, separator_id)
                elif return_tensors == "np":
                    self._np_flatten(ret, feature_name, feature_val, separator_id)
                elif return_tensors == "tf":
                    self._tf_flatten(ret, feature_name, feature_val, separator_id)
                else:
                    raise ValueError(f"Invalid return_tensors value: {return_tensors}")

            # add labels if not provided
            if not is_labels_provided:
                print(features[idx]["input_ids"], "features[idx][input_ids]")
                if return_tensors == "pt":
                    self._torch_flatten(
                        ret, "labels", features[idx]["input_ids"], separator_id
                    )
                elif return_tensors == "np":
                    self._np_flatten(
                        ret, "labels", features[idx]["input_ids"], separator_id
                    )
                elif return_tensors == "tf":
                    self._tf_flatten(
                        ret, "labels", features[idx]["input_ids"], separator_id
                    )
                else:
                    raise ValueError(f"Invalid return_tensors value: {return_tensors}")

            # add position ids
            if self.return_position_ids:
                position_ids = list(
                    range(len(features[idx]["input_ids"]))
                )  # len compatible with all types
                if "position_ids" not in ret:
                    ret["position_ids"] = position_ids
                else:
                    ret["position_ids"] += position_ids

        # cast to desired return type
        return default_data_collator([ret], return_tensors)


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