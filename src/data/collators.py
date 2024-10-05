from typing import List, Optional

from transformers import DataCollatorForLanguageModeling

from src.data.objects import StringObject


class CustomDataCollator:
    """
    Wraps DataCollatorForLanguageModeling
    allows us to include elements which are not
    tensors with seq_len dimension, eg. dataset names
    """

    def __init__(
        self,
        tokenizer,
        mlm=False,
        ignore_gaps: bool = False,
        feature_names: Optional[List[str]] = None,
    ):
        self.tokenizer = tokenizer
        self.base_collator = DataCollatorForLanguageModeling(tokenizer, mlm=mlm)
        self.ignore_gaps = ignore_gaps
        self.feature_names = feature_names

    def __call__(self, examples):
        # TODO: maybe I have an issue with blending data with different keys?
        # need to handle either in collator or by standardising in tokenizer.
        def keep_feature(feature_name):
            return self.feature_names is None or feature_name in self.feature_names

        if self.feature_names is not None:
            assert all(
                [f in examples[0].keys() for f in self.feature_names]
            ), "All feature names must be present in the examples"

        non_string_data = [
            {k: v for k, v in e.items() if (not isinstance(v, str)) and keep_feature(k)}
            for e in examples
        ]
        string_data = [
            {k: v for k, v in e.items() if isinstance(v, str) and keep_feature(k)}
            for e in examples
        ]
        string_data_keys = set(k for obs in string_data for k in obs.keys())
        try:
            batch = self.base_collator(non_string_data)
        except Exception as e:
            print("Error in collator")
            print(string_data)
            # print(non_string_data)
            raise e
        if self.ignore_gaps:
            batch["labels"][
                batch["labels"] == self.tokenizer.convert_tokens_to_ids("-")
            ] = -100
        # dont predict mask tokens.
        batch["labels"][batch["labels"] == self.tokenizer.mask_token_id] = -100
        # n.b. padding tokens should already be -100 due to base collator.
        for str_key in string_data_keys:
            str_vals = [obs.get(str_key, "") for obs in string_data]
            str_obj = StringObject()
            str_obj.text = str_vals
            batch[str_key] = str_obj
        return batch
