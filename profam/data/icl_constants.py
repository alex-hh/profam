"""Token-id constants for the ICL fine-tune.

The shipped tokenizer reserves ``[SP1]``..``[SP10]`` (ids 52..61) as unused special tokens.
For the in-context-learning supervised fine-tune we repurpose the first two:

* ``[VAL]``      -> ``[SP1]`` (id 52): marker token whose hidden state is regressed to
  predict the continuous fitness value y.
* ``[VAL_SLOT]`` -> ``[SP2]`` (id 53): placeholder token whose token embedding is
  replaced at runtime by ``W_in @ phi(y)`` for the corresponding labelled example.

Document layout used by :class:`profam.data.builders.proteingym_icl.ProteinGymICLDataset`:

    [BOS] [DOC_TYPE] x_1 [VAL] [VAL_SLOT] [SEP] x_2 [VAL] [VAL_SLOT] [SEP] ... x_q [VAL]

The query ends with ``[VAL]`` and *no* ``[VAL_SLOT]`` so that the prediction at the query
``[VAL]`` cannot peek at any value that follows.
"""

VAL_TOKEN = "[SP1]"
VAL_SLOT_TOKEN = "[SP2]"

VAL_TOKEN_ID = 52
VAL_SLOT_TOKEN_ID = 53

ICL_AUX_TENSOR_KEYS = (
    "value_slot_mask",
    "values",
    "val_marker_mask",
    "predict_mask",
    "target_values",
)
