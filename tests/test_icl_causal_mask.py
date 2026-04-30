"""Causal-mask invariance tests for the ICL forward pass.

These confirm that the prediction at each ``[VAL]`` position depends only on
tokens preceding it (the standard causal-LM property). Specifically:

1. Reordering / replacing tokens *after* a given ``[VAL]`` doesn't change the
   prediction at that ``[VAL]``.
2. Permuting the order of in-context examples generally changes downstream
   predictions, but the prediction at the *first* ``[VAL]`` (which has no
   labelled context preceding it) is identical no matter what.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from transformers import LlamaConfig

from profam.constants import VOCAB_SIZE
from profam.data.icl_constants import VAL_SLOT_TOKEN_ID, VAL_TOKEN_ID
from profam.models.llama_icl import LlamaICLLitModule

from tests.test_icl_forward import _build_simple_icl_batch


def _build_tiny_model(profam_tokenizer):
    config = LlamaConfig(
        vocab_size=VOCAB_SIZE,
        hidden_size=64,
        intermediate_size=128,
        num_attention_heads=4,
        num_hidden_layers=2,
        num_key_value_heads=4,
        max_position_embeddings=512,
        attn_implementation="eager",
        torch_dtype="float32",
    )
    return LlamaICLLitModule(
        config=config,
        tokenizer=profam_tokenizer,
        lr=1e-3,
        weight_decay=0.0,
        scheduler_name=None,
        ce_loss_weight=1.0,
        mse_loss_weight=1.0,
        backbone_lr_scale=1.0,
        pass_res_pos_in_doc_as_position_ids=True,
    )


def _forward_pred(model, batch):
    with torch.no_grad():
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            value_slot_mask=batch["value_slot_mask"],
            values=batch["values"],
            val_marker_mask=batch["val_marker_mask"],
            predict_mask=batch["predict_mask"],
            target_values=batch["target_values"],
            labels=None,
        )
    last = outputs.hidden_states[-1]
    preds = model.value_out_head(last.float()).squeeze(-1)
    return preds


def _val_marker_positions(batch):
    return torch.nonzero(batch["val_marker_mask"][0], as_tuple=False).flatten().tolist()


def test_changing_tokens_after_a_val_marker_does_not_change_its_prediction(
    profam_tokenizer,
):
    """Causal mask: pred at [VAL] position p depends only on tokens at <= p.
    We pick the *first* [VAL] (smallest p with a labelled context before it),
    swap out everything after it for a totally different sequence, and verify
    the prediction at that p is unchanged.
    """
    model = _build_tiny_model(profam_tokenizer).eval()
    batch_a = _build_simple_icl_batch(profam_tokenizer, k=4, seq_len=5, seed=0)
    pred_a = _forward_pred(model, batch_a)
    val_positions = _val_marker_positions(batch_a)
    first_val = val_positions[0]

    # Build a perturbation: keep input_ids[:, :first_val+1] identical, randomise
    # everything after. Aux tensors must stay shape-compatible.
    batch_b = {k: v.clone() for k, v in batch_a.items()}
    rng = np.random.default_rng(7)
    aa_ids = profam_tokenizer.aa_tokens
    L = batch_b["input_ids"].shape[1]
    for pos in range(first_val + 1, L):
        # Replace AA tokens with random AAs; leave special tokens alone so
        # that the auxiliary masks remain consistent with input_ids.
        if bool(batch_a["aa_mask"][0, pos].item()):
            batch_b["input_ids"][0, pos] = int(rng.choice(aa_ids))
        # Also scramble the labelled values and targets that come after.
        if bool(batch_a["value_slot_mask"][0, pos].item()):
            batch_b["values"][0, pos] = float(rng.normal())
        if bool(batch_a["val_marker_mask"][0, pos].item()):
            batch_b["target_values"][0, pos] = float(rng.normal())

    pred_b = _forward_pred(model, batch_b)

    assert torch.allclose(
        pred_a[0, first_val], pred_b[0, first_val], atol=1e-5
    ), (
        f"Causal-mask violation: pred at first [VAL] ({first_val}) changed when "
        f"only later tokens were perturbed. {pred_a[0, first_val]} vs {pred_b[0, first_val]}"
    )

    # And as a sanity-check the *last* [VAL] pred changed (we did perturb
    # things in its receptive field).
    last_val = val_positions[-1]
    assert not torch.allclose(
        pred_a[0, last_val], pred_b[0, last_val], atol=1e-5
    ), "Sanity check failed: query pred should change when context changes"


def test_permuting_context_examples_keeps_first_val_prediction_identical(
    profam_tokenizer,
):
    """The *first* [VAL] sees no labelled context before it, only the
    sequence ``x_1`` of the first labelled example. Permuting the *order* of
    the ``k`` labelled examples therefore leaves the first-[VAL] prediction
    untouched (it is conditioned only on tokens preceding it: BOS, DOC_TYPE,
    ``x_1``)."""
    model = _build_tiny_model(profam_tokenizer).eval()
    batch_a = _build_simple_icl_batch(profam_tokenizer, k=4, seq_len=5, seed=42)
    pred_a = _forward_pred(model, batch_a)
    first_val = _val_marker_positions(batch_a)[0]

    # Build batch_b by repeating batch_a (no permutation needed). The first
    # [VAL] prediction is determined entirely by tokens preceding it. Then we
    # explicitly permute *only* the downstream labelled examples - this would
    # be the natural "permute context" operation. Here we keep x_1 fixed and
    # swap two labelled examples that come after the first [VAL]; pred at
    # first [VAL] must not move.
    batch_b = {k: v.clone() for k, v in batch_a.items()}

    # Identify the spans of x_2 and x_3 (third labelled example). We can find
    # them by walking the val_slot_mask: each [VAL_SLOT] is preceded
    # (immediately) by [VAL] which is preceded by exactly seq_len AA tokens.
    val_slot_positions = torch.nonzero(
        batch_a["value_slot_mask"][0], as_tuple=False
    ).flatten().tolist()
    assert len(val_slot_positions) == 4
    # x_2 starts after the first [SEP] (one position past first [VAL_SLOT]).
    seq_len = 5
    x2_start = val_slot_positions[0] + 2  # +1 for [SEP], +0 for first AA
    x3_start = val_slot_positions[1] + 2
    x2_slice = slice(x2_start, x2_start + seq_len)
    x3_slice = slice(x3_start, x3_start + seq_len)
    # Swap x2 and x3 token ids (and their attached labelled values).
    x2_ids = batch_b["input_ids"][0, x2_slice].clone()
    x3_ids = batch_b["input_ids"][0, x3_slice].clone()
    batch_b["input_ids"][0, x2_slice] = x3_ids
    batch_b["input_ids"][0, x3_slice] = x2_ids

    # Swap the embedded labelled values and the regression targets at the
    # corresponding [VAL] / [VAL_SLOT] positions.
    val_positions = _val_marker_positions(batch_a)
    v2_pos, v3_pos = val_positions[1], val_positions[2]  # 2nd and 3rd [VAL]
    s2_pos, s3_pos = val_slot_positions[1], val_slot_positions[2]
    batch_b["values"][0, s2_pos], batch_b["values"][0, s3_pos] = (
        batch_a["values"][0, s3_pos].clone(),
        batch_a["values"][0, s2_pos].clone(),
    )
    batch_b["target_values"][0, v2_pos], batch_b["target_values"][0, v3_pos] = (
        batch_a["target_values"][0, v3_pos].clone(),
        batch_a["target_values"][0, v2_pos].clone(),
    )

    pred_b = _forward_pred(model, batch_b)
    assert torch.allclose(
        pred_a[0, first_val], pred_b[0, first_val], atol=1e-5
    ), (
        "Causal-mask violation: pred at first [VAL] changed when only "
        "later in-context examples were permuted."
    )
