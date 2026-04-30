"""Tests for ProteinGymICLDataset and ICLDocumentBatchCollator.

These build a tiny synthetic ProteinGym-shaped CSV on disk so they don't
depend on the real DMS files.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from profam.data.builders.proteingym_icl import (
    ProteinGymICLDataset,
    standardise_context,
)
from profam.data.collators import ICLDocumentBatchCollator
from profam.data.icl_constants import VAL_SLOT_TOKEN_ID, VAL_TOKEN_ID


def _write_synthetic_assays(
    gym_dir: Path, n_variants: int = 64, n_assays: int = 2
) -> None:
    """Write a tiny ProteinGym-shaped directory.

    Creates ``DMS_substitutions.csv`` plus per-assay CSVs under
    ``DMS_ProteinGym_substitutions/``. Each assay gets ``n_variants`` rows
    with a 30-residue WT and a few random substitutions per variant.
    """
    rng = np.random.default_rng(0)
    aas = list("ACDEFGHIKLMNPQRSTVWY")
    sub_dir = gym_dir / "DMS_ProteinGym_substitutions"
    msa_dir = gym_dir / "PoET_DMS_msa_files" / "DMS_substitutions"
    sub_dir.mkdir(parents=True, exist_ok=True)
    msa_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for a in range(n_assays):
        wt = "".join(rng.choice(aas, size=30))
        dms_id = f"SYN_ASSAY_{a:02d}"
        # Build variants by random single-AA substitutions.
        seqs = []
        scores = []
        for v in range(n_variants):
            seq = list(wt)
            pos = int(rng.integers(0, len(seq)))
            seq[pos] = rng.choice(aas)
            seqs.append("".join(seq))
            scores.append(float(rng.normal(loc=v * 0.01, scale=1.0)))
        df = pd.DataFrame(
            {
                "mutated_sequence": seqs,
                "DMS_score": scores,
            }
        )
        csv_name = f"{dms_id}.csv"
        df.to_csv(sub_dir / csv_name, index=False)
        # Write a tiny .a3m so build_gym_df's existence check passes.
        (msa_dir / f"{dms_id}.a3m").write_text(f">{dms_id}\n{wt}\n")
        rows.append(
            {
                "DMS_id": dms_id,
                "DMS_filename": csv_name,
                "MSA_filename": "ignored.a2m",
                "target_seq": wt,
                "seq_len": len(wt),
            }
        )
    pd.DataFrame(rows).to_csv(gym_dir / "DMS_substitutions.csv", index=False)


@pytest.fixture
def synthetic_gym_dir(tmp_path):
    gym_dir = tmp_path / "ProteinGym"
    _write_synthetic_assays(gym_dir, n_variants=64, n_assays=3)
    return gym_dir


def test_standardise_context_basic():
    labels = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    mu, sigma = standardise_context(labels)
    assert mu == pytest.approx(3.0)
    assert sigma > 0.0


def test_standardise_context_constant_falls_back_to_floor():
    labels = np.zeros(8, dtype=np.float32)
    _, sigma = standardise_context(labels, sigma_floor=1e-3)
    # Need positive sigma to avoid division-by-zero downstream.
    assert sigma >= 1e-3


def test_dataset_token_layout(synthetic_gym_dir, profam_tokenizer):
    ds = ProteinGymICLDataset(
        name="syn",
        dms_ids=["SYN_ASSAY_00", "SYN_ASSAY_01"],
        gym_data_dir=str(synthetic_gym_dir),
        max_tokens_per_example=4096,
        k_choices=(8,),
        seed=0,
        tokenizer=profam_tokenizer,
    )
    sample = ds[0]
    assert set(sample.keys()) >= {
        "input_ids",
        "attention_mask",
        "aa_mask",
        "value_slot_mask",
        "val_marker_mask",
        "predict_mask",
        "values",
        "target_values",
    }
    L = sample["input_ids"].shape[0]
    assert L >= 9 * 30 + 1  # 8 labelled + 1 query, 30 AAs each, plus markers
    # Special tokens at the document head.
    assert sample["input_ids"][0] == profam_tokenizer.bos_token_id
    # We expect exactly k labelled examples => k VAL_SLOT positions, k+1 VAL markers.
    n_val_slots = int((sample["input_ids"] == VAL_SLOT_TOKEN_ID).sum())
    n_val_markers = int((sample["input_ids"] == VAL_TOKEN_ID).sum())
    assert n_val_slots == 8, f"expected 8 [VAL_SLOT] tokens, got {n_val_slots}"
    assert n_val_markers == 9, f"expected 9 [VAL] tokens (k+1), got {n_val_markers}"
    # Predict-mask aligns with VAL markers (we predict at every [VAL]).
    assert (sample["predict_mask"] == sample["val_marker_mask"]).all()
    # Value-slot mask aligns 1:1 with [VAL_SLOT] tokens.
    assert (
        (sample["input_ids"] == VAL_SLOT_TOKEN_ID) == sample["value_slot_mask"]
    ).all()
    # Last [VAL] is the query: it should be the very last token.
    last_marker_pos = int(np.flatnonzero(sample["val_marker_mask"])[-1])
    assert last_marker_pos == L - 1, "Query [VAL] must be the final token"


def test_dataset_no_value_leakage(synthetic_gym_dir, profam_tokenizer):
    """The standardiser uses only the k labelled values; the query target lives
    in target_values, never in values."""
    ds = ProteinGymICLDataset(
        name="syn",
        dms_ids=["SYN_ASSAY_00"],
        gym_data_dir=str(synthetic_gym_dir),
        max_tokens_per_example=4096,
        k_choices=(8,),
        seed=0,
        tokenizer=profam_tokenizer,
    )
    sample = ds[0]
    # values[predict_mask] should all be 0 (we never embed the prediction target).
    pred_positions = np.flatnonzero(sample["predict_mask"])
    assert np.allclose(sample["values"][pred_positions], 0.0)

    # value_slot_mask and predict_mask should be disjoint.
    overlap = sample["value_slot_mask"] & sample["predict_mask"]
    assert not overlap.any()

    # The labelled values fed to the model should have mean ~0 / std ~1
    # (within-context z-score). We allow a small tolerance because the floor
    # only kicks in for tiny std.
    labelled_values = sample["values"][sample["value_slot_mask"]]
    assert labelled_values.size == 8
    assert labelled_values.mean() == pytest.approx(0.0, abs=1e-5)
    assert labelled_values.std() == pytest.approx(1.0, rel=0.05)


def test_collator_masks_val_tokens_in_labels(synthetic_gym_dir, profam_tokenizer):
    ds = ProteinGymICLDataset(
        name="syn",
        dms_ids=["SYN_ASSAY_00", "SYN_ASSAY_01"],
        gym_data_dir=str(synthetic_gym_dir),
        max_tokens_per_example=4096,
        k_choices=(8,),
        seed=1,
        tokenizer=profam_tokenizer,
    )
    collator = ICLDocumentBatchCollator(tokenizer=profam_tokenizer)
    batch = collator([ds[0], ds[1]])
    # All four ICL aux tensors must be present and aligned with input_ids.
    L = batch["input_ids"].shape[1]
    for key in (
        "value_slot_mask",
        "val_marker_mask",
        "predict_mask",
        "values",
        "target_values",
        "attention_mask",
        "labels",
        "aa_mask",
    ):
        assert key in batch, f"missing key {key}"
        assert batch[key].shape[1] == L, f"{key} shape mismatch with input_ids"

    # CE label ignore-index for [VAL] / [VAL_SLOT].
    val_positions = batch["input_ids"] == VAL_TOKEN_ID
    val_slot_positions = batch["input_ids"] == VAL_SLOT_TOKEN_ID
    assert (batch["labels"][val_positions] == -100).all()
    assert (batch["labels"][val_slot_positions] == -100).all()


def test_dataset_seed_reproducibility(synthetic_gym_dir, profam_tokenizer):
    ds = ProteinGymICLDataset(
        name="syn",
        dms_ids=["SYN_ASSAY_00", "SYN_ASSAY_01"],
        gym_data_dir=str(synthetic_gym_dir),
        max_tokens_per_example=4096,
        k_choices=(8,),
        seed=123,
        tokenizer=profam_tokenizer,
    )
    a = ds[7]
    b = ds[7]
    assert np.array_equal(a["input_ids"], b["input_ids"])
    assert np.allclose(a["values"], b["values"])
    assert np.allclose(a["target_values"], b["target_values"])
