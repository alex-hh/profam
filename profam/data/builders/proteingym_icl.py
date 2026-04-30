"""ICL fine-tune dataset over ProteinGym DMS substitutions.

Each ``__getitem__`` returns a single packed document of the form

    [BOS] [DOC_TYPE] x_1 [VAL] [VAL_SLOT] [SEP] ... x_k [VAL] [VAL_SLOT] [SEP] x_q [VAL]

together with a set of auxiliary tensors that the ICL model uses to
substitute the embedded value at each ``[VAL_SLOT]`` and to compute the
regression loss at each ``[VAL]`` position. See
``profam/data/icl_constants.py`` for the token ids.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from profam.data.icl_constants import VAL_SLOT_TOKEN_ID, VAL_TOKEN_ID
from profam.data.tokenizers import ProFamTokenizer


def build_icl_assay_index(
    dms_ids: Optional[List[str]],
    gym_data_dir: str,
    csv_filename: str = "DMS_substitutions.csv",
    max_completion_length: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Locate the per-assay DMS CSVs without requiring MSA files.

    The standard ``build_gym_df`` helper used by the zero-shot pipeline asserts
    MSA files exist; the ICL dataset does not need them in v1, so we keep the
    lookup minimal and skip MSA validation entirely.
    """
    df = pd.read_csv(os.path.join(gym_data_dir, csv_filename))
    if dms_ids is not None:
        df = df[df["DMS_id"].isin(dms_ids)].sort_values("DMS_id")
    if max_completion_length is not None and "seq_len" in df.columns:
        df = df[df["seq_len"] <= max_completion_length]
    if "indels" in csv_filename:
        dms_dir = "DMS_ProteinGym_indels"
    else:
        dms_dir = "DMS_ProteinGym_substitutions"
    df = df.copy()
    df["DMS_filename"] = df["DMS_filename"].apply(
        lambda x: os.path.join(gym_data_dir, dms_dir, x)
    )
    df["ds_name"] = "gym_icl"
    missing = [p for p in df["DMS_filename"] if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            f"DMS CSV files missing for {len(missing)} assays; first few: {missing[:3]}"
        )
    return df[["DMS_id", "DMS_filename", "ds_name"]].to_dict("records")


def load_split_assays(split_csv: str, split: str) -> List[str]:
    """Read ``DMS_id`` values for one side of a cluster split CSV.

    The CSV is expected to contain the columns ``DMS_id``, ``cluster_rep``, and
    ``split`` (with ``split in {"train", "test"}``). Missing files raise so the
    user is forced to build the split first.
    """
    df = pd.read_csv(split_csv)
    if not {"DMS_id", "split"}.issubset(df.columns):
        raise ValueError(
            f"{split_csv} must contain DMS_id and split columns; got {df.columns.tolist()}"
        )
    sub = df[df["split"] == split]
    if sub.empty:
        raise ValueError(f"No rows with split={split!r} in {split_csv}")
    return sub["DMS_id"].tolist()


def featurize_value_linear(values: np.ndarray) -> np.ndarray:
    """Identity featurisation; the actual lift to ``d`` happens in ``W_in``."""
    return values.reshape(-1, 1).astype(np.float32)


def standardise_context(
    labels: np.ndarray,
    sigma_floor: float = 1e-3,
) -> Tuple[float, float]:
    """Within-context z-score statistics computed from the labelled examples only.

    Returns ``(mu, sigma)``. The query value (which is the regression target) must
    be passed through the same affine transform afterwards but must NOT participate
    in computing ``mu`` or ``sigma``, otherwise label information leaks.
    """
    mu = float(np.mean(labels))
    sigma = float(np.std(labels))
    if sigma < sigma_floor:
        sigma = max(sigma_floor, sigma if labels.size > 1 else 1.0)
    if labels.size <= 1:
        sigma = 1.0
    return mu, sigma


class ProteinGymICLDataset(Dataset):
    """In-context-learning supervised regression on ProteinGym DMS assays.

    Per ``__getitem__``:

    1. Sample an assay ``a`` (each assay weighted equally; epoch length is
       ``samples_per_epoch``).
    2. Sample ``k+1`` variants without replacement from the assay's CSV.
    3. Compute ``mu, sigma`` from the labelled ``k`` values; standardise both
       the labelled values and the query target.
    4. Build the token stream and the four auxiliary tensors needed by
       :class:`profam.models.llama_icl.LlamaICLLitModule`.

    The returned dict has keys: ``input_ids`` (int64 numpy array),
    ``attention_mask`` (int64 numpy array), ``aa_mask`` (bool numpy array),
    ``value_slot_mask`` (bool), ``val_marker_mask`` (bool), ``predict_mask``
    (bool), ``values`` (float32), ``target_values`` (float32), plus the string
    fields ``ds_name`` and ``DMS_id``.
    """

    def __init__(
        self,
        name: str,
        split: str = "train",
        split_csv: Optional[str] = None,
        dms_ids: Optional[List[str]] = None,
        gym_data_dir: Optional[str] = None,
        csv_filename: str = "DMS_substitutions.csv",
        max_tokens_per_example: int = 8192,
        k_choices: Tuple[int, ...] = (32, 64, 128, 256),
        seed: int = 42,
        samples_per_epoch: Optional[int] = None,
        document_token: str = "[RAW]",
        sigma_floor: float = 1e-3,
        max_completion_length: Optional[int] = None,
        tokenizer: Optional[ProFamTokenizer] = None,
    ):
        self.name = name
        self.split = split
        self.split_csv = split_csv
        self.gym_data_dir = gym_data_dir
        self.csv_filename = csv_filename
        self.max_tokens_per_example = max_tokens_per_example
        self.k_choices = tuple(int(k) for k in k_choices)
        if min(self.k_choices) < 1:
            raise ValueError("k_choices must all be >= 1")
        self.seed = int(seed)
        self.document_token = document_token
        self.sigma_floor = float(sigma_floor)
        self._tokenizer = tokenizer

        if dms_ids is None:
            if split_csv is None:
                raise ValueError("Either dms_ids or split_csv must be provided")
            dms_ids = load_split_assays(split_csv, split)
        self.dms_ids = list(dms_ids)
        if len(self.dms_ids) == 0:
            raise ValueError(f"Empty dms_ids list for split={split!r}")

        # Resolve filenames via the existing ProteinGym helper. We deliberately
        # ignore the MSA path here because v1 does not use a homolog prefix.
        effective_gym_dir = (
            self.gym_data_dir
            if self.gym_data_dir is not None
            else os.path.join("../data", "ProteinGym")
        )
        self._rows = build_icl_assay_index(
            dms_ids=self.dms_ids,
            gym_data_dir=effective_gym_dir,
            csv_filename=self.csv_filename,
            max_completion_length=max_completion_length,
        )
        self._dms_id_to_row = {r["DMS_id"]: r for r in self._rows}
        self.samples_per_epoch = (
            int(samples_per_epoch) if samples_per_epoch is not None else len(self._rows)
        )

        # Per-assay in-memory cache for the (mutated_sequence, score) tables.
        self._assay_tables: Dict[str, pd.DataFrame] = {}

    # -- helpers -----------------------------------------------------------

    def __len__(self) -> int:
        return self.samples_per_epoch

    @property
    def tokenizer(self) -> ProFamTokenizer:
        if self._tokenizer is None:
            raise RuntimeError("ProteinGymICLDataset has no tokenizer attached")
        return self._tokenizer

    def _load_assay_table(self, dms_id: str) -> pd.DataFrame:
        if dms_id not in self._assay_tables:
            row = self._dms_id_to_row[dms_id]
            df = pd.read_csv(row["DMS_filename"])
            if "mutated_sequence" not in df.columns or "DMS_score" not in df.columns:
                raise ValueError(
                    f"Assay {dms_id} CSV missing mutated_sequence or DMS_score columns"
                )
            df = df.dropna(subset=["mutated_sequence", "DMS_score"]).reset_index(
                drop=True
            )
            self._assay_tables[dms_id] = df
        return self._assay_tables[dms_id]

    def _tokenize_seq(self, seq: str) -> np.ndarray:
        """Encode a single AA string to token ids (no special tokens)."""
        out = self.tokenizer(
            seq,
            add_special_tokens=False,
            return_tensors="np",
            return_token_type_ids=False,
        )
        ids = out["input_ids"][0].astype(np.int64)
        if (ids == self.tokenizer.unk_token_id).any():
            raise ValueError(
                f"UNK tokens encountered while tokenising sequence of length {len(seq)}"
            )
        return ids

    def _select_k(self, table_size: int, rng: np.random.Generator) -> int:
        eligible = [k for k in self.k_choices if k + 1 <= table_size]
        if not eligible:
            # Fall back: use as much labelled context as we can while leaving the query.
            return max(table_size - 1, 1)
        return int(rng.choice(eligible))

    def _build_document(
        self,
        seq_token_streams: List[np.ndarray],
        labelled_values: np.ndarray,
        query_target_value: float,
    ) -> Dict[str, np.ndarray]:
        """Splice the per-variant token streams into a single ICL document.

        ``seq_token_streams`` is length ``k+1`` and the *last* entry is the query.
        ``labelled_values`` are the standardised values for the first ``k`` examples.
        ``query_target_value`` is the standardised regression target for the query.
        """
        tok = self.tokenizer
        bos_id = int(tok.bos_token_id)
        sep_id = int(tok.sep_token_id)
        doc_id = int(tok.convert_tokens_to_ids(self.document_token))

        ids: List[int] = [bos_id, doc_id]
        value_slot_mask: List[bool] = [False, False]
        val_marker_mask: List[bool] = [False, False]
        predict_mask: List[bool] = [False, False]
        values: List[float] = [0.0, 0.0]
        target_values: List[float] = [0.0, 0.0]
        aa_mask: List[bool] = [False, False]

        k = len(seq_token_streams) - 1
        for i, seq_ids in enumerate(seq_token_streams):
            for tid in seq_ids.tolist():
                ids.append(int(tid))
                value_slot_mask.append(False)
                val_marker_mask.append(False)
                predict_mask.append(False)
                values.append(0.0)
                target_values.append(0.0)
                aa_mask.append(True)

            # Append [VAL] marker for every example, including the query.
            ids.append(VAL_TOKEN_ID)
            value_slot_mask.append(False)
            val_marker_mask.append(True)
            predict_mask.append(True)
            values.append(0.0)
            if i < k:
                target_values.append(float(labelled_values[i]))
            else:
                target_values.append(float(query_target_value))
            aa_mask.append(False)

            if i < k:
                # Labelled example: append [VAL_SLOT] (its embedding is overridden
                # at runtime) then [SEP].
                ids.append(VAL_SLOT_TOKEN_ID)
                value_slot_mask.append(True)
                val_marker_mask.append(False)
                predict_mask.append(False)
                values.append(float(labelled_values[i]))
                target_values.append(0.0)
                aa_mask.append(False)

                ids.append(sep_id)
                value_slot_mask.append(False)
                val_marker_mask.append(False)
                predict_mask.append(False)
                values.append(0.0)
                target_values.append(0.0)
                aa_mask.append(False)
            # Query: nothing follows the trailing [VAL] - causal mask makes the
            # prediction at that position depend only on what came before.

        L = len(ids)
        return {
            "input_ids": np.asarray(ids, dtype=np.int64),
            "attention_mask": np.ones(L, dtype=np.int64),
            "aa_mask": np.asarray(aa_mask, dtype=bool),
            "value_slot_mask": np.asarray(value_slot_mask, dtype=bool),
            "val_marker_mask": np.asarray(val_marker_mask, dtype=bool),
            "predict_mask": np.asarray(predict_mask, dtype=bool),
            "values": np.asarray(values, dtype=np.float32),
            "target_values": np.asarray(target_values, dtype=np.float32),
        }

    # -- main API ----------------------------------------------------------

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rng = np.random.default_rng(self.seed + int(idx))
        # Round-robin over assays so every assay gets seen, but jitter the
        # starting offset by idx to decorrelate workers.
        assay_pos = int(rng.integers(0, len(self._rows)))
        dms_id = self._rows[assay_pos]["DMS_id"]
        table = self._load_assay_table(dms_id)
        n = len(table)
        if n < 2:
            raise RuntimeError(
                f"Assay {dms_id} has only {n} rows; need at least 2 for k>=1 + query"
            )

        k = self._select_k(n, rng)
        # Now budget for tokens: every variant contributes its own length plus
        # 3 tokens (VAL + VAL_SLOT + SEP) for labelled examples, +1 (VAL) for the
        # query, and 2 tokens (BOS + DOC_TYPE) of fixed overhead.
        chosen_indices = rng.choice(n, size=k + 1, replace=False)
        seqs = [table.iloc[i]["mutated_sequence"] for i in chosen_indices]
        labels = np.asarray(
            [float(table.iloc[i]["DMS_score"]) for i in chosen_indices],
            dtype=np.float32,
        )
        seq_token_streams = [self._tokenize_seq(s) for s in seqs]

        # Trim to fit the token budget if needed; always keep the query.
        budget = self.max_tokens_per_example - 2  # BOS + DOC_TYPE
        # cost per example: len(seq_ids) + 3 (labelled) or +1 (query)
        keep_indices: List[int] = []  # indices into seqs
        # Always keep the query (last element) first.
        query_cost = len(seq_token_streams[-1]) + 1
        if query_cost > budget:
            raise RuntimeError(
                f"Assay {dms_id}: query alone needs {query_cost} > {budget} tokens"
            )
        used = query_cost
        for i in range(len(seq_token_streams) - 1):
            cost = len(seq_token_streams[i]) + 3
            if used + cost > budget:
                continue
            keep_indices.append(i)
            used += cost
        if not keep_indices:
            # Cannot fit any labelled context - that shouldn't happen at v1
            # ``k`` schedule for normal-sized targets, but we still return a
            # valid document with k=0 (only the query). Causal attention then
            # has nothing to condition on, which is fine for testing but
            # bypasses the supervised signal. Surface the situation loudly.
            raise RuntimeError(
                f"Assay {dms_id}: no labelled context fits in {budget} tokens "
                f"with query of length {len(seq_token_streams[-1])}"
            )

        kept_seqs = [seq_token_streams[i] for i in keep_indices] + [
            seq_token_streams[-1]
        ]
        kept_labels = labels[keep_indices]
        query_label = float(labels[-1])

        mu, sigma = standardise_context(kept_labels, sigma_floor=self.sigma_floor)
        labelled_values = (kept_labels - mu) / sigma
        query_target = (query_label - mu) / sigma

        doc = self._build_document(
            seq_token_streams=kept_seqs,
            labelled_values=labelled_values,
            query_target_value=query_target,
        )
        doc["ds_name"] = self.name
        doc["DMS_id"] = dms_id
        return doc
