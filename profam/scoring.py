"""Shared scoring utilities for ProFam models."""

from __future__ import annotations

import random
import sys
from typing import List, Optional

import numpy as np
import torch
from tqdm.auto import tqdm

from profam.models.llama import LlamaLitModule


def score_variants_ensemble(
    model: LlamaLitModule,
    completion_ids: torch.Tensor,
    tokenized_conditioning_sequences: List[List[int]],
    ensemble_size: int,
    scoring_max_tokens: int,
    start_tokens: Optional[list[int]] = None,
    max_tokens_override: Optional[int] = None,
    weights: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Compute mean log-likelihoods using an ensemble of sampled prompts."""
    if start_tokens is None:
        start_tokens = [47, 63]
    random.seed(42)
    rng = random.Random(42)
    rng_np = np.random.default_rng(42)

    seq_lengths = [len(seq) for seq in tokenized_conditioning_sequences]
    total_seqs = len(seq_lengths)
    completion_length = completion_ids.shape[-1]

    max_tokens = (
        max_tokens_override if max_tokens_override is not None else model.max_tokens
    )
    max_context_tokens = (max_tokens - completion_length) - 5

    avg_seq_len = sum(seq_lengths) / len(seq_lengths) if len(seq_lengths) > 0 else 0
    min_seq_len = min(seq_lengths) if len(seq_lengths) > 0 else 0
    assumed_seq_len = (min_seq_len + avg_seq_len) / 2

    max_n_by_tokens = (
        max(0, min(int(max_context_tokens // assumed_seq_len) + 2, total_seqs))
        if avg_seq_len > 0
        else 0
    )

    lower_bound = min(max_n_by_tokens, 2)
    upper_bound = min(max_n_by_tokens, total_seqs)
    vals_in_range = list(np.arange(lower_bound, upper_bound + 1, dtype=int))
    if len(vals_in_range) == 0:
        vals_in_range = [0]

    n_opt = int(rng.choice(vals_in_range))
    n_seqs_list: list[int] = []
    variant_lls: List[np.ndarray] = []

    if completion_length + 2 > max_tokens:
        n_opt = 0
        repeats = 1
    else:
        repeats = min(ensemble_size, total_seqs) if total_seqs > 0 else 1

    sep_token_id = model.tokenizer.sep_token_id
    p = None
    if weights is not None:
        w = np.asarray(weights, dtype=np.float64)
        w = np.clip(w, 0.0, None)
        s = float(w.sum())
        p = (w / s) if s > 0 else None
    for _ in tqdm(
        range(repeats), desc="Scoring sequences", unit="prompt", file=sys.stderr
    ):
        while True:
            if n_opt == 0 and 0 in n_seqs_list:
                if len(vals_in_range) > 0:
                    n_opt = int(random.choice(vals_in_range))
                else:
                    n_opt = 0
                    break

            if total_seqs > 0:
                idxs = rng_np.choice(
                    np.arange(total_seqs),
                    size=min(n_opt, total_seqs),
                    replace=False,
                    p=p,
                ).tolist()
                rng.shuffle(idxs)
                tok_cnt = sum(seq_lengths[i] for i in idxs)
            else:
                idxs = []
                tok_cnt = 0

            prompt_len_estimate = len(start_tokens) + tok_cnt + len(idxs)
            if prompt_len_estimate + completion_length <= max_tokens:
                break
            n_opt = max(0, n_opt - 1)

        if n_opt == 0 or len(idxs) == 0:
            prompt_ids_list = []
        else:
            prompt_ids_list = list(start_tokens)
            for i, idx in enumerate(idxs):
                prompt_ids_list.extend(tokenized_conditioning_sequences[idx])
                if i < len(idxs) - 1:
                    prompt_ids_list.append(sep_token_id)

        if len(prompt_ids_list) > 0:
            input_ids = torch.tensor(
                prompt_ids_list, dtype=torch.long, device=model.device
            ).unsqueeze(0)
        else:
            input_ids = None

        L = completion_ids.shape[-1]
        L_prompt = 0 if input_ids is None else input_ids.shape[-1]
        completion_ids_device = completion_ids.to(model.device)

        lls = model.score_seqs(
            input_ids,
            completion_ids_device,
            use_cache=getattr(model, "use_kv_cache_for_scoring", True),
            batch_size=max(int(scoring_max_tokens) // (L + L_prompt), 1)
            if getattr(model, "use_kv_cache_for_scoring", True)
            else 1,
        )

        variant_lls.append(lls)
        n_seqs_list.append(n_opt)

        if len(vals_in_range) > 0:
            n_opt = rng.choice(vals_in_range)

    lls_array = np.stack(variant_lls, axis=0)
    return lls_array.mean(axis=0)
