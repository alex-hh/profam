"""Shared scoring utilities for ProFam models."""

from __future__ import annotations

import random
import sys
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from tqdm.auto import tqdm

from profam.models.llama import LlamaLitModule
from profam.models.utils import InputAwareDynamicCache, log_likelihood_from_outputs


def score_variants_ensemble(
    model: LlamaLitModule,
    completion_ids: torch.Tensor,
    tokenized_conditioning_sequences: List[List[int]],
    ensemble_size: int,
    scoring_max_tokens: int,
    start_tokens: Optional[list[int]] = None,
    max_tokens_override: Optional[int] = None,
    weights: Optional[np.ndarray] = None,
    seed: int = 42,
    return_per_residue: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, List[np.ndarray]]]:
    """Compute mean log-likelihoods using an ensemble of sampled prompts.

    ``seed`` controls the RNGs that pick ensemble prompt sub-samples.

    If ``return_per_residue`` is True, also returns a list of per-residue
    log-likelihood arrays (one per completion), averaged across the
    ensemble dimension. The returned ``mean_lls`` are then derived from
    the ensemble-averaged per-residue arrays (numerically equivalent to
    the non-per-residue mean because averaging commutes across the
    ensemble and residue dimensions when completion lengths are fixed).
    """
    if start_tokens is None:
        start_tokens = [47, 63]
    random.seed(seed)
    rng = random.Random(seed)
    rng_np = np.random.default_rng(seed)

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
    n_conditioning_choices = list(np.arange(lower_bound, upper_bound + 1, dtype=int))
    if len(n_conditioning_choices) == 0:
        n_conditioning_choices = [0]

    n_conditioning = int(rng.choice(n_conditioning_choices))
    n_conditioning_history: list[int] = []
    variant_lls: List[np.ndarray] = []
    # When per-residue: one list[np.ndarray] per ensemble member, each of
    # length n_candidates, with per-candidate per-position log-likelihoods.
    variant_per_residue: List[List[np.ndarray]] = []

    if completion_length + 2 > max_tokens:
        n_conditioning = 0
        repeats = 1
    else:
        repeats = min(ensemble_size, total_seqs) if total_seqs > 0 else 1

    sep_token_id = model.tokenizer.sep_token_id
    p = None
    if weights is not None:
        w = np.asarray(weights, dtype=np.float64)
        if np.any(w < 0):
            raise ValueError("weights must be non-negative")
        s = float(w.sum())
        p = (w / s) if s > 0 else None
    for _ in tqdm(
        range(repeats), desc="Scoring sequences", unit="prompt", file=sys.stderr
    ):
        while True:
            if n_conditioning == 0 and 0 in n_conditioning_history:
                if len(n_conditioning_choices) > 0:
                    n_conditioning = int(random.choice(n_conditioning_choices))
                else:
                    n_conditioning = 0
                    break

            if total_seqs > 0:
                idxs = rng_np.choice(
                    np.arange(total_seqs),
                    size=min(n_conditioning, total_seqs),
                    replace=False,
                    p=p,
                ).tolist()
                rng.shuffle(idxs)
                conditioning_token_count = sum(seq_lengths[i] for i in idxs)
            else:
                idxs = []
                conditioning_token_count = 0

            prompt_len_estimate = (
                len(start_tokens) + conditioning_token_count + len(idxs)
            )
            if prompt_len_estimate + completion_length <= max_tokens:
                break
            n_conditioning = max(0, n_conditioning - 1)

        if n_conditioning == 0 or len(idxs) == 0:
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

        use_kv_cache = getattr(model, "use_kv_cache_for_scoring", True)
        member_batch_size = (
            max(int(scoring_max_tokens) // (L + L_prompt), 1) if use_kv_cache else 1
        )
        if return_per_residue:
            per_residue = model.score_seqs(
                input_ids,
                completion_ids_device,
                use_cache=use_kv_cache,
                batch_size=member_batch_size,
                return_per_residue=True,
            )
            variant_per_residue.append(per_residue)
        else:
            lls = model.score_seqs(
                input_ids,
                completion_ids_device,
                use_cache=use_kv_cache,
                batch_size=member_batch_size,
            )
            variant_lls.append(lls)
        n_conditioning_history.append(n_conditioning)

        if len(n_conditioning_choices) > 0:
            n_conditioning = rng.choice(n_conditioning_choices)

    if return_per_residue:
        n_candidates = completion_ids.shape[1]
        per_residue_out: List[np.ndarray] = []
        mean_lls = np.zeros(n_candidates, dtype=np.float64)
        for i in range(n_candidates):
            stacked = np.stack(
                [variant_per_residue[e][i] for e in range(len(variant_per_residue))],
                axis=0,
            )  # (ensemble_size, L_i)
            ensemble_mean = stacked.mean(axis=0)  # (L_i,)
            per_residue_out.append(ensemble_mean)
            mean_lls[i] = float(ensemble_mean.mean()) if ensemble_mean.size > 0 else 0.0
        return mean_lls, per_residue_out

    lls_array = np.stack(variant_lls, axis=0)
    return lls_array.mean(axis=0)


def build_prompt_ids(
    tokenized_conditioning_sequences: List[List[int]],
    sep_token_id: int,
    start_tokens: Optional[List[int]] = None,
    max_context_tokens: int = 8192,
    n_sequences: Optional[int] = None,
    weights: Optional[np.ndarray] = None,
    rng: Optional[np.random.Generator] = None,
) -> List[int]:
    """Build a prompt from conditioning (MSA) sequences.

    Pattern: start_tokens + seq1 + [SEP] + seq2 + [SEP] + ... (no trailing [SEP]).

    Args:
        tokenized_conditioning_sequences: per-sequence token id lists.
        sep_token_id: the [SEP] token id between conditioning sequences.
        start_tokens: prefix tokens (defaults to ``[47, 63]`` =
            ``[start-of-document][RAW]`` for ProFam-1's vocab).
        max_context_tokens: cap on the prompt's total length.
        n_sequences: number of sequences to sample (defaults to all that fit).
        weights: optional sampling weights (e.g. homology-diversity weights).
        rng: numpy Generator for reproducible sampling.
    """
    if start_tokens is None:
        start_tokens = [47, 63]
    if rng is None:
        rng = np.random.default_rng(42)

    total_seqs = len(tokenized_conditioning_sequences)
    if total_seqs == 0:
        return list(start_tokens)

    if n_sequences is None:
        n_sequences = total_seqs
    n_sequences = min(n_sequences, total_seqs)

    p = None
    if weights is not None:
        w = np.asarray(weights, dtype=np.float64)
        w = np.clip(w, 0.0, None)
        s = float(w.sum())
        if s > 0:
            p = w / s

    idxs = rng.choice(
        np.arange(total_seqs), size=min(n_sequences, total_seqs), replace=False, p=p
    ).tolist()

    prompt_ids = list(start_tokens)
    for i, idx in enumerate(idxs):
        tokens_to_add = tokenized_conditioning_sequences[idx]
        # +1 for the [SEP] token between sequences (none after the last one)
        extra = len(tokens_to_add) + (1 if i < len(idxs) - 1 else 0)
        if len(prompt_ids) + extra > max_context_tokens:
            break
        prompt_ids.extend(tokens_to_add)
        if i < len(idxs) - 1:
            prompt_ids.append(sep_token_id)

    return prompt_ids


def cache_context(model: LlamaLitModule, prompt_ids: List[int], device) -> tuple:
    """Compute and cache KV states for an MSA conditioning context (no-grad).

    Returns the ``past_key_values`` tuple from a forward pass on the prompt.
    Intended to be called once with a frozen copy of the model and reused
    across all variant forwards during preference fine-tuning.
    """
    input_ids = torch.tensor(prompt_ids, dtype=torch.long, device=device).unsqueeze(0)
    with torch.no_grad():
        outputs = model.model(input_ids=input_ids, use_cache=True)
    return outputs.past_key_values


def _score_variants_kv(
    model: LlamaLitModule,
    past_key_values,
    completion_ids: torch.Tensor,
    sub_batch_size: int,
) -> torch.Tensor:
    """Shared inner loop for score_variants_{differentiable,no_grad}."""
    pad_token_id = model.tokenizer.pad_token_id
    N = completion_ids.shape[1]
    L = completion_ids.shape[2]
    all_scores = []

    for batch_start in range(0, N, sub_batch_size):
        batch_end = min(batch_start + sub_batch_size, N)
        this_ids = completion_ids[0, batch_start:batch_end]
        actual_bs = this_ids.shape[0]

        # Trim trailing padding for efficiency
        mask = this_ids != pad_token_id
        indices = torch.arange(L, device=this_ids.device).expand(actual_bs, -1)
        indices = torch.where(mask, indices, torch.zeros_like(indices))
        max_len = indices.max().item() + 1
        this_ids = this_ids[:, :max_len]

        cache = InputAwareDynamicCache.from_legacy_cache(past_key_values)
        cache.batch_repeat_interleave(actual_bs)

        outputs = model.model(
            input_ids=this_ids,
            past_key_values=cache,
            use_cache=False,
        )

        labels = torch.where(this_ids == pad_token_id, -100, this_ids.clone())
        log_ll = log_likelihood_from_outputs(outputs, labels, start_ix=0)

        shift_labels = labels[..., 1:].to(log_ll.device)
        valid = shift_labels != -100
        denom = valid.sum(dim=-1).clamp(min=1)
        ll_mean = (log_ll * valid).sum(dim=-1) / denom
        all_scores.append(ll_mean)

    return torch.cat(all_scores)


def score_variants_differentiable(
    model: LlamaLitModule,
    past_key_values,
    completion_ids: torch.Tensor,
    sub_batch_size: int = 4,
) -> torch.Tensor:
    """Score variants with gradient flow against a frozen KV-cache context.

    Args:
        model: ProFam Lightning module.
        past_key_values: detached/frozen KV cache for the conditioning prompt.
        completion_ids: (1, N, L) tokenised candidate sequences (with bos+eos sep).
        sub_batch_size: forward-pass micro-batch size.

    Returns:
        Tensor of shape (N,) holding mean per-token log-likelihoods, with grad.
    """
    return _score_variants_kv(model, past_key_values, completion_ids, sub_batch_size)


@torch.no_grad()
def score_variants_no_grad(
    model: LlamaLitModule,
    past_key_values,
    completion_ids: torch.Tensor,
    batch_size: int = 8,
) -> np.ndarray:
    """No-grad variant of :func:`score_variants_differentiable` for evaluation."""
    scores = _score_variants_kv(model, past_key_values, completion_ids, batch_size)
    return scores.cpu().float().numpy()
