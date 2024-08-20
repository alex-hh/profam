from typing import List, Optional, Sequence, TypeVar

import numpy as np
import torch
from poet.alphabets import Uniprot21
from poet.models.modules.packed_sequence import PackedTensorSequences
from poet.models.poet import PoET
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
from tqdm import trange


T = TypeVar("T", np.ndarray, torch.Tensor)


def _get_logps_tiered_fast(
    memory: Optional[list[PackedTensorSequences]],
    variants: Sequence[np.ndarray],
    model: PoET,
    batch_size: int,
    alphabet: Uniprot21,
    pbar_position: Optional[int] = None,
) -> np.ndarray:
    max_variant_length = max(len(v) for v in variants)
    memory = model.logits_allocate_memory(
        memory=memory,
        batch_size=batch_size,
        length=max_variant_length - 1,  # discount stop token
    )
    criteria = nn.CrossEntropyLoss(ignore_index=alphabet.mask_token, reduction="none")
    logps = []
    if pbar_position is not None:
        pbar = trange(
            0,
            len(variants),
            batch_size,
            desc=f"[{pbar_position}] decoding",
            leave=False,
            position=pbar_position,
        )
    else:
        pbar = range(0, len(variants), batch_size)
    for start_idx in pbar:
        this_variants = variants[start_idx : start_idx + batch_size]
        this_variants = pad_sequence(
            [torch.from_numpy(v).long() for v in this_variants],
            batch_first=True,
            padding_value=alphabet.mask_token,
        )
        if this_variants.size(1) < max_variant_length:
            this_variants = F.pad(
                this_variants,
                (0, max_variant_length - this_variants.size(1)),
                value=alphabet.mask_token,
            )
        assert (this_variants == alphabet.gap_token).sum() == 0
        this_variants = this_variants.cuda()
        logits = model.logits(this_variants[:, :-1], memory, preallocated_memory=True)
        targets = this_variants[:, 1:]
        score = -criteria.forward(logits.transpose(1, 2), targets).float().sum(dim=1)
        logps.append(score.cpu().numpy())
    return np.hstack(logps)


def get_logps_tiered_fast(
    msa_sequences: Sequence[np.ndarray],
    variants: Sequence[np.ndarray],
    model: PoET,
    batch_size: int,
    alphabet: Uniprot21,
    pbar_position: Optional[int] = None,
) -> np.ndarray:
    if len(msa_sequences) > 0:
        segment_sizes = torch.tensor([len(s) for s in msa_sequences]).cuda()
        msa_sequences: torch.Tensor = torch.cat(
            [torch.from_numpy(s).long() for s in msa_sequences]
        ).cuda()
        memory = model.embed(
            msa_sequences.unsqueeze(0),
            segment_sizes.unsqueeze(0),
            pbar_position=pbar_position,
        )
    else:
        memory = None

    return _get_logps_tiered_fast(
        memory=memory,
        variants=variants,
        model=model,
        batch_size=batch_size,
        alphabet=alphabet,
        pbar_position=pbar_position,
    )


def append_startstop(x: T, alphabet: Uniprot21) -> T:
    x_ndim = x.ndim
    assert x_ndim in {1, 2}
    if x_ndim == 1:
        x = x[None, :]

    if isinstance(x, torch.Tensor):
        empty_func = torch.empty
    else:
        empty_func = np.empty
    x_ = empty_func((x.shape[0], x.shape[1] + 2), dtype=x.dtype)
    x_[:, 0] = alphabet.start_token
    x_[:, -1] = alphabet.stop_token
    x_[:, 1:-1] = x
    if x_ndim == 1:
        x_ = x_.flatten()
    return x_


class PoETModel:
    """Wrapper for poet implementing methods that make it compatible with our evaluators."""

    def __init__(self, model: PoET):
        self.model = model
        self.alphabet = Uniprot21(
            include_gap=True,
            include_startstop=True,
            distinct_startstop=True,
        )

    @classmethod
    def from_pretrained(cls, path: str):
        raise NotImplementedError()

    def sample_seqs(
        self,
        sequence_prompt: List[str],
        num_samples: int,
        temperature: float = 1.0,
        maxlen: int = 1000,
        batch_size: int = 1,
    ):
        """c.f. https://github.com/OpenProteinAI/PoET/issues/1"""
        # TODO: support top_k, top_p sampling.
        all_samples = []
        with torch.no_grad():
            prompt_sequences = [
                append_startstop(self.alphabet.encode(s.encode()))
                for s in sequence_prompt
            ]
            segment_sizes = torch.tensor([len(s) for s in prompt_sequences]).cuda()
            xs: torch.Tensor = torch.cat(
                [torch.from_numpy(s).long() for s in prompt_sequences]
            ).cuda()
            num_batches = num_samples // batch_size
            for _ in range(num_batches):
                all_samples.append(
                    self.model.sample(
                        xs,
                        segment_sizes,
                        temperature=temperature,
                        maxlen=maxlen,
                        batch_size=batch_size,
                        alphabet=self.alphabet,
                    )
                )
        all_scores = torch.cat([s[1] for s in all_samples], dim=0)
        all_samples = torch.cat([s[0] for s in all_samples], dim=0)
        return all_samples

    def score_seqs(
        self,
        sequence_prompt: List[str],
        mutated_sequences: List[str],
        batch_size: int = 8,
    ):
        """Based on:
        https://github.com/OpenProteinAI/PoET/blob/main/scripts/score.py
        """
        # shuffle?
        with torch.no_grad():
            prompt_sequences = [
                append_startstop(self.alphabet.encode(s.encode()))
                for s in sequence_prompt
            ]
            total_tokens = sum(len(s) for s in prompt_sequences)
            print("Total tokens:", total_tokens)
            variants = [
                append_startstop(self.alphabet.encode(s.encode()))
                for s in mutated_sequences
            ]
            return get_logps_tiered_fast(
                msa_sequences=prompt_sequences,
                variants=variants,
                model=self.model,
                batch_size=batch_size,
                alphabet=self.alphabet,
            )
