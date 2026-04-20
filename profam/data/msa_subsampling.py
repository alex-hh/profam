import hashlib
import math
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal, Optional, Sequence, Union

import numba as nb
import numpy as np
import torch

"""
This module contains functions for computing homology weights and sampling MSA sequences.
code originally from:
https://github.com/OpenProteinAI/PoET/blob/main/poet/msa/sampling.py
"""


def hash_of_string_list(lst: list[str]) -> str:
    m = hashlib.sha1()
    for elt in lst:
        m.update(elt.encode("utf-8"))
    return m.hexdigest()


def compute_hamming_csim_np(
    seqs: np.ndarray,
    ungapped_msa: np.ndarray,
    gap_token: int = 20,
    gap_token_mask: int = 255,
) -> np.ndarray:
    """
    This function has an awkward spec. The point was to test
    compute_homology_weights_np and demonstrate its flaws wrt handling gap tokens.

    Compute the hamming similiarity between sequences in seqs and ungapped_msa, among
    non-gap tokens.

    Assumes

    - seqs[gap tokens in seqs] == gap_token
    - ungapped_msa[gap tokens in ungapped_msa] == gap_token_mask

    Example
    ---
    1. hamming_sim(
        ABC,  # a sequence in seqs
        ABA,  # a sequence in ungapped_msa
    ) = 2

    2. hamming_sim(
        A-C,  # a sequence in seqs
        ABA,  # a sequence in ungapped_msa
    ) = 1

    3. hamming_sim(
        AB-,  # a sequence in seqs
        ABA,  # a sequence in ungapped_msa
    ) = 2

    4. hamming_sim(
        AB-,  # a sequence in seqs
        AB-,  # a sequence in ungapped_msa
    ) = 2  # not 3 b/c of the matching gaps

    5. hamming_sim(
        ABC,  # a sequence in seqs
        AB-,  # a sequence in ungapped_msa
    ) = 2
    """
    return (seqs[:, np.newaxis] == ungapped_msa).sum(axis=2, dtype=np.uint16)


def compute_hamming_csim_torch(
    seqs: torch.Tensor,
    ungapped_msa: torch.Tensor,
    gap_token: int = 20,
    gap_token_mask: int = 255,
) -> torch.Tensor:
    return (seqs.unsqueeze(1) == ungapped_msa).sum(dim=2)


@nb.njit(locals={"sim": nb.uint16})
def hamming_sim(x, y, N):
    # Compute the Hamming sim between two sequences x and y
    sim = 0
    for i in range(N):
        if x[i] == y[i]:
            sim += 1
    return sim


@nb.njit(parallel=True)
def compute_hamming_csim_nb(
    seqs: np.ndarray,
    ungapped_msa: np.ndarray,
    gap_token: int = 20,
    gap_token_mask: int = 255,
) -> np.ndarray:
    """See compute_hamming_csim_np"""
    N1, M = seqs.shape
    N2, _ = ungapped_msa.shape
    sims = np.zeros((N1, N2), dtype=np.uint16)  # Initialize an array to store sims

    # Compute sims between all pairs of sequences
    for i in nb.prange(N1):
        for j in range(N2):
            sims[i, j] = hamming_sim(seqs[i], ungapped_msa[j], M)

    return sims


def _compute_homology_weights(
    ungapped_msa: np.ndarray,
    gap_token: int,
    gap_token_mask: int,
    theta: float,
    hamming_csim_func: Callable,
    max_memory: int = 20,
    can_use_torch: bool = True,
) -> np.ndarray:
    use_torch = can_use_torch and torch.cuda.is_available()
    if use_torch:
        hamming_csim_func = compute_hamming_csim_torch
    batch_size = math.floor(
        2
        * 1024
        * 1024
        * 1024
        / (ungapped_msa.shape[0] * ungapped_msa.shape[1])
        * max_memory
        / 40
    )

    batch_size = 1 if batch_size == 0 else batch_size

    neighbors = []
    if not use_torch:
        masked_ungapped_msa = ungapped_msa.copy()
    else:
        ungapped_msa = torch.from_numpy(ungapped_msa).byte().cuda()
        masked_ungapped_msa = ungapped_msa.clone()
    masked_ungapped_msa[masked_ungapped_msa == gap_token] = gap_token_mask
    for b_start in range(0, len(ungapped_msa), batch_size):
        b_end = b_start + batch_size
        seqs = ungapped_msa[b_start:b_end]

        sim = hamming_csim_func(
            seqs=seqs,
            ungapped_msa=masked_ungapped_msa,
            gap_token=gap_token,
            gap_token_mask=gap_token_mask,
        )
        if not use_torch:
            sim = sim / (seqs != gap_token).sum(axis=1, keepdims=True)
            d = 1 - sim
            assert ((d >= 0) & (d <= 1)).all()
            this_neighbors = (d <= theta).sum(axis=1)
        else:
            sim = sim / (seqs != gap_token).sum(dim=1, keepdim=True)
            d = 1 - sim
            assert ((d >= 0) & (d <= 1)).all()
            this_neighbors = (d <= theta).sum(dim=1).cpu()
        neighbors.append(this_neighbors)
    return np.concatenate(neighbors)


def compute_homology_weights(
    ungapped_msa: np.ndarray,
    theta: float = 0.2,
    gap_token: int = 20,
    gap_token_mask: int = 255,
    hamming_csim_func: Callable = compute_hamming_csim_nb,
    result_cache_dir: Optional[Path] = None,
    can_use_torch: bool = True,
) -> tuple[int, np.ndarray]:
    """
    Calculate the effective number of sequences and sampling probability for the NEIGHBORS and NEIGHBORS_NO_LIMIT sampling methods using numpy.

    Parameters:

        ungapped_msa (np.ndarray): The MSA (from .fa).
        theta (float, optional): A parameter used to determine the similarity between sequences. Default is 0.2.
        gap_token (int, optional): The token representing gaps in the encoded MSA. Default is 20.
        gap_token_mask (int): token for masking gaps. should be a token not representing any other value.

    Returns:

        tuple[int, np.ndarray]: A tuple containing the effective number of sequences and the sampling probability for each sequence in the MSA.
    """
    assert gap_token >= 0
    if result_cache_dir is not None:
        result_cache_dir = result_cache_dir / "compute_homology_weights"
        result_cache_dir.mkdir(exist_ok=True, parents=True)
        result_hash = hashlib.sha1(ungapped_msa.view(np.uint8)).hexdigest()
        additional_hash_components = []
        additional_hash_components.append(f"{theta=}")
        additional_hash_components.append(f"{gap_token=}")
        additional_hash_components.append(f"{gap_token_mask=}")
        result_hash = hash_of_string_list(additional_hash_components + [result_hash])
        result_filepath = (result_cache_dir / result_hash).with_suffix(".pkl")
        if result_filepath.is_file():
            return pickle.load(open(result_filepath, "rb"))

    neighbors = _compute_homology_weights(
        ungapped_msa=ungapped_msa,
        gap_token=gap_token,
        gap_token_mask=gap_token_mask,
        theta=theta,
        hamming_csim_func=hamming_csim_func,
        can_use_torch=can_use_torch,
    )
    n_eff = np.sum(1 / neighbors)

    p = 1 / neighbors
    p /= np.sum(p)

    if result_cache_dir is not None:
        pickle.dump((n_eff, p), open(result_filepath, "wb"))
    return n_eff, p


@dataclass
class NeighborsSampler:
    sampler_type: Literal["neighbors"] = "neighbors"
    theta: float = 0.2
    can_use_torch: bool = True

    def get_weights(
        self, msa: np.ndarray, gap_token: int, result_cache_dir: Optional[Path] = None
    ) -> tuple[Optional[float], Optional[np.ndarray]]:
        assert msa.dtype == np.uint8
        return compute_homology_weights(
            ungapped_msa=msa,
            theta=self.theta,
            gap_token=gap_token,
            gap_token_mask=255,
            result_cache_dir=result_cache_dir,
            can_use_torch=self.can_use_torch,
        )

    def get_sample_idxs(
        self,
        msa: np.ndarray,
        weights: Optional[np.ndarray] = None,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        assert weights is not None
        if len(msa) == 0:
            return np.array([], dtype=int)
        size = len(msa)
        rng = np.random.default_rng(seed) if seed is not None else np.random
        return rng.choice(len(msa), replace=False, size=size, p=weights / weights.sum())


@dataclass
class MSASampler:
    # TODO: refactor msa sampling code...
    method: Union[NeighborsSampler]
    force_include_first: bool = False
    max_similarity: float = 1.0
    max_dissimilarity: float = 1.0

    def _get_sim_filtered_idxs(self, msa: np.ndarray) -> np.ndarray:
        nonnormalized_sim = (msa == msa[[0]]).sum(axis=1)
        normfactor = msa.shape[1]
        norm_sim = nonnormalized_sim / normfactor

        assert (norm_sim.min() >= 0) and (norm_sim.max() <= 1)
        dsim = 1 - norm_sim

        max_sim_filter = norm_sim <= self.max_similarity
        max_dissim_filter = dsim <= self.max_dissimilarity
        return np.where(max_sim_filter & max_dissim_filter)[0]

    def get_sample_idxs(
        self,
        msa: np.ndarray,
        gap_token: int,
        seed: Optional[int] = None,
        result_cache_dir: Optional[Path] = None,
    ) -> np.ndarray:
        _, weights = self.method.get_weights(
            msa=msa, gap_token=gap_token, result_cache_dir=result_cache_dir
        )

        original_msa_sample_idxs = np.arange(len(msa))
        sample_idxs = self._get_sim_filtered_idxs(msa)
        original_msa_sample_idxs = original_msa_sample_idxs[sample_idxs]
        msa = msa[sample_idxs]
        weights = weights[sample_idxs]

        sample_idxs = self.method.get_sample_idxs(msa=msa, weights=weights, seed=seed)
        original_msa_sample_idxs = original_msa_sample_idxs[sample_idxs]
        del msa, weights

        if self.force_include_first:
            original_msa_sample_idxs = np.concatenate(
                [[0], original_msa_sample_idxs[original_msa_sample_idxs != 0]]
            )
        return original_msa_sample_idxs


def encode_msa_sequences_to_uint8(seqs: list[str]) -> np.ndarray:
    """Encode an aligned list of sequences to the uint8 format expected by
    `compute_homology_weights`.

    Any unknown or gap-like character (including '-') is mapped to the GAP token.
    """
    _GAP_TOKEN_IDX = 20
    _AA_TO_IDX = {aa: i for i, aa in enumerate("ACDEFGHIKLMNPQRSTVWY")}
    seq_len = len(seqs[0]) if seqs else 0
    arr = np.zeros((len(seqs), seq_len), dtype=np.uint8)
    for i, s in enumerate(seqs):
        arr[i] = [_AA_TO_IDX.get(ch, _GAP_TOKEN_IDX) for ch in s]  # unknowns → GAP
    return arr


def calculate_file_hash(filepath: str) -> str:
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:

        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def compute_homology_sequence_weights_with_cache(
    msa_file: str,
    sequences: list[str],
    theta: float = 0.2,
    force_recalc: bool = False,
) -> np.ndarray:
    """Return 1/neighbor-count weights for every sequence in *sequences*.

    If a cached file ``<msa_file_base>_weights.npz`` exists it is loaded instead of
    recomputing.  To override this behaviour pass *force_recalc*=True.
    The cache file includes a hash of the MSA file to ensure validity.
    """

    cache_path = os.path.splitext(msa_file)[0] + "_weights.npz"
    try:
        current_file_hash = calculate_file_hash(msa_file)
    except Exception as e:
        print(f"Warning: could not calculate hash for {msa_file}: {e}")
        current_file_hash = ""

    if (not force_recalc) and os.path.exists(cache_path):
        try:
            data = np.load(cache_path)
            cached_hash = str(data.get("file_hash", ""))
            # If the cached file has no hash (legacy), we might want to recompute or warn.
            # But to be safe and compatible with existing non-hashed cache files, we can check if it exists.
            # However, user requested to use hash. So if hash doesn't match or missing, we recompute.
            # If current_file_hash is empty (failed to read), we skip check? No, better recompute.

            if cached_hash == current_file_hash and current_file_hash != "":
                return data["sequence_weights"]
            elif "file_hash" not in data:
                print(f"Cached weights at {cache_path} missing hash. Recomputing...")
            elif cached_hash != current_file_hash:
                print(f"Cached weights hash mismatch for {cache_path}. Recomputing...")
        except Exception as e:
            print(
                f"Failed to load cached weights from {cache_path}: {e}. Recomputing …"
            )

    # Encode → compute → normalise
    encoded = encode_msa_sequences_to_uint8(sequences)

    # Note: compute_homology_weights is defined in this file
    _GAP_TOKEN_IDX = 20
    _, p = compute_homology_weights(
        ungapped_msa=encoded,
        theta=theta,
        gap_token=_GAP_TOKEN_IDX,
        gap_token_mask=255,
        can_use_torch=False,  # CPU is fine here
    )

    np.savez_compressed(cache_path, sequence_weights=p, file_hash=current_file_hash)
    return p
