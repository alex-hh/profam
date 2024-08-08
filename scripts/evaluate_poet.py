import argparse
import glob
import os
import string
from typing import Callable, Optional, Sequence

import numpy as np
import torch
from datasets import load_dataset
from hydra import compose, initialize_config_dir
from poet.alphabets import Uniprot21
from poet.models.poet import PoET
from poet.msa.sampling import MSASampler, NeighborsSampler

from src.constants import BASEDIR
from src.data.fasta import read_fasta_lines

ASCII_LOWERCASE_BYTES = string.ascii_lowercase.encode()

T = TypeVar("T", np.ndarray, torch.Tensor)


def get_encoded_msa_from_a3m_seqs(
    msa_sequences: list[bytes], alphabet: Uniprot21
) -> np.ndarray:
    return np.vstack(
        [
            alphabet.encode(s.translate(None, delete=ASCII_LOWERCASE_BYTES))
            for s in msa_sequences
        ]
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


def sample_msa_sequences(
    get_sequence_fn: Callable[[int], bytes],
    sample_idxs: Sequence[int],
    max_tokens: int,
    alphabet: Uniprot21,
    shuffle: bool = True,
    shuffle_seed: Optional[int] = None,
    truncate: bool = True,
) -> list[np.ndarray]:
    assert alphabet.start_token != -1
    assert alphabet.stop_token != -1
    if not shuffle:
        assert shuffle_seed is None

    seqs, total_tokens = [], 0
    for idx in sample_idxs:
        next_sequence = get_sequence_fn(idx)
        seqs.append(append_startstop(alphabet.encode(next_sequence), alphabet=alphabet))
        total_tokens += len(seqs[-1])
        if total_tokens > max_tokens:
            break

    # shuffle order and truncate to max tokens
    if shuffle:
        rng = (
            np.random.default_rng(shuffle_seed)
            if shuffle_seed is not None
            else np.random
        )
        final_permutation = rng.permutation(len(seqs))
    else:
        final_permutation = np.arange(len(seqs))
    final_seqs, total_tokens = [], 0
    for seq in [seqs[i] for i in final_permutation]:
        if truncate and (total_tokens + len(seq) > max_tokens):
            seq = seq[: max_tokens - total_tokens]
        total_tokens += len(seq)
        final_seqs.append(seq)
        if total_tokens >= max_tokens:
            break
    return final_seqs


def main(args):
    with initialize_config_dir(os.path.join(BASEDIR, "configs/data/dataset")):
        cfg = compose(config_name=args.dataset)  # for example
    if cfg.data_path_pattern is not None:
        # replace hf path resolution with manual glob, to allow repetition
        # https://github.com/huggingface/datasets/blob/98fdc9e78e6d057ca66e58a37f49d6618aab8130/src/datasets/data_files.py#L323
        data_files = glob.glob(os.path.join(args.data_dir, cfg.data_path_pattern))
    else:
        assert cfg.data_path_file is not None
        with open(os.path.join(args.data_dir, cfg.data_path_file), "r") as f:
            data_files = [
                os.path.join(args.data_dir, data_file)
                for data_file in f.read().splitlines()
            ]

    assert isinstance(data_files, list)
    data_files = data_files * cfg.file_repeats
    print(
        f"Loading {cfg.name} dataset from {len(data_files)} files, "
        f"({cfg.file_repeats} repeats), "
        f"{os.path.join(args.data_dir, cfg.data_path_pattern)}"
    )
    if cfg.is_parquet:
        dataset = load_dataset(
            path="parquet",
            data_files=data_files,
            split="train",
            streaming=True,
            ignore_verifications=True,
        )
    else:
        dataset = load_dataset(
            "text",
            data_files=data_files,
            split="train",
            streaming=True,
            sample_by="document",
        )

    all_metrics = []

    for i in range(len(dataset)):
        if "text" in dataset.column_names:
            fasta_text = dataset[i]["text"]
            sequences = [
                seq
                for _, seq in read_fasta_lines(
                    fasta_text.split("\n"),
                    keep_gaps=cfg.keep_gaps,
                    keep_insertions=cfg.keep_insertions,
                    to_upper=cfg.to_upper,
                )
            ]
        else:
            sequences = [
                s.upper().replace("-", "").replace(".", "")
                for s in dataset[i]["sequences"]
            ]
        sequences = [bytes(s) for s in sequences]

        ckpt = torch.load(args.ckpt_path)
        model = PoET(**ckpt["hyper_parameters"]["model_spec"]["init_args"])
        model.load_state_dict(
            {k.split(".", 1)[1]: v for k, v in ckpt["state_dict"].items()}
        )
        del ckpt
        model = model.cuda().half().eval()
        alphabet = Uniprot21(
            include_gap=True, include_startstop=True, distinct_startstop=True
        )

        # process msa
        msa = get_encoded_msa_from_a3m_seqs(msa_sequences=sequences, alphabet=alphabet)

        # TODO: we should actually sample using our own utils for fair comparison
        sampler = MSASampler(
            method=NeighborsSampler(
                can_use_torch=False,
            ),
            max_similarity=args.max_similarity,
        )
        sample_idxs = sampler.get_sample_idxs(
            msa=msa,
            gap_token=alphabet.gap_token,
            seed=args.seed,
        )
        # create the sequence-of-sequences
        this_msa_sequences = sample_msa_sequences(
            get_sequence_fn=lambda ii: msa_sequences[ii]
            .upper()
            .translate(None, delete=b"-"),
            sample_idxs=sample_idxs,
            max_tokens=args.max_tokens,
            alphabet=alphabet,
            shuffle_seed=args.seed,
            truncate=False,
        )

        with torch.no_grad():
            # same as model.embed
            # https://github.com/OpenProteinAI/PoET/blob/9b2239be84ee39691ec6ad4184925156f2ac332f/scripts/score.py#L175
            segment_sizes = torch.tensor([len(s) for s in this_msa_sequences]).cuda()
            xs: torch.Tensor = torch.cat(
                [torch.from_numpy(s).long() for s in this_msa_sequences]
            ).cuda()
            logits = model(xs.unsqueeze(0), segment_sizes.unsqueeze(0))
            # `targets = this_variants[:, 1:]
            # score = -criteria.forward(logits.transpose(1, 2), targets).float().sum(dim=1)
            # logps.append(score.cpu().numpy())`


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("--data_dir", default="../data")
    parser.add_argument("--max_tokens", type=int, default=8192)
    parser.add_argument("--max_similarity", type=float, default=0.9)
    args = parser.parse_args()
    main(args)
