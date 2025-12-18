import functools
import os
import re
import warnings
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerFast

from src.data.msa_subsampling import (
    compute_homology_sequence_weights_with_cache,
    compute_homology_weights,
)
from src.data.objects import ProteinDocument
from src.data.processors import transforms
from src.data.processors.transforms import (
    preprocess_aligned_sequences_sampling_to_max_tokens,
)
from src.data.tokenizers import ProFamTokenizer
from src.sequence import fasta


def tokenize_msa(
    sample,
    tokenizer: ProFamTokenizer,
    document_token: Optional[str] = "[RAW]",
):
    # todo replace with subsample_and_tokenize_protein_data
    # gym msas don't contain insertions so no need to worry about that and default position indexing is fine
    proteins = ProteinDocument(
        sequences=sample["MSA"],
    )
    tokenized = tokenizer.encode(
        proteins, document_token=document_token, add_final_sep=False
    )  # sep gets added in completion bos
    sample["input_ids"] = tokenized.input_ids.squeeze()

    return sample


def get_token_from_name(name: str, tokenizer: PreTrainedTokenizerFast):
    if name == "bos":
        return tokenizer.bos_token
    elif name == "sep":
        return tokenizer.sep_token
    elif name in tokenizer.vocab:
        return name
    else:
        raise ValueError(f"Token {name} not found in tokenizer vocabulary")


def tokenize_completions(
    sample,
    tokenizer: ProFamTokenizer,
    bos_token="sep",
):
    tokenized = tokenizer.encode_completions(
        sequences=sample["completion_seqs"],
        bos_token=get_token_from_name(bos_token, tokenizer),
    )
    sample["completion_ids"] = tokenized.input_ids

    return sample


def tokenize(
    sample,
    tokenizer: PreTrainedTokenizerFast,
    mutant_bos_token="sep",
    document_token="[RAW]",
):
    has_context = "MSA" in sample and sample["MSA"] is not None
    if not has_context:
        sample["MSA"] = [""]
        sample["seq_pos"] = []
        msa_document_token = (
            ""  # document token will be added to start of completions instead
        )
        assert (
            mutant_bos_token == document_token
        )  # completions must start with non AA token
    else:
        msa_document_token = document_token

    sample = tokenize_msa(
        sample,
        tokenizer,
        document_token=msa_document_token,
    )

    sample = tokenize_completions(
        sample,
        tokenizer,
        bos_token=mutant_bos_token,
    )
    return sample


def load_msa_for_row(
    row,
    seed,
    tokenizer,
    max_tokens,
    max_context_seqs: Optional[int] = None,
    keep_wt=False,
    drop_wt=True,
    keep_gaps=False,
    use_filtered_msa: bool = False,
    extra_tokens_per_document: int = 2,
    use_msa_pos: bool = True,
    use_msa_seq_weights: bool = False,
):
    msa_file = row["MSA_filename"]
    if not os.path.exists(msa_file):
        msa_file = msa_file.replace(".a2m", ".a3m")
        if not os.path.exists(msa_file):
            raise FileNotFoundError(f"MSA file {msa_file} not found")
    if use_filtered_msa:
        msa_file = msa_file.replace(".a2m", "_reformat_hhfilter.a3m")
    print(f"Loading MSA from {msa_file}")
    seq_ids, seqs = fasta.read_fasta(  # initially load without changes for pos calc
        msa_file,
        keep_insertions=True,
        to_upper=True,
        keep_gaps=True if use_msa_pos else keep_gaps,
    )
    # ------------------------------------------------------------------
    # Sequence weights
    # ------------------------------------------------------------------
    if use_msa_seq_weights:
        (
            _,
            seqs_for_weights,
        ) = fasta.read_fasta(  # initially load without changes for pos calc
            msa_file, keep_insertions=False, to_upper=True, keep_gaps=True
        )
        # Homology-based weights with on-disk caching
        sequence_weights = compute_homology_sequence_weights_with_cache(
            msa_file=msa_file,
            sequences=seqs_for_weights,
        ).tolist()
    else:
        sequence_weights = [1.0 for _ in seqs]

    # Load coverage and similarity data if available
    sequence_similarities = None
    coverages = None
    npz_file = os.path.splitext(msa_file)[0] + ".npz"
    # print(f"Attempting to load coverage and similarity data from {npz_file}")
    if os.path.exists(npz_file):
        npz_data = np.load(npz_file)
        # Replace any NaN values with 0 before converting to list
        sequence_similarities = np.nan_to_num(
            npz_data["sequence_similarities"], nan=0.0
        ).tolist()
        coverages = np.nan_to_num(npz_data["coverages"], nan=0.0).tolist()
        # print(
        #     f"mean sequence similarity to wt: {np.mean(sequence_similarities)}, num_seqs: {len(sequence_similarities)}"
        # )
        if len(sequence_similarities) != len(seqs):
            print(
                f"Warning: Number of sequences in MSA ({len(seqs)}) doesn't match number in .npz file ({len(sequence_similarities)})"
            )
            sequence_similarities = None
            coverages = None
    seq_indices = [
        i
        for i, s in enumerate(seqs)
        if "X" not in s
        and "U" not in s
        and "Z" not in s
        and "O" not in s
        and "B" not in s
        and "J" not in s
    ]
    seqs = [seqs[i] for i in seq_indices]
    if sequence_similarities is not None:
        sequence_similarities = [sequence_similarities[i] for i in seq_indices]
    if coverages is not None:
        coverages = [coverages[i] for i in seq_indices]
    if sequence_weights is not None:
        sequence_weights = [sequence_weights[i] for i in seq_indices]
    proteins = ProteinDocument(
        sequences=seqs,
        accessions=None,
        identifier=row["DMS_id"],
        sequence_similarities=sequence_similarities,
        coverages=coverages,
        sequence_weights=sequence_weights,
    )
    # need to allow room for the completion
    # todo should be max completion length (once we handle indels)
    max_tokens_for_msa = max_tokens - max([len(s) for s in seqs]) - 2
    proteins = preprocess_aligned_sequences_sampling_to_max_tokens(
        proteins,
        tokenizer=tokenizer,
        seed=seed,
        drop_first=drop_wt and len(proteins) > 1,
        keep_first=keep_wt,
        max_tokens=max_tokens_for_msa,
        extra_tokens_per_document=extra_tokens_per_document,
        sequence_converter=functools.partial(
            transforms.convert_aligned_sequence_adding_positions,
            use_msa_pos=use_msa_pos,
            to_upper=True,
            keep_insertions=True,
            keep_gaps=keep_gaps,
        ),
    )
    if max_context_seqs is not None:
        proteins = proteins[:max_context_seqs]

    assert len(proteins.sequences) > 0, "No sequences sampled - check max tokens"
    row["MSA"] = proteins.sequences
    # Ensure coverage and similarity data are always present for consistent schema
    if proteins.sequence_similarities is None:
        proteins.sequence_similarities = [0.0 for _ in proteins.sequences]
    if proteins.coverages is None:
        proteins.coverages = [0.0 for _ in proteins.sequences]
    row["sequence_similarities"] = proteins.sequence_similarities
    row["coverages"] = proteins.coverages
    if use_msa_seq_weights:
        row["sequence_weights"] = proteins.sequence_weights
    return row


def load_comp_seq_dms_for_row(
    row,
    seed,
    tokenizer,
    max_mutated_sequences,
    use_msa_pos: bool = True,
    keep_gaps: bool = False,
):

    dms_df = pd.read_csv(row["DMS_filename"])
    if max_mutated_sequences is not None and max_mutated_sequences < len(dms_df):
        dms_df = dms_df.sample(n=max_mutated_sequences, random_state=seed)
    completion_seqs = dms_df["mutated_sequence"].tolist()
    proteins = ProteinDocument(
        sequences=completion_seqs,
        accessions=None,
        identifier=None,
    )
    proteins = transforms.preprocess_aligned_sequences_sampling_to_max_tokens(
        proteins,
        tokenizer,
        sequence_converter=functools.partial(
            transforms.convert_aligned_sequence_adding_positions,
            keep_gaps=keep_gaps,  # no gaps in DMS sequences
            keep_insertions=True,  # no insertions in DMS sequences
            to_upper=True,
            use_msa_pos=use_msa_pos,
        ),
        max_tokens=None,
        shuffle=False,
    )
    row["DMS_scores"] = dms_df["DMS_score"].tolist()
    row["completion_seqs"] = proteins.sequences
    return row


def build_gym_df(
    dms_ids,
    gym_data_dir: str,
    use_foldseek_msa: bool = False,
    max_completion_length: Optional[bool] = None,
    msa_folder_name: str = "DMS_msa_files",
    task_index: Optional[int] = None,
    num_tasks: Optional[int] = None,
    csv_filename: str = "DMS_substitutions.csv",
):
    """We pre-load and pre-sample MSAs, ensuring they are same at each validation step."""
    df = pd.read_csv(os.path.join(gym_data_dir, csv_filename))

    if dms_ids is not None:
        df = df[df["DMS_id"].isin(dms_ids)].sort_values("DMS_id")
    else:
        print("dms_ids is None so evaluating on all ProteinGym assays")

    if max_completion_length is not None:
        df = df[df["seq_len"] <= max_completion_length]

    if task_index is not None and num_tasks is not None:
        batch_size = len(df) // num_tasks
        start_idx = task_index * batch_size
        if task_index == num_tasks - 1:
            end_idx = len(df)
        else:
            end_idx = start_idx + batch_size
        df = df.iloc[start_idx:end_idx]

    if use_foldseek_msa:
        df["MSA_filename"] = df["MSA_filename"].apply(
            lambda x: os.path.join(gym_data_dir, "foldseek_s50_DMS_msa_files", x)
        )
    elif "PoET" in msa_folder_name:
        df["MSA_filename"] = df["DMS_id"].apply(
            lambda x: os.path.join(gym_data_dir, msa_folder_name, x + ".a3m")
        )
    elif "filtered_msas_poet" in msa_folder_name:
        df["MSA_filename"] = df["DMS_id"].apply(
            lambda x: os.path.join(gym_data_dir, msa_folder_name, x + "_filtered.fasta")
        )
    elif "msa_pairformer" in msa_folder_name:
        df["MSA_filename"] = df["MSA_filename"].apply(
            lambda x: os.path.join(
                gym_data_dir, msa_folder_name, x.split(".")[0] + "_ranked.fasta"
            )
        )
    else:
        df["MSA_filename"] = df["MSA_filename"].apply(
            lambda x: os.path.join(gym_data_dir, msa_folder_name, x)
        )

    if "indels" in csv_filename:
        dms_dir = "DMS_ProteinGym_indels"
        df = df[~df.MSA_filename.str.contains("PSAE_PICP2")]
    else:
        dms_dir = "DMS_ProteinGym_substitutions"
    assert all(
        os.path.exists(msa_file) for msa_file in df["MSA_filename"]
    ), "MSA files do not exist"

    df["DMS_filename"] = df["DMS_filename"].apply(
        lambda x: os.path.join(gym_data_dir, dms_dir, x)
    )
    df["ds_name"] = "gym"
    return df[
        [
            "DMS_id",
            "MSA_filename",
            "DMS_filename",
            "ds_name",
        ]
    ]


class ProteinGymDataset(Dataset):
    def __init__(
        self,
        name: str,
        dms_ids: List[str],
        seed: Optional[int] = 42,  # for msa sampling
        max_mutated_sequences: Optional[int] = None,
        mutant_bos_token: str = "sep",
        keep_gaps: bool = False,
        use_filtered_msa: bool = True,
        extra_tokens_per_document: int = 2,
        use_msa_pos: bool = True,
        num_proc: Optional[int] = None,
        gym_data_dir: Optional[str] = None,
        max_tokens_per_example: Optional[int] = None,
        use_foldseek_msa: bool = False,
        max_context_seqs: Optional[
            int
        ] = None,  # 0 means no family context, None means use all
        max_completion_length=None,
        keep_wt: bool = False,
        drop_wt: bool = True,
        msa_folder_name: str = "DMS_msa_files",
        use_msa_seq_weights: bool = False,
        task_index: Optional[int] = None,
        num_tasks: Optional[int] = None,
        csv_filename: str = "DMS_substitutions.csv",
        tokenizer: Optional[ProFamTokenizer] = None,
    ):
        """Thing that's a bit different about Gym (and family classification)
        is that we have this prompt/completions structure.

        We can still use a preprocessor to build the prompt, but we need
        to additionally handle preprocessing of completions.

        We can still train on these datasets - just by setting seed None and
        not setting val dataset name. In this case, model will ignore completions.
        """
        self.name = name
        self.dms_ids = dms_ids
        self.seed = seed
        self.max_mutated_sequences = max_mutated_sequences
        self.mutant_bos_token = mutant_bos_token
        self.keep_gaps = keep_gaps
        self.use_filtered_msa = use_filtered_msa
        self.extra_tokens_per_document = extra_tokens_per_document
        self.use_msa_pos = use_msa_pos
        self.num_proc = num_proc
        self.gym_data_dir = gym_data_dir
        self.max_tokens_per_example = max_tokens_per_example
        self.max_context_seqs = max_context_seqs
        self.max_completion_length = max_completion_length
        self.keep_wt = keep_wt
        self.drop_wt = drop_wt
        self.use_foldseek_msa = use_foldseek_msa
        self.msa_folder_name = msa_folder_name
        self.use_msa_seq_weights = use_msa_seq_weights
        self.task_index = task_index
        self.num_tasks = num_tasks
        self.csv_filename = csv_filename
        if max_context_seqs == 0:
            if mutant_bos_token != self.document_token:
                warnings.warn(
                    "Setting self.mutant_bos_token to self.document_token because max_context_seqs is 0"
                )
                self.mutant_bos_token = self.document_token
            # this is necessary because the first completion sequence token cannot be
            # and AA otherwise we can't extract the likelihood for the first AA
        self.print_settings()
        self._tokenizer = tokenizer

        # Build static table of assays to evaluate. We keep rows in-memory as dicts.
        effective_gym_dir = (
            self.gym_data_dir
            if self.gym_data_dir is not None
            else os.path.join("../data", "ProteinGym")
        )
        df = build_gym_df(
            self.dms_ids,
            gym_data_dir=effective_gym_dir,
            use_foldseek_msa=self.use_foldseek_msa,
            max_completion_length=self.max_completion_length,
            msa_folder_name=self.msa_folder_name,
            task_index=self.task_index,
            num_tasks=self.num_tasks,
            csv_filename=self.csv_filename,
        )
        # Store as list of dicts for fast indexing in __getitem__
        self._rows: List[Dict[str, Any]] = df.to_dict("records")

    def __len__(self):
        return len(self._rows)

    def __getitem__(self, idx):
        # Copy row to avoid mutating cached list
        row = dict(self._rows[idx])

        # Prepare MSA context (lazy, per-sample); uses on-disk caches where available
        row = load_msa_for_row(
            row=row,
            seed=self.seed,
            tokenizer=self._tokenizer,  # tokenizer not needed for this step
            max_tokens=self.max_tokens_per_example,
            keep_gaps=self.keep_gaps,
            use_filtered_msa=self.use_filtered_msa,
            extra_tokens_per_document=self.extra_tokens_per_document,
            use_msa_pos=self.use_msa_pos,
            max_context_seqs=self.max_context_seqs,
            keep_wt=self.keep_wt,
            drop_wt=self.drop_wt,
            use_msa_seq_weights=self.use_msa_seq_weights,
        )

        # Load completion (mutant) sequences and positions
        row = load_comp_seq_dms_for_row(
            row=row,
            seed=self.seed,
            tokenizer=self._tokenizer,  # tokenizer not needed for this step
            use_msa_pos=self.use_msa_pos,
            keep_gaps=self.keep_gaps,
            max_mutated_sequences=self.max_mutated_sequences,
        )

        row = tokenize(
            sample=row,
            tokenizer=self._tokenizer,
            mutant_bos_token=self.mutant_bos_token,
            document_token=self.document_token,
        )

        # Select and return the fields expected downstream
        out = {
            "input_ids": row["input_ids"],
            "completion_ids": row["completion_ids"],
            "DMS_scores": row["DMS_scores"],
            "ds_name": row.get("ds_name", "gym"),
            "DMS_id": row["DMS_id"],
        }
        # Optional metadata
        if "sequence_similarities" in row:
            out["sequence_similarities"] = row["sequence_similarities"]
        if "coverages" in row:
            out["coverages"] = row["coverages"]
        if "sequence_weights" in row:
            out["sequence_weights"] = row["sequence_weights"]
        # Ensure metadata fields are numpy arrays so default collate produces tensors
        for _k in (
            "sequence_similarities",
            "coverages",
            "sequence_weights",
            "DMS_scores",
        ):
            if _k in out and out[_k] is not None:
                try:
                    out[_k] = np.asarray(out[_k], dtype=np.float32)
                except Exception:
                    pass
        return out

    @property
    def document_token(self):
        if self.keep_gaps:
            return "[MSA]"
        elif self.use_msa_pos:
            return "[RAW-WITH-MSA-POS]"
        else:
            return "[RAW]"

    def print_settings(self):
        print(f"ProteinGymDataset settings:")
        print(f"  max_context_seqs: {self.max_context_seqs}")
        print(f"  max_tokens_per_example: {self.max_tokens_per_example}")
        print(f"  max_mutated_sequences: {self.max_mutated_sequences}")
        print(f"  keep_gaps: {self.keep_gaps}")
        print(f"  use_filtered_msa: {self.use_filtered_msa}")
        print(f"  keep_wt: {self.keep_wt}")
        print(f"  drop_wt: {self.drop_wt}")
        print(f"  mutant_bos_token: {self.mutant_bos_token}")
        print(f"  document_token: {self.document_token}")
        print(f"  gym_data_dir: {self.gym_data_dir}")
        print(f"  num_proc: {self.num_proc}")
        print(f"  seed: {self.seed}")
        print(f"  extra_tokens_per_document: {self.extra_tokens_per_document}")
        print(f"  use_msa_pos: {self.use_msa_pos}")
        print(f"  max_completion_length: {self.max_completion_length}")
        print(f"  dms_ids: {self.dms_ids}")
        print(f"  msa_folder_name: {self.msa_folder_name}")

    # Deprecated HF-style API retained for backward compatibility, but unused.
    def process(self, *args, **kwargs):  # pragma: no cover
        raise NotImplementedError(
            "ProteinGymDataset is now a PyTorch Dataset; no process()."
        )

    def load(self, *args, **kwargs):  # pragma: no cover
        raise NotImplementedError(
            "ProteinGymDataset is now a PyTorch Dataset; no load()."
        )
