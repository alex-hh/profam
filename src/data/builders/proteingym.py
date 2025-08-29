import functools
import os
import re
import warnings
from typing import List, Optional

import numpy as np
import pandas as pd
from datasets import Dataset
from transformers import PreTrainedTokenizerFast

from src.data.objects import ProteinDocument
from src.data.processors import transforms
from src.data.processors.transforms import (
    preprocess_aligned_sequences_sampling_to_max_tokens,
)
from src.data.tokenizers import ProFamTokenizer
from src.sequence import fasta

from .base import BaseProteinDataset


def has_no_indels(string_list):
    pattern = r"[.\-a-z]"
    return not any(re.search(pattern, s) for s in string_list)

def extract_sequence_weights_from_seq_ids(seq_ids: list) -> np.ndarray[float]:
    return np.array([float(e.split("score=")[-1].split(" ")[0]  ) for e in seq_ids])

def tokenize_msa(
    sample,
    tokenizer: ProFamTokenizer,
    document_token: Optional[str] = "[RAW]",
):
    # todo replace with subsample_and_tokenize_protein_data
    # gym msas don't contain insertions so no need to worry about that and default position indexing is fine
    proteins = ProteinDocument(
        sequences=sample["MSA"],
        residue_positions=sample["seq_pos"],
    )
    tokenized = tokenizer.encode(
        proteins, document_token=document_token, add_final_sep=False
    )  # sep gets added in completion bos
    sample["input_ids"] = tokenized.input_ids.squeeze()
    if tokenizer.embed_residue_index:
        sample["residue_index"] = tokenized.data["residue_index"]
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
        residue_positions=sample["completion_residue_positions"],
        bos_token=get_token_from_name(bos_token, tokenizer),
    )
    sample["completion_ids"] = tokenized.input_ids
    if tokenizer.embed_residue_index:
        sample["completion_residue_index"] = tokenized.data["residue_index"]
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
    if use_msa_seq_weights:
        sequence_weights = extract_sequence_weights_from_seq_ids(seq_ids)
    else:
        sequence_weights = [1 for _ in seqs]
    
    # Load coverage and similarity data if available
    sequence_similarities = None
    coverages = None
    npz_file = os.path.splitext(msa_file)[0] + ".npz"
    print(f"Attempting to load coverage and similarity data from {npz_file}")
    if os.path.exists(npz_file):
        npz_data = np.load(npz_file)
        # Replace any NaN values with 0 before converting to list
        sequence_similarities = np.nan_to_num(
            npz_data["sequence_similarities"], nan=0.0
        ).tolist()
        coverages = np.nan_to_num(npz_data["coverages"], nan=0.0).tolist()
        # Ensure we have the same number of sequences
        if len(sequence_similarities) != len(seqs):
            print(f"Warning: Number of sequences in MSA ({len(seqs)}) doesn't match number in .npz file ({len(sequence_similarities)})")
            sequence_similarities = None
            coverages = None
    seq_indices = [i for i, s in enumerate(seqs) if "X" not in s and "U" not in s and "Z" not in s and "O" not in s and "B" not in s and "J" not in s]
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
        residue_positions=None,
        plddts=None,
        backbone_coords=None,
        structure_tokens=None,
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
    row["seq_pos"] = proteins.residue_positions
    # Also store the coverage and similarity data in the row
    if proteins.sequence_similarities is not None:
        row["sequence_similarities"] = proteins.sequence_similarities
    if proteins.coverages is not None:
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
    assert has_no_indels(completion_seqs), "Comp seq indel handling not implemented"
    proteins = ProteinDocument(
        sequences=completion_seqs,
        accessions=None,
        identifier=None,
        residue_positions=None,
        plddts=None,
        backbone_coords=None,
        structure_tokens=None,
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
    row["completion_residue_positions"] = proteins.residue_positions
    return row


def build_gym_df(
    dms_ids,
    gym_data_dir: str,
    use_foldseek_msa: bool = False,
    max_completion_length: Optional[bool] = None,
    msa_folder_name: str = "DMS_msa_files",
):
    """We pre-load and pre-sample MSAs, ensuring they are same at each validation step."""
    df = pd.read_csv(os.path.join(gym_data_dir, "DMS_substitutions.csv"))
    if dms_ids is not None:
        df = df[df["DMS_id"].isin(dms_ids)].sort_values("DMS_id")
    else:
        print("dms_ids is None so evaluating on all ProteinGym assays")
    if max_completion_length is not None:
        df = df[df["seq_len"] <= max_completion_length]
    if use_foldseek_msa:
        df["MSA_filename"] = df["MSA_filename"].apply(
            lambda x: os.path.join(gym_data_dir, "foldseek_s50_DMS_msa_files", x)
        )
    elif "PoET" in msa_folder_name:
        df["MSA_filename"] = df["DMS_id"].apply(
            lambda x: os.path.join(gym_data_dir, msa_folder_name, x + ".a3m")
        )
    elif "msa_pairformer" in msa_folder_name:
        df["MSA_filename"] = df["MSA_filename"].apply(
            lambda x: os.path.join(gym_data_dir, msa_folder_name, x.split(".")[0] + "_ranked.fasta")
        )
    else:
        df["MSA_filename"] = df["MSA_filename"].apply(
            lambda x: os.path.join(gym_data_dir, msa_folder_name, x)
        )
    assert all(
        os.path.exists(msa_file) for msa_file in df["MSA_filename"]
    ), "MSA files do not exist"
    df["DMS_filename"] = df["DMS_filename"].apply(
        lambda x: os.path.join(gym_data_dir, "DMS_ProteinGym_substitutions", x)
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


class ProteinGymDataset(BaseProteinDataset):
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
        max_completion_length = None,
        keep_wt: bool = False,
        drop_wt: bool = True,
        msa_folder_name: str = "DMS_msa_files",
        use_msa_seq_weights: bool = False,
    ):
        """Thing that's a bit different about Gym (and family classification)
        is that we have this prompt/completions structure.

        We can still use a preprocessor to build the prompt, but we need
        to additionally handle preprocessing of completions.

        We can still train on these datasets - just by setting seed None and
        not setting val dataset name. In this case, model will ignore completions.
        """
        super().__init__(name=name, preprocessor=None)
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
        if max_context_seqs == 0:
            if mutant_bos_token != self.document_token:
                warnings.warn(
                    "Setting self.mutant_bos_token to self.document_token because max_context_seqs is 0"
                )
                self.mutant_bos_token = self.document_token
            # this is necessary because the first completion sequence token cannot be
            # and AA otherwise we can't extract the likelihood for the first AA
        self.print_settings()

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

    def process(
        self,
        dataset: Dataset,
        tokenizer: ProFamTokenizer,
        feature_names: Optional[List[str]] = None,
        **kwargs,
    ):
        """mutant_bos_token should almost always be sep.

        n.b. we just ignore pack_to_max_tokens here.
        """
        remove_columns = [
            "completion_seqs",
            "DMS_filename",
            "MSA_filename",
        ]
        if self.max_context_seqs is None or self.max_context_seqs > 0:
            remove_columns.append("MSA")
            dataset = dataset.map(
                functools.partial(
                    load_msa_for_row,
                    tokenizer=tokenizer,
                    seed=self.seed,  # For what?
                    max_tokens=self.max_tokens_per_example,
                    keep_gaps=self.keep_gaps,
                    use_filtered_msa=self.use_filtered_msa,
                    extra_tokens_per_document=self.extra_tokens_per_document,
                    use_msa_pos=self.use_msa_pos,
                    max_context_seqs=self.max_context_seqs,
                    keep_wt=self.keep_wt,
                    drop_wt=self.drop_wt,
                    use_msa_seq_weights=self.use_msa_seq_weights,
                ),
                batched=False,
                num_proc=self.num_proc,
            )
        dataset = dataset.map(
            functools.partial(
                load_comp_seq_dms_for_row,
                seed=self.seed,
                tokenizer=tokenizer,
                use_msa_pos=self.use_msa_pos,
                keep_gaps=self.keep_gaps,
                max_mutated_sequences=self.max_mutated_sequences,
            ),
            batched=False,
            num_proc=self.num_proc,
        )
        dataset = dataset.map(
            functools.partial(
                tokenize,
                tokenizer=tokenizer,
                mutant_bos_token=self.mutant_bos_token,
                document_token=self.document_token,
            ),
            batched=False,
            remove_columns=remove_columns,
            num_proc=self.num_proc,  # https://huggingface.co/docs/datasets/v2.20.0/en/process#multiprocessing
        )
        # https://discuss.huggingface.co/t/dataset-map-return-only-list-instead-torch-tensors/15767
        columns = ["input_ids", "completion_ids", "DMS_scores", "ds_name", "DMS_id"]

        if tokenizer.embed_residue_index:
            columns += ["residue_index", "completion_residue_index"]

        # Add coverage and similarity fields if they exist in the dataset
        if "sequence_similarities" in dataset.column_names:
            columns.append("sequence_similarities")
        if "coverages" in dataset.column_names:
            columns.append("coverages")
        if "sequence_weights" in dataset.column_names:
            columns.append("sequence_weights")
        # TODO: what is right here?
        dataset.set_format(
            type="torch",
            columns=columns,
        )
        return dataset

    def load(self, data_dir="data", world_size: int = 1, verbose: bool = False):
        df = build_gym_df(
            self.dms_ids,
            gym_data_dir=os.path.join(data_dir, "ProteinGym")
            if self.gym_data_dir is None
            else self.gym_data_dir,
            use_foldseek_msa=self.use_foldseek_msa,
            max_completion_length=self.max_completion_length,
            msa_folder_name=self.msa_folder_name,
        )
        # n.b. this isn't streamed
        dataset = Dataset.from_pandas(df, preserve_index=False)
        return dataset
