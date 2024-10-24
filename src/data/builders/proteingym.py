import functools
import os
import re
from typing import List, Optional

import pandas as pd
from datasets import Dataset
from transformers import PreTrainedTokenizerFast

from src.data.objects import ProteinDocument
from src.data.processors import transforms
from src.data.processors.transforms import preprocess_sequences_sampling_to_max_tokens
from src.data.tokenizers import ProFamTokenizer
from src.sequence import fasta

from .base import BaseProteinDataset


def has_no_indels(string_list):
    pattern = r"[.\-a-z]"
    return not any(re.search(pattern, s) for s in string_list)


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
    else:
        pass


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
    sample = tokenize_msa(
        sample,
        tokenizer,
        document_token=document_token,
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
    keep_wt=True,
    drop_wt=False,
    keep_gaps=False,
    use_filtered_msa: bool = False,
    extra_tokens_per_document: int = 2,
    use_msa_pos: bool = True,
):
    msa_file = row["MSA_filename"]
    if use_filtered_msa:
        msa_file = msa_file.replace(".a2m", "_reformat_hhfilter.a3m")
    _, seqs = fasta.read_fasta(  # initially load without changes for pos calc
        msa_file,
        keep_insertions=True,
        to_upper=True,
        keep_gaps=True if use_msa_pos else keep_gaps,
    )
    proteins = ProteinDocument(
        sequences=seqs,
        accessions=None,
        identifier=row["DMS_id"],
        residue_positions=None,
        plddts=None,
        backbone_coords=None,
        structure_tokens=None,
    )
    # need to allow room for the completion
    # todo should be max completion length (once we handle indels)
    max_tokens_for_msa = max_tokens - max([len(s) for s in seqs]) - 2
    proteins = preprocess_sequences_sampling_to_max_tokens(
        proteins,
        tokenizer=tokenizer,
        seed=seed,
        drop_first=drop_wt,
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

    assert len(proteins.sequences) > 0, "No sequences sampled - check max tokens"
    row["MSA"] = proteins.sequences
    row["seq_pos"] = proteins.residue_positions
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
    proteins = transforms.preprocess_sequences(
        proteins,
        tokenizer,
        sequence_converter=functools.partial(
            transforms.convert_aligned_sequence_adding_positions,
            keep_gaps=keep_gaps,  # no gaps in DMS sequences
            keep_insertions=True,  # no insertions in DMS sequences
            to_upper=True,
            use_msa_pos=use_msa_pos,
        ),
    )
    row["DMS_scores"] = dms_df["DMS_score"].tolist()
    row["completion_seqs"] = proteins.sequences
    row["completion_residue_positions"] = proteins.residue_positions
    return row


def build_gym_df(dms_ids, gym_data_dir: str):
    """We pre-load and pre-sample MSAs, ensuring they are same at each validation step."""
    df = pd.read_csv(os.path.join(gym_data_dir, "DMS_substitutions.csv"))
    df = df[df["DMS_id"].isin(dms_ids)].sort_values("DMS_id")
    df["MSA_filename"] = df["MSA_filename"].apply(
        lambda x: os.path.join(gym_data_dir, "DMS_msa_files", x)
    )
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

    def process(
        self,
        dataset: Dataset,
        tokenizer: ProFamTokenizer,
        feature_names: Optional[List[str]] = None,
        **kwargs,
    ):
        """mutant_bos_token should almost always be sep.

        when using a BaseSingleSequenceLitModule, however, we want it
        to be bos, since no context sequences are passed during scoring.

        n.b. we just ignore pack_to_max_tokens here.
        """
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
                document_token="[MSA]" if self.keep_gaps else "[RAW]",
            ),
            batched=False,
            remove_columns=[
                "DMS_id",
                "MSA",
                "completion_seqs",
                "DMS_filename",
                "MSA_filename",
            ],
            num_proc=self.num_proc,  # https://huggingface.co/docs/datasets/v2.20.0/en/process#multiprocessing
        )
        # https://discuss.huggingface.co/t/dataset-map-return-only-list-instead-torch-tensors/15767
        columns = ["input_ids", "completion_ids", "DMS_scores", "ds_name"]
        if tokenizer.embed_residue_index:
            columns += ["residue_index", "completion_residue_index"]

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
        )
        # n.b. this isn't streamed
        dataset = Dataset.from_pandas(df, preserve_index=False)
        return dataset
