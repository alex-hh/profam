import functools
import os
import re
from typing import List, Optional

import numpy as np
import pandas as pd
from biotite.structure.residues import get_residue_starts
from datasets import Dataset
from transformers import PreTrainedTokenizerFast

from src.data import transforms
from src.data.datasets import BaseProteinDatasetBuilder
from src.data.objects import ProteinDocument
from src.data.transforms import sample_to_max_tokens
from src.sequence import fasta
from src.structure.pdb import get_atom_coords_residuewise, load_structure
from src.utils.tokenizers import ProFamTokenizer


def has_no_indels(string_list):
    pattern = r"[.\-a-z]"
    return not any(re.search(pattern, s) for s in string_list)


def tokenize_msa(
    sample,
    tokenizer: ProFamTokenizer,
    document_token: Optional[str] = "[RAW]",
):
    # gym msas don't contain insertions so no need to worry about that and default position indexing is fine
    proteins = ProteinDocument(
        sequences=sample["MSA"],
        positions=sample["seq_pos"],
    )
    tokenized = tokenizer.encode(
        proteins, document_token=document_token, add_final_sep=False
    )  # sep gets added in completion bos
    sample["input_ids"] = tokenized.input_ids.squeeze()
    if tokenizer.use_seq_pos:
        sample["seq_pos"] = tokenized.data["seq_pos"]
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
        positions=sample["completion_seq_pos"],
        bos_token=get_token_from_name(bos_token, tokenizer),
    )
    sample["completion_ids"] = tokenized.input_ids
    if tokenizer.use_seq_pos:
        sample["completion_seq_pos"] = tokenized.data["seq_pos"]
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


def load_msa_document(
    msa_filename: str,
):
    labels, seqs = fasta.read_fasta(
        msa_filename,
        keep_insertions=True,
        to_upper=True,
        keep_gaps=True,
    )
    msa = ProteinDocument(
        sequences=seqs,
        accessions=labels,
        representative_accession=labels[0],
    )
    return msa


def load_msa_for_row(
    row,
    seed,
    max_tokens,
    keep_wt=True,
    drop_wt=False,
    keep_gaps=False,
    shuffle: bool = True,
    use_filtered_msa: bool = False,
    extra_tokens_per_document: int = 2,
    use_msa_pos: bool = True,
):
    msa_file = row["MSA_filename"]
    if use_filtered_msa:
        msa_file = msa_file.replace(".a2m", "_reformat_hhfilter.a3m")

    proteins = load_msa_document(msa_file)
    max_tokens_for_msa = max_tokens - max([len(s) for s in proteins.sequences]) - 2
    proteins = sample_to_max_tokens(
        proteins,
        seed=seed,
        drop_first=drop_wt,
        keep_first=keep_wt,
        shuffle=shuffle,
        max_tokens=max_tokens_for_msa,
        extra_tokens_per_document=extra_tokens_per_document,
    )
    proteins = transforms.convert_sequences_adding_positions(
        proteins,
        keep_gaps=keep_gaps,
        keep_insertions=True,  # Gym-specific: Gym MSAs use lower case for non-focus cols not insertions
        to_upper=True,
        use_msa_pos=use_msa_pos,
        truncate_after_n_sequences=None,
    )

    assert len(proteins.sequences) > 0, "No sequences sampled - check max tokens"
    print(f"Sampled {len(proteins.sequences)} sequences for MSA")
    row["MSA"] = proteins.sequences
    row["seq_pos"] = proteins.positions
    return row


def load_completions(
    dms_filename: str,
    max_mutated_sequences: Optional[int] = None,
    seed: Optional[int] = None,
    include_backbone_coords: bool = False,
):
    dms_df = pd.read_csv(dms_filename)
    if max_mutated_sequences is not None and max_mutated_sequences < len(dms_df):
        dms_df = dms_df.sample(n=max_mutated_sequences, random_state=seed)
    if include_backbone_coords:
        uniprot_id = "_".join(os.path.basename(dms_filename).split("_")[:-2])
        pdb_file = os.path.join(
            os.path.dirname(os.path.dirname(dms_filename)),
            "ProteinGym_AF2_structures",
            f"{uniprot_id}.pdb",
        )
        structure = load_structure(
            pdb_file,
            chain="A",
            extra_fields=["b_factor"],
        )
        coords = get_atom_coords_residuewise(
            ["N", "CA", "C", "O"], structure
        )  # residues, atoms, xyz
        plddt = np.array(structure.b_factor[get_residue_starts(structure)])
        # TODO: assert wt sequence matches up
        backbone_coords = [coords] * len(dms_df)
        plddts = [plddt] * len(dms_df)
    else:
        backbone_coords = None
        plddts = None
    completions = ProteinDocument(
        sequences=dms_df["mutated_sequence"].tolist(),
        accessions=dms_df["mutant"].tolist(),
        identifier=None,
        positions=None,
        plddts=plddts,
        backbone_coords=backbone_coords,
        structure_tokens=None,
    )
    assert has_no_indels(
        completions.sequences
    ), "Comp seq indel handling not implemented"
    return completions, dms_df


def load_completions_for_row(
    row,
    seed,
    max_mutated_sequences,
):
    proteins, dms_df = load_completions(
        row["DMS_filename"],
        max_mutated_sequences=max_mutated_sequences,
        seed=seed,
    )
    # TODO: figure out how to handle msa pos etc.
    # for substitutions, use_msa_pos True and False should be the same
    # TODO: this step might not be necessary? We can just trivially add positions...maybe do this within convert_sequences_adding_positions?
    proteins = transforms.convert_sequences_adding_positions(
        proteins,
        keep_gaps=False,  # no gaps in DMS sequences
        keep_insertions=True,  # no insertions in DMS sequences
        to_upper=True,
        use_msa_pos=False,
        truncate_after_n_sequences=None,
    )
    row["DMS_scores"] = dms_df["DMS_score"].tolist()
    row["completion_seqs"] = proteins.sequences
    row["completion_seq_pos"] = proteins.positions
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


class GymDatasetBuilder(BaseProteinDatasetBuilder):
    def __init__(
        self,
        name: str,
        dms_ids: List[str],
        seed: Optional[int] = 42,  # for msa sampling
        max_mutated_sequences: Optional[int] = None,
        mutant_bos_token: str = "sep",
        keep_gaps: bool = False,
        use_filtered_msa: bool = False,
        extra_tokens_per_document: int = 2,
        use_msa_pos: bool = True,
        num_proc: Optional[int] = None,
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

    def process(
        self,
        dataset: Dataset,
        tokenizer: ProFamTokenizer,
        max_tokens_per_example: Optional[int] = None,
        shuffle_proteins_in_document: bool = True,
        feature_names: Optional[List[str]] = None,
    ):
        """mutant_bos_token should almost always be sep.

        when using a BaseSingleSequenceLitModule, however, we want it
        to be bos, since no context sequences are passed during scoring.
        """
        print(f"Processing gym dataset for evaluation, keeping gaps: {self.keep_gaps}")
        dataset = dataset.map(
            functools.partial(
                load_msa_for_row,
                seed=self.seed,  # For what?
                max_tokens=max_tokens_per_example,
                keep_gaps=self.keep_gaps,
                use_filtered_msa=self.use_filtered_msa,
                extra_tokens_per_document=self.extra_tokens_per_document,
                use_msa_pos=self.use_msa_pos,
                shuffle=shuffle_proteins_in_document,
            ),
            batched=False,
            num_proc=self.num_proc,
        )
        dataset = dataset.map(
            functools.partial(
                load_completions_for_row,
                seed=self.seed,
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
        if tokenizer.use_seq_pos:
            columns += ["seq_pos", "completion_seq_pos"]

        dataset.set_format(
            type="torch",
            columns=columns,
        )
        return dataset

    def load(self, data_dir="data", world_size: int = 1, verbose: bool = False):
        df = build_gym_df(
            self.dms_ids,
            gym_data_dir=os.path.join(data_dir, "ProteinGym"),
        )
        # n.b. this isn't streamed
        dataset = Dataset.from_pandas(df, preserve_index=False)
        return dataset
