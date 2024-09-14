import functools
import os
import re
from typing import List, Optional

import pandas as pd
from datasets import Dataset
from lightning import LightningDataModule
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerFast

from src.data import fasta, transforms
from src.data.datasets import ProteinDatasetConfig, load_protein_dataset
from src.data.objects import ProteinDocument
from src.data.transforms import sample_to_max_tokens
from src.data.utils import CustomDataCollator
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


def load_msa_for_row(
    row,
    seed,
    max_tokens,
    gym_data_dir,
    keep_wt=True,
    drop_wt=False,
    keep_gaps=False,
    use_filtered_msa: bool = False,
    extra_tokens_per_document: int = 2,
    use_msa_pos: bool = True,
):
    msa_file = os.path.join(gym_data_dir, "DMS_msa_files", row["MSA_filename"])
    if use_filtered_msa:
        msa_file = msa_file.replace(".a2m", "_reformat_hhfilter.a3m")
    _, seqs = fasta.read_fasta(  # initially load without changes for pos calc
        msa_file,
        keep_insertions=True,
        to_upper=True,
        keep_gaps=True if use_msa_pos else keep_gaps,
    )
    # need to allow room for the completion
    # todo should be max completion length (once we handle indels)
    max_tokens_for_msa = max_tokens - max([len(s) for s in seqs]) - 2
    proteins = ProteinDocument(
        sequences=seqs,
        accessions=None,
        identifier=None,
        positions=None,
        plddts=None,
        backbone_coords=None,
        structure_tokens=None,
        validate_shapes=True,
    )
    proteins = sample_to_max_tokens(
        proteins,
        seed=seed,
        drop_first=drop_wt,
        keep_first=keep_wt,
        max_tokens=max_tokens_for_msa,
        extra_tokens_per_document=extra_tokens_per_document,
    )

    proteins = transforms.convert_sequences_adding_positions(
        proteins,
        keep_gaps=keep_gaps,
        keep_insertions=True,
        to_upper=True,
        use_msa_pos=use_msa_pos,
        truncate_after_n_sequences=None,
    )

    assert len(proteins.sequences) > 0, "No sequences sampled - check max tokens"
    print(f"Sampled {len(proteins.sequences)} sequences for MSA")
    row["MSA"] = proteins.sequences
    row["seq_pos"] = proteins.positions
    return row


def load_comp_seq_dms_for_row(
    row,
    seed,
    max_mutated_sequences,
    gym_data_dir,
    use_msa_pos: bool = True,
    keep_gaps: bool = False,
):

    dms_df = pd.read_csv(
        os.path.join(gym_data_dir, "DMS_ProteinGym_substitutions", row["DMS_filename"])
    )
    if max_mutated_sequences is not None and max_mutated_sequences < len(dms_df):
        dms_df = dms_df.sample(n=max_mutated_sequences, random_state=seed)
    completion_seqs = dms_df["mutated_sequence"].tolist()
    assert has_no_indels(completion_seqs), "Comp seq indel handling not implemented"
    proteins = ProteinDocument(
        sequences=completion_seqs,
        accessions=None,
        identifier=None,
        positions=None,
        plddts=None,
        backbone_coords=None,
        structure_tokens=None,
        validate_shapes=True,
    )
    proteins = transforms.convert_sequences_adding_positions(
        proteins,
        keep_gaps=keep_gaps,  # no gaps in DMS sequences
        keep_insertions=True,  # no insertions in DMS sequences
        to_upper=True,
        use_msa_pos=use_msa_pos,
        truncate_after_n_sequences=None,
    )
    row["DMS_scores"] = dms_df["DMS_score"].tolist()
    row["completion_seqs"] = proteins.sequences
    row["completion_seq_pos"] = proteins.positions
    return row


def build_gym_structure_prompt_df():
    pass


def build_gym_df(
    dms_ids,
    gym_data_dir: str,
    seed: Optional[int] = None,
    max_mutated_sequences: Optional[int] = None,
    max_tokens: int = 5000,
    keep_gaps: bool = False,
    use_filtered_msa: bool = False,
    extra_tokens_per_document: int = 2,
    use_msa_pos: bool = True,
):
    """We pre-load and pre-sample MSAs, ensuring they are same at each validation step."""
    df = pd.read_csv(os.path.join(gym_data_dir, "DMS_substitutions.csv"))
    df = df[df["DMS_id"].isin(dms_ids)].sort_values("DMS_id")
    df = df.apply(
        load_msa_for_row,
        axis=1,
        seed=seed,
        gym_data_dir=gym_data_dir,
        max_tokens=max_tokens,
        keep_gaps=keep_gaps,
        use_filtered_msa=use_filtered_msa,
        extra_tokens_per_document=extra_tokens_per_document,
        use_msa_pos=use_msa_pos,
        keep_wt=True,
        drop_wt=False,
    )
    df = df.apply(
        load_comp_seq_dms_for_row,
        axis=1,
        seed=seed,
        max_mutated_sequences=max_mutated_sequences,
        gym_data_dir=gym_data_dir,
        use_msa_pos=use_msa_pos,
    )
    df["ds_name"] = "gym"
    return df[
        [
            "DMS_id",
            "MSA",
            "seq_pos",
            "DMS_scores",
            "completion_seqs",
            "completion_seq_pos",
            "ds_name",
        ]
    ]


def load_gym_dataset(
    dms_ids,
    tokenizer,
    seed: Optional[int] = None,
    max_mutated_sequences: Optional[int] = None,
    max_tokens: int = 5000,
    mutant_bos_token: str = "sep",
    gym_data_dir: str = "data/example_data/ProteinGym",
    keep_gaps: bool = False,
    num_proc: Optional[int] = None,
    use_filtered_msa: bool = False,
    use_msa_pos: bool = True,
):
    """mutant_bos_token should almost always be sep.

    when using a BaseSingleSequenceLitModule, however, we want it
    to be bos, since no context sequences are passed during scoring.
    """
    print(f"Loading gym dataset for evaluation, keeping gaps: {keep_gaps}")
    if num_proc == 0:
        num_proc = None
    df = build_gym_df(
        dms_ids,
        gym_data_dir=gym_data_dir,
        seed=seed,
        max_mutated_sequences=max_mutated_sequences,
        max_tokens=max_tokens,
        keep_gaps=keep_gaps,
        use_filtered_msa=use_filtered_msa,
        extra_tokens_per_document=tokenizer.num_start_tokens,
        use_msa_pos=use_msa_pos,
    )
    dataset = Dataset.from_pandas(df, preserve_index=False)
    print("Loading gym dataset")
    dataset = dataset.map(
        functools.partial(
            tokenize,
            tokenizer=tokenizer,
            mutant_bos_token=mutant_bos_token,
            document_token="[MSA]" if keep_gaps else "[RAW]",
        ),
        batched=False,
        remove_columns=["DMS_id", "MSA", "completion_seqs"],
        num_proc=num_proc,  # https://huggingface.co/docs/datasets/v2.20.0/en/process#multiprocessing
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


def load_gym_msa_dataset(
    dms_id,
    tokenizer,
    gym_data_dir: str = "data/example_data/ProteinGym",
    keep_gaps: bool = True,
    num_proc: Optional[int] = None,
):
    """For single-sequence training."""
    if num_proc == 0:
        num_proc = None
    df = pd.read_csv(os.path.join(gym_data_dir, "DMS_substitutions.csv"))
    row = df[df["DMS_id"] == dms_id].iloc[0]
    _, seqs = fasta.read_fasta(
        os.path.join(gym_data_dir, "DMS_msa_files", row["MSA_filename"]),
        keep_insertions=True,
        to_upper=True,
        keep_gaps=keep_gaps,
    )
    dataset = Dataset.from_dict({"sequence": seqs})

    def tokenize_sequence(example):
        # prepend sep token to ensure data lines up with completion ids
        if isinstance(example["sequence"], str):
            return tokenizer(
                tokenizer.sep_token + example["sequence"] + tokenizer.sep_token,
                return_tensors="pt",
            )
        else:
            return tokenizer(
                [
                    tokenizer.sep_token + s + tokenizer.sep_token
                    for s in example["sequence"]
                ],
                return_tensors="pt",
            )

    dataset = dataset.map(
        tokenize_sequence,
        batched=True,
        remove_columns=["sequence"],
        num_proc=num_proc,
    )
    return dataset


class GymSingleMSADataModule(LightningDataModule):
    """For training on a single protein gym MSA."""

    def __init__(
        self,
        tokenizer: ProFamTokenizer,
        gym_dms_id: str,
        gym_data_dir: str,
        batch_size: int,
        max_gym_sequences: Optional[int] = None,
        num_workers: int = 0,
        keep_gaps: bool = True,
        use_seq_pos: bool = False,
    ):
        super().__init__()
        self.gym_data_dir = gym_data_dir
        self.batch_size = batch_size
        self.max_gym_sequences = max_gym_sequences
        self.gym_dms_id = gym_dms_id
        self.num_workers = num_workers
        self.keep_gaps = keep_gaps
        self.use_seq_pos = use_seq_pos
        self.tokenizer = tokenizer
        self.collator = CustomDataCollator(self.tokenizer, mlm=False)
        # TODO: fix to avoid hardcoding
        assert self.gym_dms_id is not None
        assert self.gym_data_dir is not None
        self.gym_dataset = load_gym_dataset(
            dms_ids=[gym_dms_id],
            tokenizer=self.tokenizer,
            max_mutated_sequences=self.max_gym_sequences,
            gym_data_dir=self.gym_data_dir,
            num_proc=self.num_workers,
            keep_gaps=self.keep_gaps,
        )
        self.msa_dataset = load_gym_msa_dataset(
            dms_id=gym_dms_id,
            tokenizer=self.tokenizer,
            gym_data_dir=self.gym_data_dir,
            keep_gaps=self.keep_gaps,
        )
        ddict = self.msa_dataset.train_test_split(test_size=0.01, seed=42)
        self.train_dataset = ddict["train"]
        self.val_dataset = ddict["test"]

    def train_dataloader(self) -> List[DataLoader]:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            collate_fn=self.collator,
            num_workers=self.num_workers,
            persistent_workers=True,
            shuffle=True,
        )

    def val_dataloader(self) -> List[DataLoader]:
        loaders = [
            DataLoader(
                self.val_dataset,
                batch_size=self.batch_size,
                collate_fn=self.collator,
                shuffle=False,
                num_workers=self.num_workers,
            )
        ]
        loaders.append(
            [
                DataLoader(
                    self.gym_dataset,
                    batch_size=1,  # gym needs batch size 1
                    shuffle=False,
                )  # n.b. in this case we do standard collation
            ]
        )
        return loaders

    def test_dataloader(self) -> List[DataLoader]:
        loaders = [
            DataLoader(
                self.test_dataset,
                batch_size=self.batch_size,
                collate_fn=self.collator,
                shuffle=False,
                num_workers=self.num_workers,
            )
        ]
        loaders.append(
            [
                DataLoader(
                    self.gym_dataset,
                    batch_size=1,  # gym needs batch size 1
                    shuffle=False,
                )  # n.b. in this case we do standard collation
            ]
        )
        return loaders


class GymMultiMSADataModule(LightningDataModule):
    """For training on multiple protein gym MSAs.

    Idea here is going to be to concatenate sequences from a single MSA up to
    the max tokens limit, then tokenize.

    This data module is therefore compatible with both single-sequence models
    and multi-sequence models. It's also compatible with single MSA training and
    multi-MSA training.

    TODO: could this be unified with the hfdatamodule?
    """

    def __init__(
        self,
        dataset_cfg: ProteinDatasetConfig,
        val_dataset_cfg: ProteinDatasetConfig,
        tokenizer: ProFamTokenizer,
        gym_dms_ids: str,
        gym_data_dir: str,
        data_dir: str,
        batch_size: int,
        max_tokens: int,
        max_gym_sequences: Optional[int] = None,
        num_workers: int = 0,
        # when using a single sequence model (BaseSingleSequenceLitModule), it
        # scoring passes as input to the model only the completion ids. Therefore
        # the completion ids should have the bos token at the start.
        # n.b. during training the model might nonetheless receive multiple concatenated
        # sequences
        mutant_bos_token: str = "sep",
        # will allow sampling multiple times from same dataset.
    ):
        super().__init__()
        self.gym_data_dir = gym_data_dir
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.max_gym_sequences = max_gym_sequences
        self.max_tokens = max_tokens
        self.gym_dms_ids = gym_dms_ids
        self.num_workers = num_workers
        self.tokenizer = tokenizer
        self.collator = CustomDataCollator(self.tokenizer, mlm=False)
        # TODO: fix to avoid hardcoding
        assert self.gym_dms_ids is not None
        assert self.gym_data_dir is not None
        self.gym_dataset = load_gym_dataset(
            dms_ids=gym_dms_ids,
            tokenizer=self.tokenizer,
            max_mutated_sequences=self.max_gym_sequences,
            gym_data_dir=self.gym_data_dir,
            mutant_bos_token=mutant_bos_token,  # we might want to set to bos
            max_tokens=max_tokens,
            num_proc=self.num_workers,
            keep_gaps=dataset_cfg.keep_gaps,
        )
        self.train_dataset = load_protein_dataset(
            dataset_cfg,
            tokenizer=self.tokenizer,
            max_tokens=self.max_tokens,
            data_dir=self.data_dir,
        )
        self.train_dataset = self.train_dataset.shuffle(
            buffer_size=self.train_dataset.n_shards // dataset_cfg.file_repeats,
            seed=42,
        )
        # TODO: fix so that train, val, test aren't all the same
        self.val_dataset = load_protein_dataset(
            val_dataset_cfg,
            tokenizer=self.tokenizer,
            max_tokens=self.max_tokens,
            data_dir=self.data_dir,
        )
        self.test_dataset = load_protein_dataset(
            val_dataset_cfg,
            tokenizer=self.tokenizer,
            max_tokens=self.max_tokens,
            data_dir=self.data_dir,
        )

    def train_dataloader(self) -> List[DataLoader]:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            collate_fn=self.collator,
            num_workers=self.num_workers,
        )

    def val_dataloader(self) -> List[DataLoader]:
        loaders = [
            DataLoader(
                self.val_dataset,
                batch_size=self.batch_size,
                collate_fn=self.collator,
                shuffle=False,
                num_workers=self.num_workers,
            )
        ]
        loaders.append(
            [
                DataLoader(
                    self.gym_dataset,
                    batch_size=1,  # gym needs batch size 1
                    shuffle=False,
                )  # n.b. in this case we do standard collation
            ]
        )
        return loaders

    def test_dataloader(self) -> List[DataLoader]:
        loaders = [
            DataLoader(
                self.test_dataset,
                batch_size=self.batch_size,
                collate_fn=self.collator,
                shuffle=False,
                num_workers=self.num_workers,
            )
        ]
        loaders.append(
            [
                DataLoader(
                    self.gym_dataset,
                    batch_size=1,  # gym needs batch size 1
                    shuffle=False,
                )  # n.b. in this case we do standard collation
            ]
        )
        return loaders
