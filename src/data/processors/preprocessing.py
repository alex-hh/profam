import functools
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from src.data.objects import ProteinDocument
from src.data.processors import transforms
from src.data.processors.batch_transforms import pack_batches
from src.data.tokenizers import ProFamTokenizer
from src.utils.utils import np_random


@dataclass
class PreprocessingConfig:
    document_token: str = "[RAW]"
    drop_first_protein: bool = False
    keep_first_protein: bool = False
    # https://github.com/mit-ll-responsible-ai/hydra-zen/issues/182
    allow_unk: bool = False
    max_tokens_per_example: Optional[int] = None
    shuffle_proteins_in_document: bool = True
    padding: str = "do_not_pad"  # "longest", "max_length", "do_not_pad"
    defer_sampling: bool = False  # when True, do not sample-to-max in preprocessing


@dataclass
class AlignedProteinPreprocessingConfig(PreprocessingConfig):
    keep_insertions: bool = False
    to_upper: bool = False
    keep_gaps: bool = False
    document_token: str = "[MSA]"
    use_msa_pos: bool = False  # for msa sequences, if true, position index will be relative to alignment cols


def default_transforms(cfg: PreprocessingConfig):
    if isinstance(cfg, AlignedProteinPreprocessingConfig):
        sequence_converter = functools.partial(
            transforms.convert_aligned_sequence_adding_positions,
            keep_gaps=cfg.keep_gaps,
            keep_insertions=cfg.keep_insertions,
            to_upper=cfg.to_upper,
            use_msa_pos=cfg.use_msa_pos,
        )
        if cfg.defer_sampling:
            preprocess_sequences_fn = functools.partial(
                transforms.prepare_aligned_sequences_no_sampling,
                sequence_converter=sequence_converter,
            )
        else:
            preprocess_sequences_fn = functools.partial(
                transforms.preprocess_aligned_sequences_sampling_to_max_tokens,
                max_tokens=cfg.max_tokens_per_example,
                shuffle=cfg.shuffle_proteins_in_document,
                sequence_converter=sequence_converter,
                drop_first=cfg.drop_first_protein,
                keep_first=cfg.keep_first_protein,
            )
    else:
        if cfg.defer_sampling:
            preprocess_sequences_fn = lambda x: x
        else:
            preprocess_sequences_fn = functools.partial(
                transforms.preprocess_raw_sequences_sampling_to_max_tokens,
                max_tokens=cfg.max_tokens_per_example,
                shuffle=cfg.shuffle_proteins_in_document,
                drop_first=cfg.drop_first_protein,
                keep_first=cfg.keep_first_protein,
            )
    return [
        preprocess_sequences_fn,
        transforms.replace_selenocysteine_pyrrolysine,
    ]


class ProteinDocumentPreprocessor:
    """
    Preprocesses protein documents by applying a set of transforms to protein data.
    """

    def __init__(
        self,
        cfg: PreprocessingConfig,  # configures preprocessing of individual proteins
        transform_fns: Optional[List[Callable]] = [],
        structure_first_prob: float = 1.0,
        single_protein_documents: bool = False,
    ):
        self.cfg = cfg
        self.transform_fns = transform_fns
        self.single_protein_documents = single_protein_documents

    def apply_transforms(
        self, proteins, tokenizer, rng: Optional[np.random.Generator] = None
    ):
        transform_fns = default_transforms(self.cfg)
        additional_transform_fns = []
        for partial_fn in self.transform_fns or []:
            additional_transform_fns.append(partial_fn)
        transform_fns += additional_transform_fns

        return transforms.apply_transforms(
            transform_fns,
            proteins,
            tokenizer,
            max_tokens=self.cfg.max_tokens_per_example,
            rng=rng,
        )

    def batched_preprocess_protein_data(
        self,
        proteins_list: List[ProteinDocument],
        tokenizer: ProFamTokenizer,
        pack_to_max_tokens: Optional[int] = None,
        allow_split_packed_documents: bool = False,
    ) -> Dict[str, List[Any]]:
        """
        a batched map is an instruction for converting a set of examples to a
        new set of examples (not necessarily of the same size). it should return a dict whose
        values are lists, where the length of the lists determines the size of the new set of examples.
        """
        if pack_to_max_tokens is not None:
            assert (
                self.cfg.padding == "do_not_pad"
            ), "padding must be do_not_pad if pack_to_max_tokens is used"

        processed_proteins_list = []
        for proteins in proteins_list:
            proteins = self.apply_transforms(proteins, tokenizer)
            processed_proteins_list.append(proteins)
        examples = tokenizer.batched_encode(
            processed_proteins_list,
            document_token=self.cfg.document_token,
            padding=self.cfg.padding,
            max_length=self.cfg.max_tokens_per_example,
            add_final_sep=True,
            allow_unk=getattr(self.cfg, "allow_unk", False),
        )
        if pack_to_max_tokens is not None:
            assert (
                self.cfg.padding == "do_not_pad"
            ), "padding must be do_not_pad if pack_to_max_tokens is used"
            examples = pack_batches(
                examples,
                max_tokens_per_batch=pack_to_max_tokens,
                tokenizer=tokenizer,
                allow_split_packed_documents=allow_split_packed_documents,
            )
        return examples

    def preprocess_protein_data(
        self,
        proteins: ProteinDocument,
        tokenizer: ProFamTokenizer,
    ) -> Dict[str, Any]:
        proteins = self.apply_transforms(proteins, tokenizer)
        example = tokenizer.encode(
            proteins,
            document_token=self.cfg.document_token,
            padding=self.cfg.padding,
            max_length=self.cfg.max_tokens_per_example,
            add_final_sep=True,
            allow_unk=getattr(self.cfg, "allow_unk", False),
        ).data
        if self.cfg.max_tokens_per_example is not None:
            assert example["input_ids"].shape[-1] <= self.cfg.max_tokens_per_example, (
                example["input_ids"].shape[-1],
                self.cfg.max_tokens_per_example,
            )
        return example
