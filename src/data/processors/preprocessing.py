import functools
import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import numpy as np
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

from src.constants import BASEDIR
from src.data.objects import ProteinDocument
from src.data.processors import transforms
from src.data.processors.batch_transforms import pack_batches
from src.data.tokenizers import ProFamTokenizer
from src.utils.utils import np_random


def load_named_preprocessor(preprocessor_name, overrides: Optional[List[str]] = None):
    with initialize_config_dir(
        os.path.join(BASEDIR, "configs/preprocessor"), version_base="1.3"
    ):
        preprocessor_cfg = compose(config_name=preprocessor_name, overrides=overrides)
    return instantiate(preprocessor_cfg, _convert_="partial")


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


@dataclass
class AlignedProteinPreprocessingConfig(PreprocessingConfig):
    keep_insertions: bool = False
    to_upper: bool = False
    keep_gaps: bool = False
    document_token: str = "[MSA]"
    use_msa_pos: bool = False  # for msa sequences, if true, position index will be relative to alignment cols


def default_transforms(
    cfg: PreprocessingConfig,
    sequence_converter: Callable,
):
    return [
        functools.partial(
            transforms.preprocess_sequences_sampling_to_max_tokens,
            max_tokens=cfg.max_tokens_per_example,
            shuffle=cfg.shuffle_proteins_in_document,
            sequence_converter=sequence_converter,
            drop_first=cfg.drop_first_protein,
            keep_first=cfg.keep_first_protein,
        ),
        transforms.fill_missing_fields,
        transforms.replace_selenocysteine_pyrrolysine,
    ]


def default_transforms_single_protein(sequence_converter: Callable):
    return [
        functools.partial(
            transforms.preprocess_sequences, sequence_converter=sequence_converter
        ),
        transforms.fill_missing_fields,
        transforms.replace_selenocysteine_pyrrolysine,
    ]


def backbone_coords_from_example(
    example,
    selected_ids: Optional[List[int]] = None,
    sequence_col="sequences",
    use_pdb_if_available_prob: float = 0.0,
):
    ns = example["N"]
    cas = example["CA"]
    cs = example["C"]
    oxys = example["O"]
    sequences = example[sequence_col]
    prot_has_pdb = (
        example["pdb_index_mask"] if "pdb_index_mask" in example else [False] * len(ns)
    )
    coords = []
    is_pdb = []
    if selected_ids is None:
        selected_ids = range(len(ns))

    for ix in selected_ids:
        seq = sequences[ix]
        has_pdb = prot_has_pdb[ix]
        use_pdb = has_pdb and np_random().rand() < use_pdb_if_available_prob

        if use_pdb:
            # I guess test that this is working is that lengths line up
            pdb_index = list(np.argwhere(example["extra_pdb_mask"]).reshape(-1)).index(
                ix
            )
            n = example["pdb_N"][pdb_index]
            ca = example["pdb_CA"][pdb_index]
            c = example["pdb_C"][pdb_index]
            o = example["pdb_O"][pdb_index]
            is_pdb.append(True)
        else:
            n = ns[ix]
            ca = cas[ix]
            c = cs[ix]
            o = oxys[ix]
            is_pdb.append(False)

        recons_coords = np.zeros((len(seq), 4, 3))
        recons_coords[:, 0] = np.array(n).reshape(-1, 3)
        recons_coords[:, 1] = np.array(ca).reshape(-1, 3)
        recons_coords[:, 2] = np.array(c).reshape(-1, 3)
        recons_coords[:, 3] = np.array(o).reshape(-1, 3)
        coords.append(recons_coords)

    return coords, is_pdb


class ProteinDocumentPreprocessor:
    """
    Preprocesses protein documents by applying a set of transforms to protein data.
    """

    def __init__(
        self,
        config: PreprocessingConfig,  # configures preprocessing of individual proteins
        transform_fns: Optional[List[Callable]] = None,
        interleave_structure_sequence: bool = False,
        structure_first_prob: float = 1.0,
        single_protein_documents: bool = False,
    ):
        self.cfg = config
        if interleave_structure_sequence:
            # handle like this because useful to have an interleave_structure_sequence attribute for lenght filtering
            transform_fns = transform_fns or []
            transform_fns.append(
                functools.partial(
                    transforms.interleave_structure_sequence,
                    structure_first_prob=structure_first_prob,
                )
            )
        self.transform_fns = transform_fns
        self.interleave_structure_sequence = (
            interleave_structure_sequence  # should this be part of config?
        )
        self.single_protein_documents = single_protein_documents

    def sequence_converter(self, sequence):
        if isinstance(self.cfg, AlignedProteinPreprocessingConfig):
            return transforms.convert_aligned_sequence_adding_positions(
                sequence,
                keep_gaps=self.cfg.keep_gaps,
                keep_insertions=self.cfg.keep_insertions,
                to_upper=self.cfg.to_upper,
                use_msa_pos=self.cfg.use_msa_pos,
            )
        else:
            return transforms.convert_raw_sequence_adding_positions(sequence)

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
        if self.single_protein_documents:
            # TODO: get rid of this condition
            transform_fns = default_transforms_single_protein(self.sequence_converter)
        else:
            transform_fns = default_transforms(self.cfg, self.sequence_converter)
        transform_fns += self.transform_fns or []
        processed_proteins_list = []
        for proteins in proteins_list:
            proteins = transforms.apply_transforms(
                transform_fns,
                proteins,
                tokenizer,
                max_tokens=self.cfg.max_tokens_per_example,
            )
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
        if self.single_protein_documents:
            # maybe handle padding differently? probably not
            transform_fns = default_transforms_single_protein(self.sequence_converter)
        else:
            transform_fns = default_transforms(self.cfg, self.sequence_converter)
        transform_fns += self.transform_fns or []
        proteins = transforms.apply_transforms(
            transform_fns,
            proteins,
            tokenizer,
            max_tokens=self.cfg.max_tokens_per_example,
        )
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
