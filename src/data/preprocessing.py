import functools
import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import numpy as np
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

from src.constants import BASEDIR
from src.data import transforms
from src.data.objects import ProteinDocument
from src.utils.tokenizers import ProFamTokenizer


def load_named_preprocessor(preprocessor_name, overrides: Optional[List[str]] = None):
    with initialize_config_dir(
        os.path.join(BASEDIR, "configs/preprocessor"), version_base="1.3"
    ):
        preprocessor_cfg = compose(config_name=preprocessor_name, overrides=overrides)
    return instantiate(preprocessor_cfg, _convert_="partial")


@dataclass
class PreprocessingConfig:
    keep_insertions: bool = False
    to_upper: bool = False
    keep_gaps: bool = False
    document_token: str = "[RAW]"
    truncate_after_n_sequences: Optional[int] = None
    use_msa_pos: bool = False  # for msa sequences, if true, position index will be relative to alignment cols
    # https://github.com/mit-ll-responsible-ai/hydra-zen/issues/182
    allow_unk: bool = False


def default_transforms(
    max_tokens: Optional[int] = None,
    shuffle: bool = True,
    seed: Optional[int] = None,
):
    return [
        functools.partial(
            transforms.sample_to_max_tokens,
            max_tokens=max_tokens,
            shuffle=shuffle,
            seed=seed,
        ),
        transforms.fill_missing_fields,
        transforms.replace_selenocysteine_pyrrolysine,
    ]


def preprocess_protein_sequences(
    proteins: ProteinDocument,
    cfg: PreprocessingConfig,
    tokenizer: ProFamTokenizer,
):
    assert isinstance(proteins, ProteinDocument), type(proteins)
    # TODO: assert that structure tokens, coords, plddt are all same shape as sequences post conversion or handle if not
    if tokenizer.use_seq_pos:
        proteins = transforms.convert_sequences_adding_positions(
            proteins,
            keep_gaps=cfg.keep_gaps,
            keep_insertions=cfg.keep_insertions,
            to_upper=cfg.to_upper,
            use_msa_pos=cfg.use_msa_pos,
            truncate_after_n_sequences=cfg.truncate_after_n_sequences,
        )
    else:
        proteins = proteins[: cfg.truncate_after_n_sequences or len(proteins)]
    return proteins


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

    def batched_preprocess_protein_data(
        self,
        proteins_list: List[ProteinDocument],
        tokenizer: ProFamTokenizer,
        max_tokens: Optional[int] = None,
        shuffle: bool = True,
    ) -> Dict[str, Any]:
        """
        a batched map is an instruction for converting a set of examples to a
        new set of examples (not necessarily of the same size). it should return a dict whose
        values are lists, where the length of the lists determines the size of the new set of examples.
        """
        transform_fns = default_transforms(max_tokens=max_tokens, shuffle=shuffle)
        transform_fns += self.transform_fns or []
        processed_proteins_list = []
        for proteins in proteins_list:
            proteins = preprocess_protein_sequences(
                proteins,
                self.cfg,
                tokenizer,
            )
            proteins = transforms.apply_transforms(
                transform_fns, proteins, tokenizer, max_tokens=max_tokens
            )
            processed_proteins_list.append(proteins)
        examples = tokenizer.batched_encode(
            processed_proteins_list,
            document_token=self.cfg.document_token,
            padding="max_length",
            max_length=max_tokens,
            add_final_sep=True,
            allow_unk=getattr(self.cfg, "allow_unk", False),
        )
        if "coords" in examples:
            # https://discuss.huggingface.co/t/dataset-map-return-only-list-instead-torch-tensors/15767
            examples["coords"] = [c.tolist() for c in examples["coords"]]
            examples["coords_mask"] = [m.tolist() for m in examples["coords_mask"]]
            if "interleaved_coords_mask" in examples:
                examples["interleaved_coords_mask"] = [
                    m.tolist() for m in examples["interleaved_coords_mask"]
                ]
        return examples

    def preprocess_protein_data(
        self,
        proteins: ProteinDocument,
        tokenizer: ProFamTokenizer,
        max_tokens: Optional[int] = None,
        shuffle: bool = True,
    ) -> Dict[str, Any]:
        # N.B. for stockholm format we need to check that sequences aren't split over
        # multiple lines
        transform_fns = default_transforms(max_tokens=max_tokens, shuffle=shuffle)
        transform_fns += self.transform_fns or []
        proteins = preprocess_protein_sequences(
            proteins,
            self.cfg,
            tokenizer,
        )
        proteins = transforms.apply_transforms(
            transform_fns, proteins, tokenizer, max_tokens=max_tokens
        )
        example = tokenizer.encode(
            proteins,
            document_token=self.cfg.document_token,
            padding="max_length",
            max_length=max_tokens,
            add_final_sep=True,
            allow_unk=getattr(self.cfg, "allow_unk", False),
        ).data
        if max_tokens is not None:
            assert example["input_ids"].shape[-1] <= max_tokens, (
                example["input_ids"].shape[-1],
                max_tokens,
            )
        if "coords" in example:
            # https://discuss.huggingface.co/t/dataset-map-return-only-list-instead-torch-tensors/15767
            example["coords"] = example["coords"].tolist()
            example["coords_mask"] = example["coords_mask"].tolist()
            if "interleaved_coords_mask" in example:
                example["interleaved_coords_mask"] = example[
                    "interleaved_coords_mask"
                ].tolist()
        return example
