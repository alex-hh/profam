from dataclasses import dataclass
from typing import Callable, ClassVar, List, Optional

import numpy as np

from src.data.fasta import read_fasta_lines


class StringObject:
    """
    Custom class to allow for
    non-tensor elements in batch
    """

    text: List[str]

    def to(self, device):
        return self


@dataclass
class Protein:
    sequence: str
    accession: str
    positions: Optional[List[int]] = None
    plddt: Optional[np.ndarray] = None
    backbone_coords: Optional[np.ndarray] = None
    backbone_coords_mask: Optional[np.ndarray] = None
    structure_tokens: Optional[str] = None
    validate_shapes: bool = True

    def __len__(self):
        assert len(self.sequence) == len(self.plddt)
        return len(self.sequence)

    def __post_init__(self):
        struct_comp = (
            [self.structure_tokens] if self.structure_tokens is not None else None
        )
        if self.validate_shapes:
            check_array_lengths(
                [self.sequence],
                [self.plddt] if self.plddt is not None else None,
                [self.backbone_coords] if self.backbone_coords is not None else None,
                [self.backbone_coords_mask]
                if self.backbone_coords_mask is not None
                else None,
                struct_comp,
            )
        if self.backbone_coords_mask is None and self.backbone_coords is not None:
            self.backbone_coords_mask = np.where(
                np.isnan(self.backbone_coords),
                np.zeros_like(self.backbone_coords),
                np.ones_like(self.backbone_coords),
            )


def check_array_lengths(*arrays):  # TODO: name better!
    sequence_lengths = []
    for arr in arrays:
        if arr is None:
            continue
        else:
            sequence_lengths.append(tuple([len(seq) for seq in arr]))

    assert all(
        l == sequence_lengths[0] for l in sequence_lengths
    ), f"{sequence_lengths} not all equal"
    return sequence_lengths


# want to be consistent with fields in parquet files so we can load from there
# TODO: look into how openai evals uses data classes or similar
# TODO: consider how to represent masks
@dataclass
class ProteinDocument:
    residue_level_fields: ClassVar[List[str]] = [
        "sequences",
        "accessions",
        "plddts",
        "backbone_coords",
        "backbone_coords_masks",
        "structure_tokens",
    ]
    sequences: List[str]
    accessions: Optional[List[str]] = None
    identifier: Optional[str] = None
    positions: Optional[List[List[int]]] = None
    plddts: Optional[List[np.ndarray]] = None
    backbone_coords: Optional[List[np.ndarray]] = None
    backbone_coords_masks: Optional[List[np.ndarray]] = None
    structure_tokens: Optional[List[str]] = None
    validate_shapes: bool = True
    representative_accession: Optional[
        str
    ] = None  # e.g. seed or cluster representative
    original_size: Optional[int] = None  # total number of proteins in original set

    def __len__(self):
        return len(self.sequences)

    @property
    def sequence_lengths(self):
        return [len(seq) for seq in self.sequences]

    def present_fields(self, residue_level_only: bool = False):
        if residue_level_only:
            return [
                field
                for field in self.residue_level_fields
                if getattr(self, field) is not None
            ]
        else:
            return [
                field
                for field in self.__dataclass_fields__.keys()
                if getattr(self, field) is not None
            ]

    @classmethod
    def from_proteins(cls, individual_proteins: List[Protein], **kwargs):
        # N.B. we ignore representative_accession here
        renaming = {
            "sequence": "sequences",
            "accession": "accessions",
            "plddt": "plddts",
            "backbone_coords_mask": "backbone_coords_masks",
        }
        attr_dict = {}
        for attr in cls.residue_level_fields:
            if any(getattr(p, attr) is not None for p in individual_proteins):
                assert all(
                    getattr(p, attr) is not None for p in individual_proteins
                ), f"Missing {attr} for some proteins"
                attr_dict[renaming.get(attr, attr)] = [
                    getattr(p, attr) for p in individual_proteins
                ]
            else:
                attr_dict[renaming.get(attr, attr)] = None
        return cls(
            **attr_dict,
            **kwargs,
        )

    @property
    def representative(self):  # use as target for e.g. inverse folding evaluations
        assert self.seed_accession is not None
        seed_index = self.accessions.index(self.seed_accession)
        return self[seed_index]

    def pop_representative(self):
        assert self.representative_accession is not None
        representative_index = self.accessions.index(self.representative_accession)
        return self.pop(representative_index)

    def filter(self, filter_fn: Callable):
        """Filter by filter_fn.

        Filter_fn should take a protein and return True if it should be kept.
        """
        indices = [i for i in range(len(self)) if filter_fn(self[i])]
        return self[indices]

    def pop(self, index):
        return Protein(
            sequence=self.sequences.pop(index),
            accession=self.accessions.pop(index)
            if self.accessions is not None
            else None,
            positions=self.positions.pop(index) if self.positions is not None else None,
            plddt=self.plddts.pop(index) if self.plddts is not None else None,
            backbone_coords=self.backbone_coords.pop(index)
            if self.backbone_coords is not None
            else None,
            backbone_coords_mask=self.backbone_coords_masks.pop(index)
            if self.backbone_coords_masks is not None
            else None,
            structure_tokens=self.structure_tokens.pop(index)
            if self.structure_tokens is not None
            else None,
            validate_shapes=self.validate_shapes,
        )

    @classmethod
    def from_fasta_str(cls, identifier: str, fasta_str: str):
        lines = fasta_str.split("\n")
        sequences = []
        accessions = []
        for accession, seq in read_fasta_lines(lines):
            sequences.append(seq)
            accessions.append(accession)
        return cls(identifier, sequences, accessions)

    def __post_init__(self):
        if self.validate_shapes:
            check_array_lengths(
                self.sequences,
                self.plddts,
                self.backbone_coords,
                self.backbone_coords_masks,
                self.structure_tokens,
            )
        if self.backbone_coords_masks is None and self.backbone_coords is not None:
            self.backbone_coords_masks = [
                np.ones_like(xyz) for xyz in self.backbone_coords
            ]

    def __getitem__(self, key):
        if isinstance(key, slice):
            return ProteinDocument(
                identifier=self.identifier,
                sequences=self.sequences[key],
                accessions=self.accessions[key]
                if self.accessions is not None
                else None,
                positions=self.positions[key] if self.positions is not None else None,
                plddts=self.plddts[key] if self.plddts is not None else None,
                backbone_coords=self.backbone_coords[key]
                if self.backbone_coords is not None
                else None,
                backbone_coords_masks=self.backbone_coords_masks[key]
                if self.backbone_coords_masks is not None
                else None,
                structure_tokens=self.structure_tokens[key]
                if self.structure_tokens is not None
                else None,
            )
        elif isinstance(key, np.ndarray) or isinstance(key, list):
            return ProteinDocument(
                identifier=self.identifier,
                sequences=[self.sequences[i] for i in key],
                accessions=[self.accessions[i] for i in key]
                if self.accessions is not None
                else None,
                positions=[self.positions[i] for i in key]
                if self.positions is not None
                else None,
                plddts=[self.plddts[i] for i in key]
                if self.plddts is not None
                else None,
                backbone_coords=[self.backbone_coords[i] for i in key]
                if self.backbone_coords is not None
                else None,
                backbone_coords_masks=[self.backbone_coords_masks[i] for i in key]
                if self.backbone_coords_masks is not None
                else None,
                structure_tokens=[self.structure_tokens[i] for i in key]
                if self.structure_tokens is not None
                else None,
            )
        elif isinstance(key, int):
            return Protein(
                sequence=self.sequences[key],
                accession=self.accessions[key] if self.accessions is not None else None,
                positions=self.positions[key] if self.positions is not None else None,
                plddt=self.plddts[key] if self.plddts is not None else None,
                backbone_coords=self.backbone_coords[key]
                if self.backbone_coords is not None
                else None,
                backbone_coords_mask=self.backbone_coords_masks[key]
                if self.backbone_coords_masks is not None
                else None,
                structure_tokens=self.structure_tokens[key]
                if self.structure_tokens is not None
                else None,
            )
        else:
            raise ValueError(f"Invalid key type: {type(key)}")

    def __len__(self):
        return len(self.sequences)

    @property
    def has_all_structure_arrays(self):
        has_arrays = [
            arr is not None
            for arr in [
                self.plddts,
                self.backbone_coords,
                self.backbone_coords_masks,
                self.structure_tokens,
            ]
        ]
        missing_arrays_msg = " ".join(
            [
                f"{name}: {missing}"
                for name, missing in zip(
                    ["plddts", "coords", "coords_masks", "tokens"], has_arrays
                )
            ]
        )
        return all(has_arrays)

    def fill_missing_structure_arrays(
        self, coords_fill=np.nan, plddts_fill=np.nan, tokens_fill="[MASK]"
    ):
        assert isinstance(tokens_fill, str)
        return self.clone(
            plddts=self.plddts
            or [np.full(len(seq), plddts_fill) for seq in self.sequences],
            backbone_coords=self.backbone_coords
            or [np.full((len(seq), 4, 3), coords_fill) for seq in self.sequences],
            structure_tokens=self.structure_tokens
            or [tokens_fill * len(seq) for seq in self.sequences],
            validate_shapes=False,  # because of mask in strs -- TODO: figure out how to deal with this
        )

    def clone(self, **kwargs):
        return ProteinDocument(
            identifier=kwargs.get("identifier", self.identifier),
            sequences=kwargs.get("sequences", self.sequences),
            accessions=kwargs.get("accessions", self.accessions),
            positions=kwargs.get("positions", self.positions),
            plddts=kwargs.get("plddts", self.plddts),
            backbone_coords=kwargs.get("backbone_coords", self.backbone_coords),
            backbone_coords_masks=kwargs.get(
                "backbone_coords_masks", self.backbone_coords_masks
            ),
            structure_tokens=kwargs.get("structure_tokens", self.structure_tokens),
            validate_shapes=kwargs.get("validate_shapes", self.validate_shapes),
            original_size=kwargs.get("original_size", self.original_size),
        )

    def extend(self, proteins: "ProteinDocument"):
        # n.b. extend may be a bad name as this is not in place
        constructor_kwargs = {}
        for field in self.present_fields(residue_level_only=True):
            attr = getattr(self, field)
            if isinstance(attr, list):
                constructor_kwargs[field] = attr + getattr(proteins, field)
            elif isinstance(attr, np.ndarray):
                constructor_kwargs[field] = np.concatenate(
                    [attr, getattr(proteins, field)]
                )
            else:
                raise ValueError(f"Unexpected type: {field} {type(attr)}")
        if self.original_size is not None and proteins.original_size is not None:
            constructor_kwargs["original_size"] = (
                self.original_size + proteins.original_size
            )
        constructor_kwargs["validate_shapes"] = (
            self.validate_shapes and proteins.validate_shapes
        )
        return ProteinDocument(**constructor_kwargs)
