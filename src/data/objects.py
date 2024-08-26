from dataclasses import dataclass
from typing import List, Optional

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
    sequences: List[str]
    accessions: Optional[List[str]] = None
    identifier: Optional[str] = None
    positions: Optional[List[List[int]]] = None
    plddts: Optional[List[np.ndarray]] = None
    backbone_coords: Optional[List[np.ndarray]] = None
    backbone_coords_masks: Optional[List[np.ndarray]] = None
    structure_tokens: Optional[List[str]] = None
    validate_shapes: bool = True

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
                np.ones_like(xyz) for xyz in self.backcbone_coords
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
        # print(f"Missing arrays: {missing_arrays_msg}")
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
        )
