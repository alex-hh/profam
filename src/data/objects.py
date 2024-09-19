import io
import json
import os
from dataclasses import asdict, dataclass
from typing import Callable, ClassVar, List, Optional

import numpy as np
from Bio.SVDSuperimposer import SVDSuperimposer
from biotite import structure as struc
from biotite.sequence import ProteinSequence
from biotite.structure import io as strucio
from biotite.structure.residues import get_residue_starts, get_residues

from src.constants import BACKBONE_ATOMS
from src.sequence.fasta import read_fasta_lines
from src.structure.pdb import get_atom_coords_residuewise, load_structure
from src.tools.foldseek import convert_pdbs_to_3di


# copying here to avoid circular imports
def _superimpose_np(reference, coords):
    """
    Superimposes coordinates onto a reference by minimizing RMSD using SVD.

    Args:
        reference:
            [N, 3] reference array
        coords:
            [N, 3] array
    Returns:
        A tuple of [N, 3] superimposed coords and the final RMSD.
    """
    sup = SVDSuperimposer()
    sup.set(reference, coords)
    sup.run()
    return sup.get_transformed(), sup.get_rms()


def plddt_to_color(plddt):
    if plddt > 90:
        return "#0053D6"
    elif plddt > 70:
        return "#65CBF3"
    elif plddt > 50:
        return "#FFDB13"
    else:
        return "#FF7D45"


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
    accession: Optional[str] = None
    positions: Optional[List[int]] = None
    plddt: Optional[np.ndarray] = None
    backbone_coords: Optional[np.ndarray] = None
    backbone_coords_mask: Optional[np.ndarray] = None
    structure_tokens: Optional[str] = None

    def __len__(self):
        return len(self.sequence)

    def __post_init__(self):
        struct_comp = (
            [self.structure_tokens] if self.structure_tokens is not None else None
        )
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

    def view_with_py3dmol(self, view):
        """view=py3Dmol.view(width=800, height=600)"""
        view.addModel(self.to_pdb_str(), "pdb")
        if self.plddt is not None:
            for i, plddt_val in enumerate(list(self.plddt) * 4):
                color = plddt_to_color(plddt_val)
                view.setStyle(
                    {"model": -1, "serial": i + 1}, {"cartoon": {"color": color}}
                )

    def view_superimposed_with_py3dmol(self, view, other, align: bool = True):
        coords = self.backbone_coords
        other_coords = other.backbone_coords
        assert coords.shape == other_coords.shape
        if align:
            superimposed, rmsd = _superimpose_np(
                coords.reshape((-1, 3)), other_coords.reshape((-1, 3))
            )
            superimposed = superimposed.reshape(other_coords.shape)
            other = other.clone(backbone_coords=superimposed)

        pdb_str = self.to_pdb_str()
        view.addModel(pdb_str, "pdb")
        view.setStyle({"model": -1}, {"cartoon": {"color": "blue"}})
        # view.addModel(other.to_pdb_str(), "pdb")
        # view.setStyle({'model': -1}, {'cartoon': {'color': 'red'}})
        view.addModel(other.to_pdb_str(), "pdb")
        view.setStyle({"model": -1}, {"cartoon": {"color": "red"}})

    def to_pdb_file(self, pdb_file):
        atoms = []
        # TODO: consider saving position information
        for res_ix, (aa, res_coords) in enumerate(
            zip(self.sequence, self.backbone_coords)
        ):
            res_name = ProteinSequence.convert_letter_1to3(aa)
            for atom_ix, atom_name in enumerate(BACKBONE_ATOMS):
                annots = (
                    {"b_factor": self.plddt[res_ix]} if self.plddt is not None else {}
                )
                atom = struc.Atom(
                    coord=res_coords[atom_ix],
                    chain_id="A",
                    res_id=res_ix + 1,
                    res_name=res_name,
                    hetero=False,
                    atom_name=atom_name,
                    element=atom_name[0],
                    **annots,
                )
                atoms.append(atom)
        arr = struc.array(atoms)
        pdb = strucio.pdb.PDBFile()
        pdb.set_structure(arr)
        pdb.write(pdb_file)

    def to_pdb_str(self):
        pdb_file_like = io.StringIO()
        self.to_pdb_file(pdb_file_like)
        return pdb_file_like.getvalue()

    @classmethod
    def from_pdb(cls, pdb_file, chain=None, bfactor_is_plddt=False, load_3di=False):
        # TODO: check chain handled correctly
        structure = load_structure(
            pdb_file,
            chain=chain,
            extra_fields=["b_factor"] if bfactor_is_plddt else None,
        )
        coords = get_atom_coords_residuewise(
            ["N", "CA", "C", "O"], structure
        )  # residues, atoms, xyz
        residue_identities = get_residues(structure)[1]
        seq = "".join(
            [ProteinSequence.convert_letter_3to1(r) for r in residue_identities]
        )
        if bfactor_is_plddt:
            plddt = np.array(structure.b_factor[get_residue_starts(structure)])
        else:
            plddt = None
        if load_3di:
            structure_tokens = convert_pdbs_to_3di([pdb_file])[0]
        else:
            structure_tokens = None
        return cls(
            sequence=seq,
            accession=os.path.splitext(os.path.basename(pdb_file))[0],
            positions=None,
            plddt=plddt,
            backbone_coords=coords,
            backbone_coords_mask=None,  # TODO: for cif files we can get mask - c.f. evogvp
            structure_tokens=structure_tokens,
        )

    def clone(self, **kwargs):
        return Protein(
            sequence=kwargs.get("sequence", self.sequence),
            accession=kwargs.get("accession", self.accession),
            positions=kwargs.get("positions", self.positions),
            plddt=kwargs.get("plddt", self.plddt),
            backbone_coords=kwargs.get("backbone_coords", self.backbone_coords),
            backbone_coords_mask=kwargs.get(
                "backbone_coords_mask", self.backbone_coords_mask
            ),
            structure_tokens=kwargs.get("structure_tokens", self.structure_tokens),
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


def convert_list_of_arrays_to_list_of_lists(list_of_arrays):
    if list_of_arrays is None:
        return None
    elif isinstance(list_of_arrays[0], np.ndarray):
        return [arr.tolist() for arr in list_of_arrays]
    else:
        return list_of_arrays


# want to be consistent with fields in parquet files so we can load from there
# TODO: look into how openai evals uses data classes or similar
# TODO: consider how to represent masks
@dataclass
class ProteinDocument:
    # TODO: make this a mapping?
    # fields that are present on individual protein instances
    protein_fields: ClassVar[List[str]] = [
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
    interleaved_coords_masks: Optional[
        List[np.ndarray]
    ] = None  # if interleaving, indicates which coords are available at each sequence position
    structure_tokens: Optional[List[str]] = None
    # L x 2, boolean mask for modality (0: sequence, 1: structure)
    # really tells us about what we are predicting: we could condition on e.g. sequence within interleaved structure.
    modality_masks: Optional[np.ndarray] = None
    representative_accession: Optional[
        str
    ] = None  # e.g. seed or cluster representative
    original_size: Optional[int] = None  # total number of proteins in original set

    def __post_init__(self):
        for field in [
            "plddts",
            "backbone_coords",
            "backbone_coords_masks",
            "interleaved_coords_masks",
            "modality_masks",
        ]:
            attr = getattr(self, field)
            if attr is not None and isinstance(attr[0], list):
                setattr(self, field, [np.array(arr) for arr in getattr(self, field)])

        if self.modality_masks is None:
            assert (
                self.interleaved_coords_masks is None
            ), "Must pass modality masks if interleaved coords are present"
            sequences_masks = [np.ones(len(seq)) for seq in self.sequences]
            has_struct = (
                self.structure_tokens is not None or self.backbone_coords is not None
            )
            structure_masks = [
                np.ones(len(seq)) if has_struct else np.zeros(len(seq))
                for seq in self.sequences
            ]
            self.modality_masks = [
                np.stack([seq_mask, struct_mask], axis=1).astype(bool)
                for seq_mask, struct_mask in zip(sequences_masks, structure_masks)
            ]

        check_array_lengths(
            self.sequences,
            self.plddts,
            self.backbone_coords,
            self.backbone_coords_masks,
            self.structure_tokens,
            self.interleaved_coords_masks,
            self.modality_masks,
        )
        if self.backbone_coords_masks is None and self.backbone_coords is not None:
            self.backbone_coords_masks = [
                np.ones_like(xyz) for xyz in self.backbone_coords
            ]

    def __len__(self):
        return len(self.sequences)

    @property
    def sequence_lengths(self):
        return [len(seq) for seq in self.sequences]

    def present_fields(self, residue_level_only: bool = False):
        if residue_level_only:
            return [
                field
                for field in self.protein_fields
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
        reverse_naming = {v: k for k, v in renaming.items()}
        attr_dict = {}
        for field in cls.protein_fields:
            single_field = reverse_naming.get(field, field)
            if any(getattr(p, single_field) is not None for p in individual_proteins):
                assert all(
                    getattr(p, single_field) is not None for p in individual_proteins
                ), f"Missing {single_field} for some proteins"
                attr_dict[field] = [
                    getattr(p, single_field) for p in individual_proteins
                ]
            else:
                attr_dict[field] = None
        return cls(
            **attr_dict,
            **kwargs,
        )

    @classmethod
    def from_json(cls, json_file, strict: bool = False):
        with open(json_file, "r") as f:
            protein_dict = json.load(f)

        if strict:
            assert all(
                field in protein_dict for field in cls.__dataclass_fields__.keys()
            ), f"Missing fields in {json_file}"
        return cls(**protein_dict)

    def to_json(self, json_file):
        with open(json_file, "w") as f:
            protein_dict = {
                k: convert_list_of_arrays_to_list_of_lists(v)
                for k, v in asdict(self).items()
            }
            json.dump(protein_dict, f)

    @property
    def representative(self):  # use as target for e.g. inverse folding evaluations
        assert self.representative_accession is not None
        rep_index = self.accessions.index(self.representative_accession)
        return self[rep_index]

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
                modality_masks=self.modality_masks[key]
                if self.modality_masks is not None
                else None,
                representative_accession=self.representative_accession,
                original_size=self.original_size,
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
                representative_accession=self.representative_accession,
                original_size=self.original_size,
                modality_masks=[self.modality_masks[i] for i in key]
                if self.modality_masks is not None
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

    def slice_arrays(self, slices):
        assert len(slices) == len(self.sequences)
        return ProteinDocument(
            identifier=self.identifier,
            sequences=[seq[s] for seq, s in zip(self.sequences, slices)],
            accessions=self.accessions,
            positions=[pos[s] for pos, s in zip(self.positions, slices)]
            if self.positions is not None
            else None,
            plddts=[plddt[s] for plddt, s in zip(self.plddts, slices)]
            if self.plddts is not None
            else None,
            backbone_coords=[xyz[s] for xyz, s in zip(self.backbone_coords, slices)]
            if self.backbone_coords is not None
            else None,
            backbone_coords_masks=[
                mask[s] for mask, s in zip(self.backbone_coords_masks, slices)
            ]
            if self.backbone_coords_masks is not None
            else None,
            structure_tokens=[
                tokens[s] for tokens, s in zip(self.structure_tokens, slices)
            ]
            if self.structure_tokens is not None
            else None,
            representative_accession=self.representative_accession,
            original_size=self.original_size,
            modality_masks=[mask[s] for mask, s in zip(self.modality_masks, slices)]
            if self.modality_masks is not None
            else None,
            interleaved_coords_masks=[
                mask[s] for mask, s in zip(self.interleaved_coords_masks, slices)
            ]
            if self.interleaved_coords_masks is not None
            else None,
        )

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
        self, coords_fill=np.nan, plddts_fill=np.nan, tokens_fill="?"
    ):
        assert isinstance(tokens_fill, str)
        return self.clone(
            plddts=self.plddts
            or [np.full(len(seq), plddts_fill) for seq in self.sequences],
            backbone_coords=self.backbone_coords
            or [np.full((len(seq), 4, 3), coords_fill) for seq in self.sequences],
            structure_tokens=self.structure_tokens
            or [tokens_fill * len(seq) for seq in self.sequences],
        )

    def clone(self, **kwargs):
        return ProteinDocument(
            identifier=kwargs.get("identifier", self.identifier),
            sequences=kwargs.get("sequences", self.sequences.copy()),
            accessions=kwargs.get(
                "accessions",
                self.accessions.copy() if self.accessions is not None else None,
            ),
            positions=kwargs.get(
                "positions",
                self.positions.copy() if self.positions is not None else None,
            ),
            plddts=kwargs.get(
                "plddts", self.plddts.copy() if self.plddts is not None else None
            ),
            backbone_coords=kwargs.get(
                "backbone_coords",
                self.backbone_coords.copy()
                if self.backbone_coords is not None
                else None,
            ),
            backbone_coords_masks=kwargs.get(
                "backbone_coords_masks",
                self.backbone_coords_masks.copy()
                if self.backbone_coords_masks is not None
                else None,
            ),
            interleaved_coords_masks=kwargs.get(
                "interleaved_coords_masks",
                self.interleaved_coords_masks.copy()
                if self.interleaved_coords_masks is not None
                else None,
            ),
            structure_tokens=kwargs.get(
                "structure_tokens",
                self.structure_tokens.copy()
                if self.structure_tokens is not None
                else None,
            ),
            representative_accession=kwargs.get(
                "representative_accession", self.representative_accession
            ),
            original_size=kwargs.get("original_size", self.original_size),
            modality_masks=kwargs.get(
                "modality_masks",
                self.modality_masks.copy() if self.modality_masks is not None else None,
            ),
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
        return ProteinDocument(**constructor_kwargs)
