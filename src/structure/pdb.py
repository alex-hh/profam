from typing import List

import biotite
import numpy as np
from biotite.structure import apply_residue_wise, filter_amino_acids, get_chains
from biotite.structure.io import pdb, pdbx

backbone_atoms = ["N", "CA", "C", "O"]


def _filter_atom_names(array, atom_names):
    return np.isin(array.atom_name, atom_names)


def custom_filter_backbone(array):
    """
    Filter all peptide backbone atoms of one array.

    N, CA, C and O

    Parameters
    ----------
    array : AtomArray or AtomArrayStack
        The array to be filtered.

    Returns
    -------
    filter : ndarray, dtype=bool
        This array is `True` for all indices in `array`, where an atom
        is a part of the peptide backbone.
    """

    return _filter_atom_names(array, backbone_atoms) & filter_amino_acids(array)


def get_atom_coords_residuewise(atoms: List[str], struct: biotite.structure.AtomArray):
    """
    Source: esm inverse folding
    Example for atoms argument: ["N", "CA", "C"]
    """

    def filterfn(s, axis=None):
        filters = np.stack([s.atom_name == name for name in atoms], axis=1)
        sum = filters.sum(0)
        if not np.all(sum <= np.ones(filters.shape[1])):
            raise RuntimeError("structure has multiple atoms with same name")
        index = filters.argmax(0)
        coords = s[index].coord
        coords[sum == 0] = float("nan")
        return coords

    return apply_residue_wise(struct, struct, filterfn)


def _load_backbone_structure(structure, chain=None):
    bbmask = custom_filter_backbone(structure)
    # bbmask = filter_backbone(structure)
    structure = structure[bbmask]
    all_chains = get_chains(structure)
    if len(all_chains) == 0:
        raise ValueError("No chains found in the input file.")
    if chain is None:
        chain_ids = all_chains
    elif isinstance(chain, list):
        chain_ids = chain
    else:
        chain_ids = [chain]
    for chain in chain_ids:
        if chain not in all_chains:
            raise ValueError(f"Chain {chain} not found in input file")
    chain_filter = [a.chain_id in chain_ids for a in structure]
    structure = structure[chain_filter]
    return structure


def load_structure(fpath, chain=None, extra_fields=None):
    """
    Modified from esm inverse folding utils to not remove O
    Args:
        fpath: filepath to either pdb or cif file
        chain: the chain id or list of chain ids to load
    Returns:
        biotite.structure.AtomArray
    """
    if isinstance(fpath, str) and fpath.endswith("cif"):
        pdbxf = pdbx.PDBxFile.read(fpath)
        structure = pdbx.get_structure(pdbxf, model=1, extra_fields=extra_fields)
    elif isinstance(fpath, str) and fpath.endswith("pdb"):
        pdbf = pdb.PDBFile.read(fpath)
        structure = pdb.get_structure(pdbf, model=1, extra_fields=extra_fields)
    return _load_backbone_structure(structure, chain=chain)


def load_structure_from_pdb(fin, chain=None, extra_fields=None):
    pdbf = pdb.PDBFile.read(fin)
    structure = pdb.get_structure(pdbf, model=1, extra_fields=extra_fields)
    return _load_backbone_structure(structure, chain=chain)
