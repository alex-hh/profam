from Bio.SVDSuperimposer import SVDSuperimposer
from tmtools import tm_align

import numpy as np
from biotite import structure as struc
from biotite.sequence import ProteinSequence
from Bio import pairwise2

from src.constants import BACKBONE_ATOMS
from src.data.objects import Protein


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


def calc_tm_score(pos_1, pos_2, seq_1, seq_2):
    # TOOD: check whether it requires only ca or this is a choice
    tm_results = tm_align(pos_1, pos_2, seq_1, seq_2)
    return tm_results.tm_norm_chain1, tm_results.tm_norm_chain2


def tm_score(protein_1: Protein, protein_2: Protein):
    """
    Compute the TM-score between two proteins.

    Args:
        protein_1: Protein
        protein_2: Protein
    """
    return calc_tm_score(
        protein_1.backbone_coords[:, 1],
        protein_2.backbone_coords[:, 1],
        protein_1.sequence,
        protein_2.sequence,
    )[0]


def rmsd(ref_prot: Protein, prot: Protein):
    """
    Compute the RMSD between two proteins.

    Args:
        protein_1: Protein
        protein_2: Protein
        align: bool
            If True, align the proteins before computing RMSD.
    """
    assert len(ref_prot) == len(prot)
    _, rmsd = _superimpose_np(
        ref_prot.backbone_coords.reshape((-1, 3)),
        prot.backbone_coords.reshape((-1, 3)),
    )
    return rmsd


def _protein_to_atom_array_with_mask(
    protein: Protein, include_mask_res_atom: np.ndarray, residue_indices: np.ndarray
):
    atoms = []
    for new_res_idx, res_idx in enumerate(residue_indices):
        aa = protein.sequence[res_idx]
        if aa in ["?", "|"]:
            aa = "X"
        res_name = ProteinSequence.convert_letter_1to3(aa)
        for atom_ix, atom_name in enumerate(BACKBONE_ATOMS):
            if include_mask_res_atom[res_idx, atom_ix]:
                coord = protein.backbone_coords[res_idx, atom_ix]
                atoms.append(
                    struc.Atom(
                        coord=coord,
                        chain_id="A",
                        res_id=new_res_idx + 1,
                        res_name=res_name,
                        hetero=False,
                        atom_name=atom_name,
                        element=atom_name[0],
                    )
                )
    return struc.array(atoms)


def lddt(
    reference: Protein,
    subject: Protein,
    use_ca_only: bool = True,
    aggregation: str = "all",
    atom_mask: np.ndarray | None = None,
    partner_mask: np.ndarray | None = None,
    inclusion_radius: float = 15.0,
    distance_bins: tuple[float, float, float, float] = (0.5, 1.0, 2.0, 4.0),
    exclude_same_residue: bool = True,
    exclude_same_chain: bool = False,
    filter_function=None,
    symmetric: bool = False,
    alignment_mode: str = "global",
):
    """
    Compute lDDT between two proteins using Biotite, masking to ensure identical atoms.

    By default, uses CA-only atoms. Set use_ca_only=False to use all backbone atoms.
    """
    assert (
        reference.backbone_coords is not None and subject.backbone_coords is not None
    ), "Both proteins must have backbone coordinates"

    ref_mask = reference.backbone_coords_mask
    subj_mask = subject.backbone_coords_mask
    if ref_mask is None:
        ref_mask = ~np.isnan(reference.backbone_coords)
    if subj_mask is None:
        subj_mask = ~np.isnan(subject.backbone_coords)

    # Align sequences and select aligned residue index pairs (i, j)
    if alignment_mode == "global":
        aligns = pairwise2.align.globalms(
            reference.sequence, subject.sequence, 2, -1, -10, -1, one_alignment_only=True
        )
    elif alignment_mode == "local":
        aligns = pairwise2.align.localms(
            reference.sequence, subject.sequence, 2, -1, -10, -1, one_alignment_only=True
        )
    else:
        raise ValueError("alignment_mode must be 'global' or 'local'")

    if not aligns:
        return float("nan")
    aln_ref, aln_subj = aligns[0][0], aligns[0][1]
    i = j = 0
    index_pairs = []
    for ca, cb in zip(aln_ref, aln_subj):
        if ca != "-" and cb != "-":
            index_pairs.append((i, j))
        if ca != "-":
            i += 1
        if cb != "-":
            j += 1

    if len(index_pairs) == 0:
        return float("nan")

    # Build include masks for both proteins restricted to aligned pairs
    include_mask_ref = np.zeros((len(reference), len(BACKBONE_ATOMS)), dtype=bool)
    include_mask_subj = np.zeros((len(subject), len(BACKBONE_ATOMS)), dtype=bool)

    if use_ca_only:
        for i_ref, j_sub in index_pairs:
            if ref_mask[i_ref, 1].all() and subj_mask[j_sub, 1].all():
                include_mask_ref[i_ref, 1] = True
                include_mask_subj[j_sub, 1] = True
    else:
        for i_ref, j_sub in index_pairs:
            for atom_ix in range(len(BACKBONE_ATOMS)):
                if ref_mask[i_ref, atom_ix].all() and subj_mask[j_sub, atom_ix].all():
                    include_mask_ref[i_ref, atom_ix] = True
                    include_mask_subj[j_sub, atom_ix] = True

    residue_indices_ref = [i for (i, j) in index_pairs if include_mask_ref[i].any()]
    residue_indices_subj = [j for (i, j) in index_pairs if include_mask_subj[j].any()]

    if len(residue_indices_ref) == 0 or len(residue_indices_subj) == 0:
        return float("nan")

    ref_arr = _protein_to_atom_array_with_mask(
        reference, include_mask_ref, np.array(residue_indices_ref)
    )
    subj_arr = _protein_to_atom_array_with_mask(
        subject, include_mask_subj, np.array(residue_indices_subj)
    )

    score = struc.lddt(
        ref_arr,
        subj_arr,
        aggregation=aggregation,
        atom_mask=atom_mask,
        partner_mask=partner_mask,
        inclusion_radius=inclusion_radius,
        distance_bins=distance_bins,
        exclude_same_residue=exclude_same_residue,
        exclude_same_chain=exclude_same_chain,
        filter_function=filter_function,
        symmetric=symmetric,
    )

    # Convert numpy scalar to Python float for convenience
    if isinstance(score, np.ndarray) and score.size == 1:
        return float(score)
    return score
