from Bio.SVDSuperimposer import SVDSuperimposer
from tmtools import tm_align

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
    assert len(protein_1) == len(protein_2)
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
